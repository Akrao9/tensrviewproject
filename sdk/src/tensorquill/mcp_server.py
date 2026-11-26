"""A read-only MCP server over TensorQuill's evidence files, for a team's coding agent.

    python -m tensorquill.mcp_server PATH [--records CAPTURE] [--runs FOLDER]

`PATH` is an `evidence.json`, a `.jsonl` capture from `tensorquill.init`, or a run folder in a
`tensorquill-server` store (`runs/<run_id>`). The server speaks the Model Context Protocol
over stdio and offers six tools:

    list_incidents         what the detectors found, with the run state and what was not checked
    get_incident           one incident in full
    get_series             one layer's, parameter's, router's, logged scalar's, or the probe's head's or
                           attention module's values over a step range
    get_all_ranks_status   every rank of a run folder: judged, unknown, clean, with incidents
    list_runs              the runs kept beside this one, by project
    compare_runs           two or more of those runs: setup, what was checked, incidents

The first four answer about the run at `PATH`. The last two read the folder that run is kept
in (`--runs` names another): the folder of a capture or evidence file, or the `runs/` folder
of the store for a run folder. They take run IDs and names found there and never a path, so a
tool argument cannot point the server at another part of the disk.

It reads the path again at every call, so it follows a run that is still going. It writes no
file, starts no process and opens no connection, and nothing in it can pause, stop or change a
run. An answer is at most `MAX_ANSWER_BYTES` long: a series is thinned to `DEFAULT_POINTS`
points with the stride reported, and anything else that is too long is cut, with a note of
where.

The part of the protocol the server needs is written out here, so that the SDK needs no
package beyond `torch`: JSON-RPC 2.0, one message a line, with `initialize`,
`notifications/initialized`, `ping`, `tools/list` and `tools/call`, as revision 2025-06-18
of the Model Context Protocol specifies them.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import math
import os
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from . import __version__
from . import compact as _compact
from . import compare as _compare
from . import manifest as _manifest
from . import runs as _runs
from .identity import MalformedRecordError, check as check_identity, in_file, is_scalar_record
from .sink import read_end_marker
from .evidence import (
    ALL_RANKS_VIEW,
    ATTN_KINDS,
    HEAD_NUMBERS,
    NO_PEER_WORDS,
    SERVER_KINDS,
    _attention_values,
    _discover_rank_folders,
    _parquet_parts,
    _router_total,  # the detectors' rule for a router's counts on a record (wave AP, AO-05)
    rank_number,  # the one reader of `rank=<n>`: ASCII digits (lane X5, DET-17: `isdecimal()` took `rank=٣`)
    all_ranks_evidence,
    all_ranks_kinds,
    attempts_first_seen,
    build_evidence,
    check_one_run,
    incident_class,
    incident_summary,
    kinds_by_value,
    load_records,
    merged_scalar_points,
    read_watchdog,
    resolve_attempts,
    scalar_points_in_order,
    set_apart_scalars,
)

SERVER_NAME = "tensorquill"
# The revision this file follows, and the older ones whose `initialize`, `ping`, `tools/list`
# and `tools/call` are the same. A client that asks for any other gets PROTOCOL_VERSION back.
PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = (PROTOCOL_VERSION, "2025-03-26", "2024-11-05")

MAX_ANSWER_BYTES = 50_000  # one whole reply line, so a long run cannot flood an agent's context
DEFAULT_POINTS = 200
MAX_POINTS = 1000
DEFAULT_INCIDENTS = 50
MAX_INCIDENTS = 200
WHERE_NAMES = 12  # names listed in an incident's `where`; the incident itself has them all
TOP_EXPERTS = 8
PER_RANK_ROWS = 64
DEFAULT_RUNS = 50
MAX_RUNS = 200
MAX_COMPARED = 8

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602

SERIES_KINDS = ("layer", "parameter", "router", "scalar", "head", "attention")
# The probe's series (`probe=`, lane X5, CLI-14): its language-model head numbers (`probe.head`)
# and each attention module's statistics (`probe.attention`). Named in an error's list of kinds
# only where the records hold some, and never chosen over another kind for a name both hold
# unless `kind` says so: a call that read a layer before these were served reads it still.
PROBE_KINDS = ("head", "attention")
KIND_WORDS = {"attention": "attention module"}
# The head's numbers a series shows: the five `head_logit_trends` reads and the positions counted.
HEAD_COLUMNS = (*HEAD_NUMBERS, "positions", "nonfinite_positions", "overflow_positions")
# Values a step record carries once for the whole model, readable as scalars.
RUN_LEVEL = ("grad_norm", "clip_grad_norm", "grad_scale", "lr", "data_wait", "step_time")
CLASS_MEANINGS = {
    "invariant": "something that should never happen in a healthy run, by a fixed rule",
    "anomaly": "a statistical judgement against a baseline, the run's own history",
    "observation": "context that claims no failure, such as an intended change at a resume",
    "unclassified": "a kind this version does not know: the run state counts it as a failure",
}
INSTRUCTIONS = (
    "Read-only evidence about one PyTorch training run, from TensorQuill. Call list_incidents first: it gives"
    " the run state, the incidents and what was not checked. Then get_incident for the one you are explaining,"
    " and get_series for the layer, parameter, router or attention module it names. For a run folder with several ranks,"
    " get_all_ranks_status says which ranks are judged and which are unknown; an unknown rank is not a clean one."
    " To set this run against another, list_runs gives the runs kept beside it and compare_runs compares them by"
    " run ID or name: read what differs in what was checked before the incidents, since a kind one run did not"
    " look for cannot be called absent there."
    " An incident of class `observation` claims no failure. A check listed under not_checked was not made, so"
    " its absence from the incidents says nothing. Name the step, the layer or parameter and the kind in your"
    " diagnosis. These tools cannot change the run, and neither should you: pausing, stopping or changing it"
    " needs a person."
)


class ToolError(Exception):
    """A tool call that cannot be answered. Its text goes to the agent, and the server carries on."""


# --- Reading the path ---------------------------------------------------------------------


def _stamp(path: Path) -> tuple[int, int]:
    try:
        found = path.stat()
    except OSError as error:
        raise ToolError(f"{path} cannot be read ({error}).") from None
    return found.st_mtime_ns, found.st_size


def _parts_stamp(folder: Path, rank: int) -> tuple[tuple[str, int | None, int | None], ...]:
    """Every part `load_records(folder, rank=rank, scalars=True)` reads, with its time of change
    and size, the `scalars-*` objects among them (WAVE_AL_CONTRACT.md section 4.5). Not the newest
    part's `(attempt, last step)` alone: a late part of an older attempt, or a part written again,
    leaves that the same (CLI-22, lane V5, Nov 19, 2026)."""
    found = []
    for part in _parquet_parts(folder, rank, scalars=True):
        try:
            stat = part.stat()
        except OSError:  # removed since it was listed: the listing itself has changed
            found.append((str(part), None, None))
        else:
            found.append((str(part), stat.st_mtime_ns, stat.st_size))
    return tuple(found)


def _rank_of(record: Mapping[str, Any]) -> int:
    rank = record.get("rank", 0)
    return rank if isinstance(rank, int) and not isinstance(rank, bool) else 0


def _step_of(record: Mapping[str, Any]) -> int | None:
    step = record.get("step")
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def _step_range(records: list[dict[str, Any]]) -> dict[str, int]:
    steps = [record["step"] for record in records]
    return {"first": min(steps), "last": max(steps), "records": len(steps)}


def _ranges(ranks: Iterable[Any]) -> str:
    """Ranks as ranges: `0-63`, `0-3, 5`."""
    spans: list[list[int]] = []
    for number in sorted({r for r in ranks if isinstance(r, int) and not isinstance(r, bool)}):
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def _complete_lines(path: Path) -> tuple[list[dict[str, Any]], int]:
    """The records in a capture that a job may still be writing, and how many lines were left
    out: the last one while it has no newline yet, and any that is not a JSON object."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ToolError(f"{path} cannot be read ({error}).") from None
    records: list[dict[str, Any]] = []
    skipped = 0
    for number, line in enumerate(data[: data.rfind(b"\n") + 1].splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if isinstance(record, dict):
            # A whole record whose identity is not valid, a step missing among them, refuses the
            # capture (WAVE_AF_CONTRACT.md, section 1); it is not a torn line.
            check_identity(record, in_file(path, number))
            records.append(record)
        else:
            skipped += 1
    return records, skipped + (0 if data.endswith(b"\n") or not data.strip() else 1)


def all_ranks_look(folder: str | Path) -> dict[str, Any]:
    """One look at every rank of a run folder: the evidence the all-ranks watcher
    (`python -m tensorquill <run folder> --rank all`) writes, returned and not written
    (`evidence.all_ranks_evidence`). No part is opened and no detector runs."""
    return all_ranks_evidence(folder)


class Source:
    """The path the server was started on. Every call looks at it again; what was read is kept
    only while the file's size and time of change stay the same."""

    def __init__(self, path: str | Path, records: str | Path | None = None, runs: str | Path | None = None) -> None:
        self.path = Path(path)
        self.records_path = Path(records) if records is not None else None
        self.runs_path = Path(runs) if runs is not None else None
        self._kept: dict[Any, tuple[Any, Any]] = {}

    def _keep(self, key: Any, stamp: Any, make: Callable[[], Any]) -> Any:
        held = self._kept.get(key)
        if held is not None and stamp is not None and held[0] == stamp:
            return held[1]
        value = make()
        if isinstance(key, tuple) and key[0] == "records":  # one rank's step records at a time
            for other in [k for k in self._kept if isinstance(k, tuple) and k[0] == "records"]:
                del self._kept[other]
        self._kept[key] = (stamp, value)
        return value

    def locate(self) -> tuple[str, Path, int | None]:
        """What the path is now: `("store", run folder, rank)`, where the rank is the one a
        `rank=<n>` folder names, `("capture", file, None)` or `("evidence", file, None)`."""
        path = self.path
        if not path.exists():
            raise ToolError(
                f"{path} does not exist. This server reads that one path: nothing has been written there yet,"
                " or the server was started on the wrong path."
            )
        if path.is_dir():
            own = rank_number(path.resolve().name)
            return ("store", path.resolve().parent, own) if own is not None else ("store", path, None)
        return ("capture" if path.suffix == ".jsonl" else "evidence"), path, None

    def runs_folder(self) -> Path:
        """Where `list_runs` and `compare_runs` look: the folder given at start (`--runs`), or
        the one this run is kept in. For a run folder of a store that is the `runs/` folder
        beside it; for a capture or an evidence file it is the file's own folder."""
        if self.runs_path is not None:
            folder = self.runs_path
        else:
            path = self.path if self.path.exists() else self.path.absolute()
            if path.is_dir():
                path = path.resolve()
                folder = path.parent.parent if rank_number(path.name) is not None else path.parent
            else:
                folder = path.parent
        if not folder.is_dir():
            raise ToolError(f"{folder} is not a folder: there are no runs to list. Start the server with --runs FOLDER "
                            "to name the folder of local files or the store the runs are kept in.")
        return folder

    # Evidence files -----------------------------------------------------------------------

    def document(self, path: Path) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            try:
                found = json.loads(path.read_text(encoding="utf-8"))
            except OSError as error:
                raise ToolError(f"{path} cannot be read ({error}).") from None
            except ValueError:
                raise ToolError(
                    f"{path} is not a JSON document. An evidence file is one JSON object; a capture of step"
                    " records must be named *.jsonl."
                ) from None
            if not isinstance(found, dict) or not (
                found.get("view") == ALL_RANKS_VIEW or isinstance(found.get("incidents"), list)
            ):
                raise ToolError(
                    f"{path} is JSON but not a TensorQuill evidence file: it has no `incidents` list and is"
                    " not an all-ranks view."
                )
            return found

        return self._keep(("document", path), _stamp(path), read)

    # Captures -----------------------------------------------------------------------------

    def capture(self, path: Path) -> tuple[list[dict[str, Any]], int]:
        """Every record of a `.jsonl` capture, all ranks and attempts, and the lines left out. The
        scalar-only records are among them, in their place (WAVE_AL_CONTRACT.md section 4.6):
        whoever reads these sets them apart (`build_evidence` does), and only a series reads them."""

        def read() -> tuple[list[dict[str, Any]], int]:
            try:
                try:
                    return load_records(path, all_attempts=True, scalars=True), 0
                except MalformedRecordError:
                    raise
                except ValueError:  # a job that is still running may be in the middle of a line
                    records, skipped = _complete_lines(path)
                    return load_records(records, all_attempts=True, scalars=True), skipped
            except MalformedRecordError as error:
                # The error's own text: the field, the value and the line (section 1).
                raise ToolError(str(error)) from None
            except OSError as error:
                raise ToolError(f"{path} cannot be read ({error}).") from None
            except (TypeError, KeyError) as error:
                raise ToolError(f"{path} does not hold TensorQuill step records ({error!r}).") from None

        return self._keep(("capture", path), _stamp(path), read)

    def _capture_stamp(self, path: Path) -> Any:
        """The stamp the capture's records now held were read under. What is built from them is
        kept under it, not under a stamp taken later: a line appended between the read and that
        stamp would have kept the shorter file's evidence as the longer one's (CLI-22, lane V5,
        Nov 19, 2026)."""
        return self._kept[("capture", path)][0]

    def capture_evidence(self, path: Path, rank: int) -> dict[str, Any]:
        records, _ = self.capture(path)
        return self._capture_evidence(path, rank, records)

    def _capture_evidence(self, path: Path, rank: int, records: list[dict[str, Any]]) -> dict[str, Any]:
        """One rank's evidence from `records`, the capture's records as last read: an answer that
        also says something of the records themselves reads them once and builds from those."""

        def build() -> dict[str, Any]:
            try:
                return build_evidence([r for r in records if _rank_of(r) == rank], run_manifest=path)
            # OverflowError: an integer too large for a float, which a detector converts (BUG-25).
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ToolError(f"the detectors could not read the records in {path} ({error!r}).") from None

        evidence = self._keep(("evidence", path, rank), self._capture_stamp(path), build)
        # The run block is read at every call, not kept: the run manifest beside the capture may be
        # written or replaced after the first query while the records stay the same (XC7, lane Y3,
        # Nov 20, 2026: the project stayed `uncategorized` until the server was started again). It
        # is the block `build_evidence` makes (`run_block` of the manifest `find_manifest` finds),
        # the one thing in the evidence the manifest decides, so the detectors do not run again.
        run_id = evidence.get("run_id")
        evidence = {**evidence, "run": _manifest.run_block(run_id, _manifest.find_manifest(path, run_id))}
        # Read at every call, not kept: the marker is written after the last record, so the file's
        # stamp does not move when it appears.
        return {**evidence, "run_end": read_end_marker(path)} if path.suffix == ".jsonl" else evidence

    # One rank's step records, for series ------------------------------------------------

    def rank_records(self, rank: int | None) -> tuple[
            list[dict[str, Any]], dict[str, dict[str, None]], dict[str, Any], list[dict[str, Any]]] | None:
        """One rank's step records in step order, the names they carry by kind, where they came
        from, and the rank's scalar-only records (WAVE_AL_CONTRACT.md section 4.6), resolved with
        the step records' cutoffs and set apart from them; None when the path is an evidence file
        with no records beside it. The scalar names are the step records' and the scalar-only
        records'."""
        kind, where, own_rank = self.locate()
        default_rank = own_rank
        if kind == "evidence":
            document = self.document(where)
            where = self._records_beside(where, document)
            if where is None:
                return None
            kind = "store" if where.is_dir() else "capture"
            if kind == "store" and rank_number(where.resolve().name) is not None:
                default_rank, where = rank_number(where.resolve().name), where.resolve().parent
            if default_rank is None:
                default_rank = _judged_rank(document)
        if kind == "capture":
            every, _ = self.capture(where)
            ranks = sorted({_rank_of(r) for r in every})
            picked = _pick_rank(rank, default_rank, ranks, f"the capture {where}")
            stamp: Any = self._capture_stamp(where)

            def read() -> list[dict[str, Any]]:
                return load_records(every, rank=picked, all_attempts=True, scalars=True)
        else:
            ranks = sorted(_discover_rank_folders(where))
            if not ranks:
                raise ToolError(f"{where} holds no rank=<n> folder yet, so there are no step records to read.")
            picked = _pick_rank(rank, default_rank, ranks, f"the run folder {where}")
            stamp = _parts_stamp(where, picked)

            def read() -> list[dict[str, Any]]:
                try:
                    return load_records(where, rank=picked, all_attempts=True, scalars=True)
                except MalformedRecordError as error:
                    raise ToolError(str(error)) from None  # the field, the value, the part and row
                except ImportError:
                    raise ToolError(
                        "reading a store's step records needs pyarrow (the SDK's `flight` extra), which is not"
                        " installed here. The incidents and the all-ranks status do not need it."
                    ) from None
                except Exception as error:  # pyarrow's own errors are not built-in ones
                    raise ToolError(f"rank {picked}'s stored parts under {where} could not be read ({error}).") from None

        def index() -> tuple[list[dict[str, Any]], dict[str, dict[str, None]], list[dict[str, Any]]]:
            # Every attempt is read, so the attempts' order is the whole stream's, taken before any
            # record is resolved away (BUG-20: a rollback can leave out every training record of the
            # older attempt); then the attempts are resolved as `load_records` resolves them. The
            # scalar-only records are set apart and put in the order their points stand
            # (`scalar_points_in_order`), which `merged_scalar_points` keeps; the step records are put
            # in step order.
            every = read()
            order = attempts_first_seen(every)
            records, scalar_only = set_apart_scalars(resolve_attempts(every))
            scalar_only = scalar_points_in_order(scalar_only, order=order)
            try:
                records = sorted(records, key=lambda r: int(r["step"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ToolError(f"the records under {where} are not TensorQuill step records ({error!r}).") from None
            # A series is of one run: records of several run IDs are refused, as list_incidents and
            # get_incident refuse them through the detectors (lane AD5, Nov 21, 2026: the series held
            # both runs' rows, a step twice where both had it). The refusal is the readers' common
            # answer today, not a ruling on whether one run should be picked instead. The scalar-only
            # records are of the run too.
            try:
                check_one_run([*records, *scalar_only])
            except ValueError as error:
                raise ToolError(f"rank {picked}'s step records in {where} cannot give a series: {error}.") from None
            names = _names(records)
            for record in scalar_only:
                logged = record.get("scalars")
                if isinstance(logged, Mapping):
                    names["scalar"].update(dict.fromkeys(str(n) for n in logged))
            return records, names, scalar_only

        records, names, scalar_only = self._keep(("records", where, picked), stamp, index)
        origin = {"records": str(where), "rank": picked}
        if len(ranks) > 1:
            origin["ranks_here"] = _ranges(ranks)
        return records, names, origin, scalar_only

    def _records_beside(self, path: Path, document: Mapping[str, Any]) -> Path | None:
        """Where an evidence file's step records are: `--records`, the capture the command line
        writes it next to (`x.jsonl` for `x.evidence.json`, and for `x.evidence.compact.json`,
        the compact file `--compact` writes and a trigger hands an agent), or an all-ranks view's
        run folder."""
        if self.records_path is not None:
            if not self.records_path.exists():
                raise ToolError(f"{self.records_path}, given as --records, does not exist.")
            return self.records_path
        for ending in (".evidence.compact.json", ".evidence.json"):
            if path.name.endswith(ending):
                capture = path.with_name(path.name[: -len(ending)] + ".jsonl")
                if capture.is_file():
                    return capture
                break
        folder = document.get("run_folder")
        if document.get("view") == ALL_RANKS_VIEW and isinstance(folder, str) and Path(folder).is_dir():
            return Path(folder)
        return None

    # What every tool starts from ----------------------------------------------------------

    def open(self, rank: int | None) -> dict[str, Any]:
        """The evidence to answer from: `{"shape", "evidence", "source"}`. `shape` is `rank` for
        one rank's evidence and `all_ranks` for the view of a whole run folder."""
        kind, where, own_rank = self.locate()
        source: dict[str, Any] = {"path": str(self.path), "kind": kind}
        if kind == "capture":
            records, skipped = self.capture(where)
            ranks = sorted({_rank_of(r) for r in records})
            picked = _pick_rank(rank, None, ranks, f"the capture {where}")
            evidence = self._capture_evidence(where, picked, records)
            source["rank"] = picked
            # The run as it stands: what a resume rolled back is left out, as the detectors
            # leave it out (`resolve_attempts`; E2, lane V5, Nov 19, 2026: the last step was the
            # highest of every attempt). Every record written is counted apart, when that differs.
            # The training records: a scalar-only record is no step (WAVE_AL_CONTRACT.md section 4.6).
            written = [r for r in records if _rank_of(r) == picked and _step_of(r) is not None and not is_scalar_record(r)]
            current = resolve_attempts(written)
            if current:
                source["steps"] = _step_range(current)
            if len(current) != len(written):
                source["steps_all_attempts"] = _step_range(written)
                source["steps_note"] = (
                    "a later attempt resumed from an earlier step, so what earlier attempts wrote from that step on is"
                    " rolled back: `steps` is the run as it stands, `steps_all_attempts` every record written"
                )
            if len(ranks) > 1:
                source["ranks_here"] = _ranges(ranks)
                source["note"] = f"this file holds ranks {_ranges(ranks)}; this answer is rank {picked} (pass rank= for another)"
            if skipped:
                source["lines_left_out"] = skipped
            shape = "rank"
        elif kind == "evidence":
            evidence = self.document(where)
            shape = "all_ranks" if evidence.get("view") == ALL_RANKS_VIEW else "rank"
            if shape == "all_ranks":
                source["kind"] = "all_ranks_evidence"
                source["written_at_ms"] = evidence.get("written_at_ms")
            else:
                own = _judged_rank(evidence)
                if rank is not None and isinstance(own, int) and own != rank:
                    raise ToolError(f"{where} is rank {own}'s evidence; it says nothing about rank {rank}.")
                if isinstance(own, int):
                    source["rank"] = own
        else:
            if not _discover_rank_folders(where) and read_watchdog(where / "watchdog.json") is None:
                raise ToolError(
                    f"{where} holds no rank=<n> folder and no watchdog.json: it is not a run folder in a"
                    " tensorquill-server store (runs/<run_id>), or nothing has been stored for the run yet."
                )
            evidence = all_ranks_look(where)
            source["path"] = str(where)
            shape = "all_ranks"
        source["run_id"] = evidence.get("run_id")
        return {"shape": shape, "evidence": evidence, "source": source, "default_rank": own_rank}


def _judged_rank(document: Mapping[str, Any]) -> int | None:
    """The rank one rank's evidence file judged: its `rank` (lane U7, Nov 19, 2026), else the
    `detection` a watcher adds; None when it says neither (a file written before the key)."""
    detection = document.get("detection")
    for value in (document.get("rank"), detection.get("rank") if isinstance(detection, Mapping) else None):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _same_rank_and_run(document: Mapping[str, Any], rank: int, records: list[dict[str, Any]]) -> bool:
    """Whether an evidence file judged exactly these records' rank and run, so its samples and
    flags can stand for them. A file that does not say its rank is not taken on trust."""
    runs = {record.get("run_id") if isinstance(record.get("run_id"), str) else "" for record in records}
    own_run = document.get("run_id") if isinstance(document.get("run_id"), str) else ""
    return _judged_rank(document) == rank and len(runs) <= 1 and own_run in (runs or {own_run})


def _steps_said(steps: list[int]) -> str:
    if not steps:
        return "no step"
    return f"step {steps[0]}" if len(steps) == 1 else f"{len(steps)} steps, steps {steps[0]} to {steps[-1]}"


def _router_steps_changed(name: str, document: Mapping[str, Any], records: list[dict[str, Any]]) -> str | None:
    """None when one rank's evidence file judged the router `name` at exactly the steps, with
    exactly the counts, that `records` hold for it now; else what differs, for the answer. The
    records may have changed after the file was written: a capture that grew, or one started again
    (BUG-25, wave AO). Steps on which neither has a point (no counts, no routed token) are left out
    on both sides, as a series leaves them out."""
    judged: dict[int, Any] = {}
    for sample in document.get("samples") or ():
        if (isinstance(sample, dict) and sample.get("router") == name and not sample.get("missing")
                and _router_point(sample.get("counts"), None, None) is not None):
            judged[int(sample["step"])] = sample.get("counts")
    held: dict[int, Any] = {}
    for record in records:
        moe = record.get("moe")
        counts = moe.get(name) if isinstance(moe, Mapping) else None
        if _router_point(counts, None, None) is not None:
            held[int(record["step"])] = counts
    if judged == held:
        return None
    at = sorted(judged)
    now = sorted(held)
    differ = next((step for step in at if step in held and held[step] != judged[step]), None)
    return (
        f"the router {name!r} at {_steps_said(at)}, and the records read here hold it at {_steps_said(now)}"
        + (f", with other counts at step {differ}" if differ is not None and at == now else "")
        + "; the records differ from what it judged, as when a capture grew or was started again"
    )


def _pick_rank(asked: int | None, default: int | None, ranks: list[int], what: str) -> int:
    """The rank to read: the one asked for, else the one the path names, else the lowest."""
    wanted = asked if asked is not None else default
    if wanted is None:
        return ranks[0] if ranks else 0
    if wanted not in ranks:
        held = f"it holds rank{'s' if len(ranks) != 1 else ''} {_ranges(ranks)}" if ranks else "it holds no step records yet"
        raise ToolError(f"rank {wanted} is not in {what}: {held}.")
    return wanted


# --- Incidents ----------------------------------------------------------------------------


def _class_of(incident: Mapping[str, Any]) -> str:
    """The class from the kind, as `run_state` takes it; an unknown kind keeps the one it carries."""
    own = incident.get("class")
    return incident_class(incident.get("kind")) or (own if isinstance(own, str) else "unclassified")


def _name(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    for key in ("name", "module", "rank"):
        if key in item:
            return item[key]
    if "expert" in item:
        return {key: item[key] for key in ("layer", "router", "expert") if key in item}
    return dict(item)


def _where(incident: Mapping[str, Any]) -> dict[str, Any]:
    """What an incident names, in one shape for every kind: the MoE layer and router with its
    expert numbers (from 1), and the parameters, modules, layers, gradients, experts or ranks.
    At most `WHERE_NAMES` of each, with the count when there are more."""
    where: dict[str, Any] = {}
    for key in ("layer", "router"):
        if incident.get(key) is not None:
            where[key] = incident[key]
    for key in ("overloaded", "dead", "parameters", "modules", "layers", "gradients", "experts", "ranks"):
        value = incident.get(key)
        if isinstance(value, Mapping):
            names = list(value)
        elif isinstance(value, list):
            names = [_name(item) for item in value]
        else:
            continue
        if names or key in ("overloaded", "dead"):
            where[key] = names[:WHERE_NAMES]
            if len(names) > WHERE_NAMES:
                where[f"{key}_count"] = len(names)
    return where


def _entries(opened: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every incident of the evidence with its id: `<kind>@<step>`, and `#2` onwards when the
    same kind is listed at the same step more than once. In the all-ranks view an entry is a
    group of ranks showing the same incident, or an incident of the run as a whole."""
    evidence = opened["evidence"]
    found: list[dict[str, Any]] = []
    if opened["shape"] == "rank":
        found += [{"incident": i} for i in evidence.get("incidents") or () if isinstance(i, dict)]
    else:
        for group in evidence.get("incident_groups") or ():
            if isinstance(group, dict) and isinstance(group.get("incident"), dict):
                found.append({"incident": group["incident"], "group": group})
        found += [{"incident": i, "run_level": True} for i in evidence.get("run_incidents") or () if isinstance(i, dict)]
    seen: dict[str, int] = {}
    for entry in found:
        incident = entry["incident"]
        step = incident.get("step")
        base = f"{incident.get('kind')}@{step}" if isinstance(step, int) and not isinstance(step, bool) else str(incident.get("kind"))
        seen[base] = seen.get(base, 0) + 1
        entry["id"] = base if seen[base] == 1 else f"{base}#{seen[base]}"
    return found


# The lookup id (wave AH, P5-1, Nov 22, 2026): what the trigger hands a woken agent as `mcp_id`,
# beside its delivery id. `<run_id>/<ranks>/<kind>@<step>`, with `#2` onwards for the second
# incident of that kind and step ON THOSE RANKS, where `<ranks>` is `rank0`, `ranks0-3,7` or `run`.
# Unlike a list id (`<kind>@<step>`, numbered over the whole view), it says whose incident it is,
# so it finds the same incident whether the server was started on the run folder, a `rank=<n>`
# folder, a capture or an evidence file, and whether the watcher looked at one rank or all of
# them. A list id never holds a `/`, so the two forms cannot be mistaken for each other.
#
# A stall's lookup id (`job_stalled`, `rank_silent`) is of its episode instead: the trigger's
# delivery id in this form, with `~2` onwards for a later episode fired at the same kind and step.
# It is resolved only against the trigger's state file (`_stall_of_episode`).


_STALL_KINDS = ("job_stalled", "rank_silent")  # read from the server's status: one at a time for a run


def _base_id(kind: Any, step: Any) -> str:
    return f"{kind}@{step}" if isinstance(step, int) and not isinstance(step, bool) else str(kind)


def lookup_id(run_id: Any, ranks: Iterable[Any], kind: Any, step: Any, nth: int = 1) -> str:
    """The lookup id of the `nth` incident of `kind` at `step` on `ranks` of `run_id` (see above)."""
    numbers = sorted({r for r in ranks if isinstance(r, int) and not isinstance(r, bool)})
    who = f"rank{'s' if len(numbers) != 1 else ''}{_ranges(numbers).replace(' ', '')}" if numbers else "run"
    return f"{run_id}/{who}/{_base_id(kind, step)}" + (f"#{nth}" if nth > 1 else "")


def _parse_lookup(wanted: str) -> dict[str, Any] | None:
    """A lookup id's parts, read from the right (a run id may hold a `/`); None for anything else,
    a list id among them."""
    head, slash, last = wanted.rpartition("/")
    run, slash2, who = head.rpartition("/")
    if not slash or not slash2 or not run or not last:
        return None
    if who == "run":
        ranks: list[int] = []
    else:
        spans = who[5:] if who.startswith("ranks") else who[4:] if who.startswith("rank") else ""
        ranks = []
        for span in spans.split(","):
            first, dash, end = span.partition("-")
            if not first.isdecimal() or (dash and not end.isdecimal()) or (dash and int(end) < int(first)):
                return None
            if int(end or first) - int(first) > 1_000_000:
                return None
            ranks.extend(range(int(first), int(end or first) + 1))
    last, tilde, episode = last.partition("~")
    if tilde and (not episode.isdecimal() or int(episode) < 1):
        return None
    base, hash_, number = last.partition("#")
    if hash_ and (not number.isdecimal() or int(number) < 1 or tilde):
        return None
    kind, at, step = base.partition("@")
    if not kind or (at and not step.isdecimal()):
        return None
    return {"run": run, "who": who, "ranks": ranks, "kind": kind, "step": int(step) if at else None,
            "nth": int(number) if hash_ else 1, "episode": int(episode) if tilde else 1}


def lookup_candidates(opened: Mapping[str, Any], kind: Any, step: Any, ranks: Iterable[Any]) -> list[dict[str, Any]]:
    """The entries a lookup id can mean, in the list's order: of that kind at that step (None: an
    incident with no whole-number step), and, in the all-ranks view, a group that holds every one
    of `ranks` or an incident of the run as a whole. One rank's evidence is of its own rank: the
    caller sees that it is the right one. The trigger numbers its lookup ids with this same list."""
    wanted = {r for r in ranks if isinstance(r, int) and not isinstance(r, bool)}
    found = []
    for entry in _entries(opened):
        incident = entry["incident"]
        at = incident.get("step")
        at = at if isinstance(at, int) and not isinstance(at, bool) else None
        if incident.get("kind") != kind or at != step:
            continue
        group = entry.get("group")
        if group is not None and not wanted <= set(group.get("ranks") or ()):
            continue
        found.append(entry)
    return found


def lookup_ids(evidence: Mapping[str, Any], run_id: Any, units: Iterable[tuple[Mapping[str, Any], Iterable[Any]]]) -> list[str]:
    """The lookup id of each `(incident, ranks)` of `evidence` (the trigger's units, at the look
    they were read in): the incident's place among the candidates for its ranks."""
    opened = {"shape": "all_ranks" if evidence.get("view") == ALL_RANKS_VIEW else "rank", "evidence": evidence}
    out = []
    for incident, ranks in units:
        ranks = list(ranks)
        step = incident.get("step")
        step = step if isinstance(step, int) and not isinstance(step, bool) else None
        candidates = lookup_candidates(opened, incident.get("kind"), step, ranks)
        nth = next((n for n, entry in enumerate(candidates, 1) if entry["incident"] is incident), 1)
        out.append(lookup_id(run_id, ranks, incident.get("kind"), step, nth))
    return out


def _looked_up(source: "Source", wanted: str, lookup: Mapping[str, Any], rank: int | None) -> tuple[dict[str, Any], dict[str, Any], int | None]:
    """The source opened for a lookup id, the entry it names there, and the rank to read series
    of. Refused by name when the id is of another run, of a rank this evidence is not of, or of
    an incident that is not listed (any more)."""
    ranks = lookup["ranks"]
    opened = source.open(rank)
    own = opened["source"].get("rank")
    if opened["shape"] == "rank" and rank is None and ranks and isinstance(own, int) and own not in ranks:
        opened = source.open(ranks[0])  # a capture of several ranks; an evidence file of another rank says so
        own = opened["source"].get("rank")
    run = opened["source"].get("run_id")
    if run is not None and str(run) != lookup["run"]:
        raise ToolError(
            f"{wanted!r} is an incident of the run {lookup['run']!r}, and this server was started on the run"
            f" {str(run)!r} ({opened['source'].get('path')}). Start a server on that run's capture, evidence file or"
            " run folder to look it up."
        )
    if opened["shape"] == "rank" and ranks and isinstance(own, int) and own not in ranks:
        raise ToolError(f"{wanted!r} is an incident of rank {_ranges(ranks)}; this answer would be rank {own}'s evidence,"
                        " which says nothing about it.")
    if lookup["kind"] in _STALL_KINDS:
        return opened, _stall_of_episode(source, opened, wanted, lookup), rank
    candidates = lookup_candidates(opened, lookup["kind"], lookup["step"], ranks)
    if len(candidates) < lookup["nth"]:
        entries = _entries(opened)
        known = ", ".join(e["id"] for e in entries[:20]) or "none"
        more = f" and {len(entries) - 20} more" if len(entries) > 20 else ""
        who = f"rank{'s' if len(ranks) != 1 else ''} {_ranges(ranks)}" if ranks else "the run"
        raise ToolError(
            f"no incident matches {wanted!r}: {_base_id(lookup['kind'], lookup['step'])} is listed"
            f" {len(candidates)} time{'s' if len(candidates) != 1 else ''} for {who} here. The incidents here are:"
            f" {known}{more}. Call list_incidents for the current list."
        )
    if rank is None and len(ranks) == 1:
        rank = ranks[0]
    return opened, candidates[lookup["nth"] - 1], rank


class _NotIdentifiable(Exception):
    """A stall's lookup id whose episode cannot be shown to be the one of the stall listed now."""

    def __init__(self, reason: str, current: Mapping[str, Any] | None) -> None:
        super().__init__(reason)
        self.reason, self.current = reason, current


def _stall_of_episode(source: "Source", opened: Mapping[str, Any], wanted: str, lookup: Mapping[str, Any]) -> dict[str, Any]:
    """The entry of the stall listed now, when it is established to be of the episode `wanted`
    fired for; `_NotIdentifiable` otherwise.

    A stall's kind and step change inside one episode, and a later stall of the run, or the job
    started again, can read exactly as an earlier one did (the same kind, rank and step). So
    nothing in the incident says which episode it is of: only the trigger's state file does, beside
    the evidence file the watcher writes. Established means all of: this server reads that evidence
    file; its state file lists, as this run's open episode, the firing the id names (its delivery
    id), kept with its baseline (not a file from before episodes were kept, nor an episode adopted
    without its firing); the run lists exactly one stall; and that stall does not end the episode
    by the trigger's own rule (`trigger.episode_ended_by`: no step beyond where it stopped, no new
    attempt). The last holds even in the moment between the watcher writing a new stall and the
    trigger looking at it, when the state file still names the earlier episode."""
    from .trigger import episode_ended_by, state_path_handed

    stalls = [e for e in _entries(opened) if e["incident"].get("kind") in _STALL_KINDS]
    current = stalls[0] if len(stalls) == 1 else None

    def no(reason: str) -> _NotIdentifiable:
        return _NotIdentifiable(reason, current)

    if not stalls:
        raise no("the run lists no stall now: the one this id fired for is over, or is not listed at the moment (as in the"
                 " window after a server restart)")
    if current is None:
        raise no(f"the evidence lists {len(stalls)} stalls, so none can be taken for the one this id fired for")
    kind, where, _ = source.locate()
    if kind != "evidence":
        raise no(f"this server was started on {where}, which has no trigger state beside it. Which stall episode is open is"
                 " kept only beside the evidence file the trigger named (`TENSORQUILL_EVIDENCE`, `evidence` in its document):"
                 " start the server on that file to look a stall up by the id the trigger gave")
    state_path = state_path_handed(where)  # the one the trigger keeps, for any output name (wave AP, AO-06)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise no(f"the trigger's state file {state_path} is missing or cannot be read, so nothing says which stall episode"
                 " is open. It is kept beside the evidence file the trigger named (`TENSORQUILL_EVIDENCE`): start the"
                 " server on that file") from None
    step = lookup["step"]
    delivery = f"{lookup['run']}/{lookup['who']}/{lookup['kind']}/{step}" + (f"/{lookup['episode']}" if lookup["episode"] > 1 else "")
    episodes = state.get("episodes") if isinstance(state, dict) else None
    episode = episodes.get(lookup["run"]) if isinstance(episodes, dict) else None
    if not isinstance(episode, dict):
        raise no("the trigger's state lists no open stall episode for this run: the one this id fired for is over")
    if episode.get("delivery") != delivery:
        raise no("the stall episode this id fired for is over: the trigger's state shows another episode open for this run"
                 f" ({episode.get('delivery')})")
    if episode.get("legacy") or episode.get("adopted") or not isinstance(episode.get("ranks"), dict):
        raise no("the trigger's state does not hold what this episode was stopped at when it fired (it was taken up from an"
                 " earlier state file), so the stall listed now cannot be shown to be the same one")
    ended = episode_ended_by(episode, current["incident"])
    if ended is not None:
        raise no("the stall listed now is not the one this id fired for: it shows "
                 + ("a new attempt of the job" if ended == "attempt" else "a step stored beyond where that one stopped")
                 + " since")
    asked = _base_id(lookup["kind"], step)
    return {**current, "looked_up_as": asked} if current["id"] != asked else dict(current)


def _delivery_id_hint(wanted: str) -> str:
    """For an id shaped as the trigger's delivery id (`<run_id>/<ranks>/<kind>/<step>`): what to pass instead."""
    parts = wanted.split("/")
    if len(parts) >= 4 and parts[-1].isdecimal() and len(parts) >= 5 and parts[-2].isdecimal():
        parts = parts[:-1]  # a later stall episode at the same step: `/<n>` after the step
    if len(parts) < 4 or not (parts[-3] == "run" or parts[-3].startswith("rank")):
        return ""
    made = "/".join(parts[:-2]) + f"/{parts[-2]}@{parts[-1]}" if parts[-1].isdecimal() else "/".join(parts[:-1])
    return (f" {wanted!r} looks like the trigger's delivery id (`id`, `TENSORQUILL_INCIDENT_ID`), which is not a lookup"
            f" id. Pass the `mcp_id` the trigger gave beside it (`TENSORQUILL_INCIDENT_MCP_ID`): most likely {made!r}.")


def _brief(entry: Mapping[str, Any]) -> dict[str, Any]:
    incident = entry["incident"]
    out: dict[str, Any] = {
        "id": entry["id"], "kind": incident.get("kind"), "class": _class_of(incident), "step": incident.get("step"),
        **_since_start(incident),
        "where": _where(incident),
    }
    group = entry.get("group")
    if group is not None:
        ranks = group.get("ranks") or []
        out.update({
            "on_ranks": _ranges(ranks), "rank_count": len(ranks),
            "on_all_judged_ranks": group.get("on_all_judged_ranks"), "numbers_differ": group.get("numbers_differ"),
        })
    elif entry.get("run_level"):
        out["on_ranks"] = "the run as a whole"
    return out


def _not_checked(opened: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What was not checked, as the evidence lists it; in the all-ranks view each rank's own
    entries are gathered, with the ranks that list them."""
    evidence = opened["evidence"]
    listed = [dict(item) for item in evidence.get("not_checked") or () if isinstance(item, dict)]
    if opened["shape"] == "rank":
        return listed
    by_text: dict[str, dict[str, Any]] = {}
    # A rank's entries are on its entry of `ranks` (a store's reports) or in `not_checked_by_rank`
    # (the view `combine_rank_evidence` builds, EV-02): both are `{"rank", "not_checked"}`.
    for entry in [*(evidence.get("ranks") or ()), *(evidence.get("not_checked_by_rank") or ())]:
        for item in entry.get("not_checked") or () if isinstance(entry, dict) else ():
            if isinstance(item, dict):
                held = by_text.setdefault(json.dumps(item, sort_keys=True, default=str), {**item, "ranks": []})
                held["ranks"].append(entry.get("rank"))
    return listed + [{**item, "ranks": _ranges(item["ranks"])} for item in by_text.values()]


def _kinds(opened: Mapping[str, Any], answer: dict[str, Any]) -> None:
    """Put what "no incidents" covers into an answer, under `kinds`: the kinds that were looked
    for with enough history, those still warming up, and those that could not be looked for.
    For a run folder they are the run's (`all_ranks_kinds`), and when the view cannot say,
    `kinds_not_stated` gives the reason in place of `kinds`."""
    evidence = opened["evidence"]
    kinds = all_ranks_kinds(evidence) if opened["shape"] == "all_ranks" else kinds_by_value(evidence)
    if kinds is not None:
        answer["kinds"] = kinds
    elif isinstance(evidence.get("kind_checks_not_stated"), dict):
        answer["kinds_not_stated"] = evidence["kind_checks_not_stated"]


def _summary(evidence: Mapping[str, Any]) -> str | None:
    try:
        return incident_summary(evidence)
    except Exception:  # an evidence file from another version: the incidents are still listed
        return None


def _rank_counts(evidence: Mapping[str, Any]) -> dict[str, Any]:
    def group(key: str) -> dict[str, Any]:
        ranks = evidence.get(key) or []
        return {"count": len(ranks), "ranks": _ranges(ranks)}

    return {
        "count": evidence.get("rank_count"), "world_size": evidence.get("world_size"),
        "judged": group("judged_ranks"), "unknown": group("unknown_ranks"), "clean": group("clean_ranks"),
        "with_incidents": group("ranks_with_incidents"),
        "note": "an unknown rank's report is not current: it is not counted as having no incidents",
    }


def _rank_entry(evidence: Mapping[str, Any], rank: int) -> dict[str, Any]:
    for entry in evidence.get("ranks") or ():
        if isinstance(entry, dict) and entry.get("rank") == rank:
            return entry
    known = _ranges(e.get("rank") for e in evidence.get("ranks") or () if isinstance(e, dict))
    raise ToolError(f"rank {rank} is not in this run. Its ranks are: {known or 'none yet'}.")


def _dropped(opened: Mapping[str, Any]) -> dict[str, Any] | None:
    """The rows the server dropped, for a null key or for a `schema_version` it does not support
    (`unsupported_schema_version`, since wave AA), from the run folder's `watchdog.json` (L22,
    lane Z5, Nov 20, 2026; `runs.dropped_rows`): None for any other source, and when no rank
    dropped a row, so every answer of such a run is as before. Batches refused whole are not
    rows and are not here (`_refused`). An upper bound: a batch sent again after a lost
    acknowledgement is counted again."""
    if opened["source"].get("kind") != "store":
        return None
    return _runs.dropped_rows(read_watchdog(Path(opened["source"]["path"]) / "watchdog.json"))


def _refused(opened: Mapping[str, Any]) -> dict[str, Any] | None:
    """The batches the server refused whole because a key column was missing or of another type
    (RS10, wave AA; `runs.refused_batches`), from the run folder's `watchdog.json`: None for any
    other source, and when no rank refused one, so every answer of such a run is as before. A
    count of batches, never added to `_dropped`'s rows (lane AB5, Nov 20, 2026)."""
    if opened["source"].get("kind") != "store":
        return None
    return _runs.refused_batches(read_watchdog(Path(opened["source"]["path"]) / "watchdog.json"))


def _not_newer(opened: Mapping[str, Any]) -> dict[str, Any] | None:
    """The rows the server received and did not store because their step was not above one it had
    kept for their attempt, and the distinct steps among them confirmed late inside one stored part
    (L30, WAVE_AE_CONTRACT.md, section 6; `runs.rows_not_newer`), from the run folder's
    `watchdog.json`: None for any other source, and when no rank has `rows_not_newer`, so every
    answer of such a run is as before. Never added to `_dropped`'s rows."""
    if opened["source"].get("kind") != "store":
        return None
    return _runs.rows_not_newer(read_watchdog(Path(opened["source"]["path"]) / "watchdog.json"))


def _rank_not_newer(found: Mapping[str, Any] | None, rank: int) -> dict[str, Any] | None:
    """One rank's part of `_not_newer`, `{"rows", "late_steps_confirmed"?, "note"}` with the
    page's sentence of that rank's counts, or None when the rank has none."""
    mine = _rank_dropped(found, rank)
    if mine is None:
        return None
    return {**mine, "note": _runs.rows_not_newer_note(mine["rows"], mine.get("late_steps_confirmed", 0))}


def _capture_stopped(opened: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Each rank's `capture_stopped` from the run folder's `watchdog.json`, copied as written, by
    rank: empty for any other source and for a rank whose capture did not stop."""
    if opened["source"].get("kind") != "store":
        return {}
    status = read_watchdog(Path(opened["source"]["path"]) / "watchdog.json")
    found: dict[int, dict[str, Any]] = {}
    for entry in (status or {}).get("ranks") or ():
        if isinstance(entry, Mapping) and isinstance(entry.get("capture_stopped"), Mapping):
            rank = entry.get("rank")
            if isinstance(rank, int) and not isinstance(rank, bool):
                found[rank] = copy.deepcopy(dict(entry["capture_stopped"]))
    return found


def _dropped_shown(dropped: Any) -> Any:
    """`dropped_rows` or `refused_batches` as an answer shows it: the total and the keys whole,
    and the ranks up to `PER_RANK_ROWS` (`ranks_omitted` counts the rest; `rank=` gives one
    rank's). A missing key or an old sender usually hits every rank, so the list would otherwise
    grow with the run."""
    if not isinstance(dropped, dict) or len(dropped.get("ranks") or ()) <= PER_RANK_ROWS:
        return dropped
    return {**dropped, "ranks": dropped["ranks"][:PER_RANK_ROWS], "ranks_omitted": len(dropped["ranks"]) - PER_RANK_ROWS}


def _rank_dropped(dropped: Mapping[str, Any] | None, rank: int) -> dict[str, Any] | None:
    """One rank's part of `_dropped` (`{"rows", "by_key"}`) or of `_refused` (`{"batches",
    "by_key"}`), or None when it dropped or refused none."""
    for item in (dropped or {}).get("ranks") or ():
        if item.get("rank") == rank:
            return {key: value for key, value in item.items() if key != "rank"}
    return None


# What the evidence says beside its incidents that an agent with only these tools could not see
# (L29, lane AD5, Nov 21, 2026): each under the evidence's own key and in its shape, only where the
# evidence has it, and as the evidence caps it. One rank's evidence has `skipped_updates` and
# `checkpoint_tensors`; the all-ranks view has `skipped_updates_by_rank` and
# `checkpoint_tensors_by_rank` (STEP_AND_EVIDENCE_CONTRACT.md, section 1;
# CHECKPOINT_TENSOR_CONTRACT.md, section 4).
# The four measurements of DETECTOR_RULES_CONTRACT.md, part 6: numbers with no threshold, which no rule
# reads; and the routing source of part 7 (`moe_source`, a router's `form` and `point`).
MEASUREMENT_BLOCKS = ("gradient_trend", "attention_trends", "update_trends", "weight_jumps")
EVIDENCE_BLOCKS = ("skipped_updates", "skipped_updates_by_rank", "checkpoint_tensors", "checkpoint_tensors_by_rank",
                   *MEASUREMENT_BLOCKS,
                   # The all-ranks view's: each judged rank's measurements and routing source, with its
                   # rank (WAVE_AF_CONTRACT.md, section 2, EV-04).
                   *(f"{name}_by_rank" for name in MEASUREMENT_BLOCKS), "moe_source_by_rank")
# Of a kind's entry in `kind_checks`, what `kinds` (names by value) leaves out: the parameters a
# stall did not judge for want of their gradient, and the replica observations' coverage. And the
# entry's `note` when it is the words for a rank with no peer (contract, section 2); a kind's other
# notes are not carried.
# Since the detector rules contract (wave AD): a router still in the state it started in
# (`moe_imbalance.starting_state`, part 3), the ranks' lists of parameters not judged
# (`stalled_weights.parameters_not_judged_by_rank`, part 5) and the modules an activation kind did
# not judge (`modules_not_judged`, part 7).
KIND_CHECK_KEYS = ("parameters_not_judged", "parameters_not_judged_by_rank", "modules_not_judged", "coverage",
                   "starting_state",
                   # A run's entry, where its ranks differ in them (EV-03).
                   "modules_not_judged_by_rank", "starting_state_by_rank")
# The blocks an incident's own answer carries, by its kind: what bears on reading that incident.
_SKIPS = ("skipped_updates", "skipped_updates_by_rank")
INCIDENT_BLOCKS = {
    "resume_incomplete": ("checkpoint_tensors", "checkpoint_tensors_by_rank"),
    **{kind: _SKIPS for kind in ("non_finite", "update_spike", "stalled_weights", "loss_scale_trouble")},
    "grad_spike": (*_SKIPS, "gradient_trend", "gradient_trend_by_rank"),
    "update_spike": (*_SKIPS, "update_trends", "weight_jumps", "update_trends_by_rank", "weight_jumps_by_rank"),
    **{kind: ("attention_trends", "attention_trends_by_rank")
       for kind in ("attention_collapse", "attention_uniform", "attention_logit_growth")},
    # A router's incident in the all-ranks view: the ranks' routing sources (one rank's is `moe_source`).
    **{kind: ("moe_source_by_rank",) for kind in ("moe_imbalance", "starved_experts")},
}

# What a reader shows for a score (DETECTOR_RULES_CONTRACT.md, part 1): the number to one decimal place,
# or why there is none. A null score is never shown as 0, and the line is never left out.
SCORE_WORDS = {
    "short_history": "no score: fewer than 10 earlier values",
    "no_dispersion": "no score: the earlier values had zero median absolute deviation",
}
# How a reader names a router's assignments by where they were taken (part 7): "dispatched" only
# when the capture says `point` is "dispatched".
ROUTING_WORDS = {
    "dispatched": "the dispatched assignments",
    "requested": "the assignments the router asked for",
    "unknown": "the assignments the model gave; whether a capacity limit was applied before this point is not known",
}


def score_display(entry: Mapping[str, Any]) -> str | None:
    """The words for an entry's `score` (`score 31.2`, or why there is none); None for an entry
    that has neither `score` nor `score_note`, as one written before the score."""
    if "score" not in entry and "score_note" not in entry:
        return None
    score = entry.get("score")
    if _finite(score) is not None:
        return f"score {score:.1f}"
    return SCORE_WORDS.get(str(entry.get("score_note")), "no score")


def _scores(incident: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each `ratio` of an incident that has a score beside it, with the words: the incident's own,
    then those of its `parameters` and `modules`, in their order."""
    found: list[dict[str, Any]] = []

    def add(at: str, entry: Mapping[str, Any]) -> None:
        said = score_display(entry)
        if said is not None and "ratio" in entry:
            found.append({"at": at, "ratio": entry.get("ratio"), "score": entry.get("score"),
                          **({"score_note": entry["score_note"]} if "score_note" in entry else {}), "display": said})

    add("incident", incident)
    for key in ("parameters", "modules"):
        listed = incident.get(key)
        for item in listed if isinstance(listed, list) else ():
            if isinstance(item, Mapping):
                name = item.get("name", item.get("module"))
                add(f"{key}: {name}" + (f" ({item['stat']})" if isinstance(item.get("stat"), str) else ""), item)
    return found


def _since_start(incident: Mapping[str, Any]) -> dict[str, Any]:
    """For a `moe_imbalance` incident of a router flagged since its first sample (part 3): when it
    began (`step`) and when it became reportable, kept apart, with the words. Empty otherwise."""
    if incident.get("since_start") is not True:
        return {}
    return {"since_start": True, "reportable_step": incident.get("reportable_step"),
            "when": f"collapsed since step {incident.get('step')}, reportable at step {incident.get('reportable_step')}"}


def routing_words(source: Any) -> str | None:
    """The words for one router's `moe_source` entry (`{"form", "point"}`); None without one."""
    return ROUTING_WORDS.get(source.get("point")) if isinstance(source, Mapping) else None


def _routing(sources: Any, routers: Iterable[Any] | None = None) -> dict[str, Any]:
    """`moe_source` (of `routers`, when given) and `moe_source_words` beside it; empty when the
    evidence or records have no `moe_source`, so a reader makes no claim."""
    if not isinstance(sources, Mapping):
        return {}
    wanted = set(routers) if routers is not None else None
    kept = {router: copy.deepcopy(dict(entry)) for router, entry in sources.items()
            if isinstance(entry, Mapping) and (wanted is None or router in wanted)}
    if not kept:
        return {}
    words = {router: said for router, entry in kept.items() if (said := routing_words(entry)) is not None}
    return {"moe_source": kept, **({"moe_source_words": words} if words else {})}


def _kind_check_keys(evidence: Mapping[str, Any], kinds: Iterable[Any] | None = None) -> dict[str, Any] | None:
    """`{kind: {...}}`: of each kind's entry in the evidence's `kind_checks` (of `kinds`, when
    given), the keys of `KIND_CHECK_KEYS` it has and its `note` when it is `NO_PEER_WORDS`, as the
    entry has them; None when no entry has any, so the answer has no such key."""
    checks = evidence.get("kind_checks")
    if not isinstance(checks, Mapping):
        return None
    wanted = set(kinds) if kinds is not None else None
    found: dict[str, Any] = {}
    for kind, check in checks.items():
        if not isinstance(check, Mapping) or (wanted is not None and kind not in wanted):
            continue
        kept = {key: copy.deepcopy(check[key]) for key in KIND_CHECK_KEYS if key in check}
        if check.get("note") == NO_PEER_WORDS:
            kept["note"] = NO_PEER_WORDS
        if kept:
            found[kind] = kept
    return found or None


def _carry_blocks(evidence: Mapping[str, Any], answer: dict[str, Any], keys: Iterable[str] = EVIDENCE_BLOCKS,
                  ranks: Iterable[Any] | None = None) -> None:
    """Put the evidence's blocks of `keys` into `answer`, copied, each only where the evidence has
    it. `ranks` keeps an all-ranks list's entries of those ranks only, with no key when none is
    left. An all-ranks list has one entry a judged rank, so it is listed up to `PER_RANK_ROWS`
    entries, with `<key>_omitted` counting the rest (`rank=` gives any rank's), as the dropped
    rows' ranks are. The evidence is the kept one, so nothing of it is changed."""
    wanted = set(ranks) if ranks is not None else None
    for key in keys:
        value = evidence.get(key)
        if isinstance(value, Mapping) and value:
            answer[key] = copy.deepcopy(dict(value))
        elif isinstance(value, list) and value:
            kept = [e for e in value if isinstance(e, Mapping) and (wanted is None or e.get("rank") in wanted)]
            if kept:
                answer[key] = copy.deepcopy(kept[:PER_RANK_ROWS])
            if len(kept) > PER_RANK_ROWS:
                answer[f"{key}_omitted"] = len(kept) - PER_RANK_ROWS


def list_incidents(source: Source, arguments: Mapping[str, Any]) -> dict[str, Any]:
    rank = _integer(arguments, "rank", minimum=0)
    limit = _integer(arguments, "limit", minimum=1, maximum=MAX_INCIDENTS) or DEFAULT_INCIDENTS
    opened = source.open(rank)
    evidence = opened["evidence"]
    entries = _entries(opened)
    answer: dict[str, Any] = {"source": opened["source"]}
    if opened["shape"] == "all_ranks":
        rank = rank if rank is not None else opened["default_rank"]  # the server was started on a rank=<n> folder
        answer["ranks"] = _rank_counts(evidence)
        if rank is not None:
            entry = _rank_entry(evidence, rank)
            detection = entry.get("detection") or {}
            entries = [e for e in entries if "group" in e and rank in (e["group"].get("ranks") or ())]
            answer["rank"] = {
                "rank": rank, "judged": detection.get("judged"), "report": detection.get("state"),
                "reason": detection.get("reason"),
                "note": "only this rank's incidents are listed; incidents of the run as a whole are left out",
            }
            if not detection.get("judged"):
                answer["rank"]["note"] = (
                    "this rank's report is not current, so nothing is said about incidents on it; an empty list"
                    " here does not mean it has none (get_all_ranks_status with rank= gives its last report)"
                )
            mine = _rank_dropped(_dropped(opened), rank)
            if mine is not None:  # L22: no key when the rank dropped none
                answer["rank"]["dropped_rows"] = mine
            refused = _rank_dropped(_refused(opened), rank)
            if refused is not None:  # RS10: no key when the rank refused none
                answer["rank"]["refused_batches"] = refused
            not_newer = _rank_not_newer(_not_newer(opened), rank)
            if not_newer is not None:  # L30: no key when the rank has none
                answer["rank"]["rows_not_newer"] = not_newer
    state = evidence.get("run_state")
    answer["run_state"] = state if isinstance(state, dict) else None
    if not isinstance(state, dict):
        answer["run_state_note"] = "this evidence file carries no run_state (it was written by an older version)"
    by_class: dict[str, list[Any]] = {}
    for entry in entries:
        kinds = by_class.setdefault(_class_of(entry["incident"]), [])
        if entry["incident"].get("kind") not in kinds:
            kinds.append(entry["incident"].get("kind"))
    answer["incident_count"] = len(entries)
    answer["by_class"] = by_class
    answer["incidents"] = [_brief(entry) for entry in entries[:limit]]
    if len(entries) > limit:
        answer["incidents_omitted"] = len(entries) - limit
    answer["not_checked"] = _not_checked(opened)
    _kinds(opened, answer)
    if isinstance(evidence.get("detection"), dict):
        answer["detection"] = evidence["detection"]
    if isinstance(evidence.get("run_end"), dict):
        answer["run_end"] = evidence["run_end"]  # a local file: whether its session closed it
    if isinstance(evidence.get("head_trends"), dict):
        # The probe's head numbers over the run (`evidence.head_logit_trends`, with the positions
        # whose sums overflow), wherever the evidence has them (L15, lane Y3): one rank's evidence
        # built from its records. No key where it has none, so other answers stay as they were.
        answer["head_trends"] = evidence["head_trends"]
    # L29 (lane AD5): skipped updates, the checkpoint's tensors, and what `kinds` leaves out of a
    # kind's entry. With `rank` on a run folder, that rank's entries of the all-ranks lists.
    _carry_blocks(evidence, answer, ranks=[rank] if opened["shape"] == "all_ranks" and rank is not None else None)
    answer.update(_routing(evidence.get("moe_source")))
    kind_keys = _kind_check_keys(evidence)
    if kind_keys is not None:
        answer["kind_checks"] = kind_keys
    summary = _summary(evidence)
    if summary:
        answer["summary"] = summary
    return answer


def _sample_context(evidence: Mapping[str, Any], incident: Mapping[str, Any]) -> dict[str, Any] | None:
    """The MoE sample an incident points at: the largest shares, and the sample before it.

    In compact evidence that left steps out, the last earlier sample it keeps is the sample
    before only when both are in one window of steps kept one after another. Otherwise it is
    `previous_kept_sample`, with a note: the steps between were left out, and it may be another
    incident's window (CLI-22, lane V5, Nov 19, 2026)."""
    step, layer = incident.get("step"), incident.get("layer")
    own = [s for s in evidence.get("samples") or () if isinstance(s, dict) and s.get("layer") == layer]
    sample = next((s for s in own if s.get("step") == step), None)
    if sample is None or not sample.get("routed_tokens"):
        return None
    tokens, counts = sample["routed_tokens"], sample.get("counts") or []
    top = sorted(range(len(counts)), key=lambda i: -counts[i])[:TOP_EXPERTS]
    out: dict[str, Any] = {
        "step": step, "layer": layer, "router": sample.get("router"), "routed_tokens": tokens,
        "experts": len(counts), "diversity": sample.get("diversity"),
        "largest_shares": [{"expert": i + 1, "share": _strict_number(counts[i] / tokens)} for i in top],
        "experts_with_no_tokens": sum(1 for c in counts if c == 0),
    }
    if sample.get("overload_rule"):
        out["overload_rule"] = sample["overload_rule"]
    before = [s for s in own if isinstance(s.get("step"), int) and s["step"] < step and s.get("routed_tokens")]
    if before:
        previous = max(before, key=lambda s: s["step"])
        found = {
            "step": previous["step"], "diversity": previous.get("diversity"),
            "largest_share": _strict_number(max(previous.get("counts") or [0]) / previous["routed_tokens"]),
        }
        marker = _compact._omits(evidence)
        windows = marker.get("kept_steps") if marker is not None else None
        same = marker is None or any(
            isinstance(w, list) and len(w) == 2 and w[0] <= previous["step"] and step <= w[1] for w in windows or ()
        )
        if same:
            out["previous_sample"] = found
        else:
            found["note"] = (
                f"the last sample before step {step} that this compact evidence keeps: the steps between were left out,"
                " so it is not known to be the sample before (start the server on the .jsonl capture or the run folder"
                " for every step)"
            )
            out["previous_kept_sample"] = found
    return out


def _look_next(incident: Mapping[str, Any], rank: int | None = None) -> list[dict[str, Any]]:
    """The `get_series` calls that show the data behind an incident, on `rank` when it is known:
    without it, a run of several ranks answers with the default rank's series (CLI-15, lane V5,
    Nov 19, 2026)."""
    calls: list[dict[str, Any]] = []

    def add(**arguments: Any) -> None:
        if rank is not None:
            arguments["rank"] = rank
        if len(calls) < 4 and arguments not in [c["arguments"] for c in calls]:
            calls.append({"tool": "get_series", "arguments": arguments})

    if isinstance(incident.get("router"), str):
        add(name=incident["router"], kind="router")
        for expert in (incident.get("overloaded") or incident.get("dead") or [])[:1]:
            add(name=incident["router"], kind="router", expert=expert)
    parameters = incident.get("parameters")
    for item in list(parameters)[:2] if isinstance(parameters, (list, Mapping)) else ():
        name = _name(item)
        if isinstance(name, str):
            add(name=name, kind="parameter")
    for item in (incident.get("modules") or [])[:2] if isinstance(incident.get("modules"), list) else ():
        if isinstance(item, Mapping) and isinstance(item.get("module"), str):
            if incident.get("kind") in ATTN_KINDS:  # the probe's attention module, not a layer (CLI-14)
                stat = "max_logit" if incident["kind"] == "attention_logit_growth" else "entropy_share"
                add(name=item["module"], kind="attention", stat=stat)
                continue
            stat = item.get("stat")
            add(name=item["module"], kind="layer", **({"stat": stat} if isinstance(stat, str) else {}))
    for name in (incident.get("layers") or [])[:2] if isinstance(incident.get("layers"), list) else ():
        if isinstance(name, str):
            add(name=name, kind="layer")
    for name in sorted(incident.get("scalars") or ())[:1] if isinstance(incident.get("scalars"), Mapping) else ():
        add(name=name, kind="scalar")
    return calls


def get_incident(source: Source, arguments: Mapping[str, Any]) -> dict[str, Any]:
    wanted = arguments.get("id")
    if not isinstance(wanted, str) or not wanted:
        raise ToolError("`id` is required: the id of an incident, as list_incidents gives it (for example grad_spike@310),"
                        " or the `mcp_id` the trigger handed over (for example my-run/rank0/grad_spike@310).")
    rank = _integer(arguments, "rank", minimum=0)
    lookup = _parse_lookup(wanted)
    if lookup is not None:
        # The trigger's lookup id: it names the run and the ranks itself (wave AH, P5-1).
        try:
            opened, entry, rank = _looked_up(source, wanted, lookup, rank)
        except _NotIdentifiable as unknown:
            # A stall's id whose episode cannot be shown to be the listed stall's: said outright,
            # and never answered with another stall's numbers.
            current = unknown.current
            return {
                "source": source.open(rank)["source"], "id": wanted, "identified": False,
                "result": "original_incident_no_longer_identifiable", "reason": unknown.reason,
                "current_stall": None if current is None else {
                    "id": current["id"], "kind": current["incident"].get("kind"), "step": current["incident"].get("step"),
                    "note": "a stall the run lists now, not established to be the one this id fired for; call"
                            " get_incident with this id to read it as a separate incident",
                },
            }
    else:
        opened = source.open(rank)
        entries = _entries(opened)
        entry = next((e for e in entries if e["id"] == wanted), None)
        if entry is None:
            known = ", ".join(e["id"] for e in entries[:20]) or "none"
            more = f" and {len(entries) - 20} more" if len(entries) > 20 else ""
            raise ToolError(
                f"no incident has the id {wanted!r}. The incidents here are: {known}{more}."
                " Call list_incidents for the current list." + _delivery_id_hint(wanted)
            )
    if opened["shape"] == "all_ranks" and rank is None:
        rank = opened["default_rank"]  # the server was started on a rank=<n> folder, as list_incidents reads it
    incident = entry["incident"]
    found = _class_of(incident)
    answer: dict[str, Any] = {
        "source": opened["source"], "id": entry["id"],
        # A stall looked up by what it was when the trigger fired (see `_looked_up`).
        **({"looked_up_as": entry["looked_up_as"],
            "note": "the stall this id named has changed its kind or step since the trigger fired; this is the run's"
                    " stall as it reads now"} if "looked_up_as" in entry else {}),
        "kind": incident.get("kind"), "class": found,
        "class_meaning": CLASS_MEANINGS.get(found, CLASS_MEANINGS["unclassified"]), "step": incident.get("step"),
        **_since_start(incident),
        "where": _where(incident),
    }
    group = entry.get("group")
    # The rank whose series show the data: the one asked for where it shows the incident, else
    # the one the numbers are from; one rank's evidence is of its own rank. None for an incident
    # of the run as a whole, or a file that does not say its rank.
    series_rank: int | None = opened["source"].get("rank") if opened["shape"] == "rank" else None
    if group is not None:
        ranks = group.get("ranks") or []
        numbers_from = group.get("incident_rank")
        series_rank = rank if rank is not None and rank in ranks else (
            numbers_from if isinstance(numbers_from, int) and not isinstance(numbers_from, bool) else None)
        answer["ranks"] = {
            "on_ranks": _ranges(ranks), "rank_count": len(ranks), "on_all_judged_ranks": group.get("on_all_judged_ranks"),
            "numbers_differ": group.get("numbers_differ"), "numbers_are_from_rank": group.get("incident_rank"),
        }
        if rank is not None and rank not in ranks:
            answer["ranks"]["note"] = f"rank {rank} is not among the ranks that show this incident"
        elif rank is not None and rank != group.get("incident_rank"):
            answer["ranks"]["note"] = (
                f"the numbers below are rank {group.get('incident_rank')}'s, not rank {rank}'s: one rank's incident is"
                " kept for each group of ranks (get_series with rank= reads any rank's own values)"
            )
    elif entry.get("run_level"):
        answer["ranks"] = {"on_ranks": "the run as a whole"}
    answer["incident"] = incident
    # L29 (lane AD5): what the evidence says beside this incident that bears on it, by its kind: the
    # skipped updates, the checkpoint's tensors, and its own kind's entry's keys `kinds` leaves out.
    # An all-ranks list keeps the ranks that show the incident.
    _carry_blocks(opened["evidence"], answer, INCIDENT_BLOCKS.get(str(incident.get("kind")), ()),
                  ranks=(group.get("ranks") or []) if group is not None else None)
    kind_keys = _kind_check_keys(opened["evidence"], [incident.get("kind")])
    if kind_keys is not None:
        answer["kind_checks"] = kind_keys
    # The score beside each ratio, in the words a reader shows (DETECTOR_RULES_CONTRACT.md, part 1),
    # and how the router's assignments were taken (part 7).
    scores = _scores(incident)
    if scores:
        answer["scores"] = scores
    if isinstance(incident.get("router"), str):
        answer.update(_routing(opened["evidence"].get("moe_source"), [incident["router"]]))
    if incident.get("kind") == "moe_imbalance":
        sample = _sample_context(opened["evidence"], incident)
        if sample is not None:
            answer["sample"] = sample
    calls = _look_next(incident, series_rank)
    if calls:
        answer["look_next"] = calls
    return answer


# --- Series -------------------------------------------------------------------------------


def _names(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, None]]:
    """The names step records carry, by kind, in the order they first appear."""
    found: dict[str, dict[str, None]] = {kind: {} for kind in SERIES_KINDS}
    fields = (("layers", "layer"), ("grad_norms", "parameter"), ("weight_norms", "parameter"), ("update_rms", "parameter"),
              ("moe", "router"), ("scalars", "scalar"))
    for record in records:
        for field, kind in fields:
            value = record.get(field)
            if isinstance(value, dict):
                found[kind].update(dict.fromkeys(value))
        for field in RUN_LEVEL:
            if record.get(field) is not None:
                found["scalar"].setdefault(field)
        probe = record.get("probe")
        if isinstance(probe, Mapping) and "error" not in probe:
            head, attention = probe.get("head"), probe.get("attention")
            if isinstance(head, Mapping) and isinstance(head.get("module"), str):
                found["head"].setdefault(head["module"])
            if isinstance(attention, Mapping):
                found["attention"].update(dict.fromkeys(k for k, v in attention.items() if isinstance(v, Mapping)))
    return found


def _sample_names(evidence: Mapping[str, Any]) -> dict[str, dict[str, None]]:
    found: dict[str, dict[str, None]] = {kind: {} for kind in SERIES_KINDS}
    for sample in evidence.get("samples") or ():
        if isinstance(sample, dict) and isinstance(sample.get("router"), str):
            found["router"].setdefault(sample["router"])
    return found


def _word(kind: str) -> str:
    return KIND_WORDS.get(kind, kind)


def _a(kind: str) -> str:
    word = _word(kind)
    return f"{'an' if word[0] in 'aeiou' else 'a'} {word}"


def _resolve(name: str, kind: str | None, names: Mapping[str, Mapping[str, None]], what: str) -> str:
    """The kind of series `name` is, or a `ToolError` that says what the names here are. Without
    `kind`, a name a probe kind shares with one other kind is that other kind (`PROBE_KINDS`)."""
    holders = [k for k in SERIES_KINDS if name in names[k]]
    if kind is not None and kind in holders:
        return kind
    if kind is None and len([k for k in holders if k not in PROBE_KINDS]) == 1:
        return next(k for k in holders if k not in PROBE_KINDS)
    if kind is None and len(holders) == 1:
        return holders[0]
    if kind is None and holders:
        raise ToolError(f"{name!r} is the name of {' and of '.join(map(_a, holders))} here: pass kind= to choose.")
    if holders:
        raise ToolError(f"{name!r} is {_a(holders[0])} here, not {_a(kind)}: pass kind={holders[0]!r}, or leave kind out.")
    # The probe's kinds are listed where the records hold them, or where they were asked for.
    shown = [k for k in SERIES_KINDS if k not in PROBE_KINDS or names[k] or k == kind]
    pool = {n: k for k in ([kind] if kind else shown) for n in names[k]}
    close = difflib.get_close_matches(name, list(pool), n=5, cutoff=0.5)
    counts = ", ".join(f"{len(names[k])} {_word(k)}{'s' if len(names[k]) != 1 else ''}" for k in shown)
    kinds = ", ".join(map(_word, shown[:-1])) + f" or {_word(shown[-1])}"
    text = f"no {_word(kind) if kind else kinds} is named {name!r} in {what}, which has {counts}."
    if close:
        text += " Closest names: " + ", ".join(f"{n} ({pool[n]})" for n in close) + "."
    else:
        first = [f"{n} ({k})" for k in ([kind] if kind else shown) for n in list(names[k])[:4]]
        text += (" Names here start: " + ", ".join(first) + ".") if first else ""
    raise ToolError(text + " Names are the ones an incident's `where` gives; a router is named, not numbered.")


def _columns(value: Any) -> dict[str, Any]:
    """One recorded value as named columns: a number is `value`, a list one column an entry."""
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items() if not isinstance(v, (Mapping, list))}
    if isinstance(value, list):
        return {f"group{i}": v for i, v in enumerate(value) if not isinstance(v, (Mapping, list))}
    return {"value": value}


def _router_point(counts: Any, diversity: Any, expert: int | None) -> dict[str, Any] | None:
    """A router's point from its counts, or None where it has none. The one rule for that is the
    detectors' (`evidence._router_total`): a list of signed 64-bit integers, no bool among them,
    whose sum is above 0; anything else is a missing sample in the evidence. Every reader of a
    record's counts here goes through it, so a series read from the records and the check of
    whether they are still what an evidence file judged (`_router_steps_changed`) hold a point at
    the steps the evidence does (wave AP, AO-05: `sum` raised on text, and a bool made a point the
    file never judged)."""
    tokens = _router_total(counts)
    if tokens is None or tokens <= 0:
        return None
    largest = max(range(len(counts)), key=lambda i: counts[i])
    point = {
        "diversity": diversity, "largest_share": counts[largest] / tokens, "largest_expert": largest + 1,
        "experts_with_no_tokens": sum(1 for c in counts if c == 0), "routed_tokens": tokens,
    }
    if expert is not None:
        point["expert_share"] = counts[expert - 1] / tokens if expert <= len(counts) else None
    return point


def _rows(kind: str, name: str, records: Iterable[Mapping[str, Any]], expert: int | None) -> list[tuple[int, dict[str, Any]]]:
    """`(step, columns)` for every record that carries the series."""
    rows: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        step = int(record["step"])
        if kind == "layer":
            layers = record.get("layers")
            stats = layers.get(name) if isinstance(layers, Mapping) else None  # section 1.5: no object, no point
            point = _columns(stats) if isinstance(stats, Mapping) else None
        elif kind == "parameter":
            point = {
                column: (record.get(field) or {})[name]
                for field, column in (("grad_norms", "grad_norm"), ("weight_norms", "weight_norm"), ("update_rms", "update_rms"))
                if isinstance(record.get(field), Mapping) and name in record[field]
            } or None
        elif kind == "router":
            moe_stats = record.get("moe_stats")
            stats = moe_stats.get(name) if isinstance(moe_stats, Mapping) else None
            diversity = stats.get("diversity") if isinstance(stats, Mapping) else None
            moe = record.get("moe")
            point = _router_point(moe.get(name) if isinstance(moe, Mapping) else None, diversity, expert)
        elif kind in PROBE_KINDS:
            point = _probe_point(kind, name, record.get("probe"))
        else:
            logged = record.get("scalars")
            value = logged[name] if isinstance(logged, Mapping) and name in logged else record.get(name) if name in RUN_LEVEL else None
            point = _columns(value) if value is not None else None
        if point:
            rows.append((step, point))
    return rows


def _merged_scalar_rows(name: str, rows: list[tuple[int, dict[str, Any]]], records: Iterable[Mapping[str, Any]],
                        scalar_only: Iterable[Mapping[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """A logged scalar's rows with the scalar-only records' points merged in by step
    (WAVE_AL_CONTRACT.md section 4.6, Q6 a: at a step one of them has, its point is the one row,
    the highest `seq`'s), in step order. A scalar that no scalar-only record names keeps its rows."""
    standing = merged_scalar_points((), scalar_only).get(name)
    if not standing:
        return rows
    kept = [(step, point) for step, point in rows if step not in standing]
    return sorted([*kept, *((step, _columns(value)) for step, value in standing.items())], key=lambda row: row[0])


def _probe_point(kind: str, name: str, probe: Any) -> dict[str, Any] | None:
    """One probe's point of the head or of an attention module, or None where the probe failed,
    did not read the head, or did not see the module. The head gives its numbers
    (`HEAD_COLUMNS`); an attention module what the detectors judge (`_attention_values`): the
    mean of its heads' finite entropy shares, its largest finite logit and how many heads had a
    share, with the rows counted. Never the per-head lists or the attention map, which are not
    one value a step, nor the head's text fields (its module, dtypes)."""
    if not isinstance(probe, Mapping) or "error" in probe:
        return None
    if kind == "head":
        head = probe.get("head")
        if not isinstance(head, Mapping) or head.get("module") != name or "not_read" in head:
            return None
        return {column: head[column] for column in HEAD_COLUMNS
                if column in head and (head[column] is None or _is_number(head[column]))} or None
    attention = probe.get("attention")
    stats = attention.get(name) if isinstance(attention, Mapping) else None
    if not isinstance(stats, Mapping):
        return None
    share, logit, heads = _attention_values(stats)
    point: dict[str, Any] = {"entropy_share": share, "max_logit": logit, "heads": heads}
    for column in ("masked_rows", "invalid_rows", "rows"):
        if _is_number(stats.get(column)):
            point[column] = stats[column]
    return point


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _not_read(kind: str, name: str, records: Iterable[Mapping[str, Any]]) -> str:
    """Why the probe's head has no point: the first probe's reason, when one gave one."""
    if kind != "head":
        return ""
    for record in records:
        probe = record.get("probe")
        head = probe.get("head") if isinstance(probe, Mapping) else None
        if isinstance(head, Mapping) and head.get("module") == name and "not_read" in head:
            return f" The probe did not read it: {head['not_read']} (step {record.get('step')})."
    return ""


def _sample_rows(name: str, evidence: Mapping[str, Any], expert: int | None) -> list[tuple[int, dict[str, Any]]]:
    """A router's series from the detectors' per-step samples, which also say how many experts
    each sample has flagged overloaded and dead."""
    rows: list[tuple[int, dict[str, Any]]] = []
    for sample in evidence.get("samples") or ():
        if not isinstance(sample, dict) or sample.get("router") != name or sample.get("missing"):
            continue
        point = _router_point(sample.get("counts"), sample.get("diversity"), expert)
        if point is not None:
            point["overloaded"] = len(sample.get("overloaded") or ())
            point["dead"] = len(sample.get("dead") or ())
            rows.append((int(sample["step"]), point))
    return sorted(rows, key=lambda row: row[0])


def _strict_number(value: Any) -> Any:
    """A recorded value for a reply: six significant digits, and a NaN or Inf as the string the
    capture writes for it, because strict JSON has neither."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return float(f"{value:.6g}")
    return value


def _too_large(value: Any) -> bool:
    """An integer no float holds (10**400: `Session.log` takes it and the capture writes it)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    try:
        float(value)
    except OverflowError:
        return True
    return False


def _finite(value: Any) -> int | float | None:
    """A finite number as recorded, else None: a bool, anything not a number, NaN, an Inf and an
    integer too large for a float (`_too_large`, BUG-25, wave AO: `math.isfinite` raised on it)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or _too_large(value) or not math.isfinite(value):
        return None
    return value


def _over_range(rows: list[tuple[int, dict[str, Any]]], column: str) -> dict[str, Any]:
    """What a column did over every recorded step in the range, sampled or not. An integer too
    large for a float is counted apart (`too_large_for_a_float`), out of min, max and median: it
    is finite, and no float holds it."""
    held = [(step, point[column]) for step, point in rows if point.get(column) is not None]
    finite = [(step, number) for step, value in held if (number := _finite(value)) is not None]
    out: dict[str, Any] = {"recorded": len(held)}
    if finite:
        low, high = min(finite, key=lambda item: item[1]), max(finite, key=lambda item: item[1])
        out.update({
            "min": _strict_number(low[1]), "min_step": low[0], "max": _strict_number(high[1]), "max_step": high[0],
            "median": _strict_number(statistics.median(number for _, number in finite)),
            "first": _strict_number(finite[0][1]), "last": _strict_number(finite[-1][1]),
        })
    large = [step for step, value in held if _too_large(value)]
    bad = [step for step, value in held if _finite(value) is None and not _too_large(value)]
    if bad:
        out.update({"not_finite": len(bad), "first_not_finite_step": bad[0]})
    if large:
        out.update({"too_large_for_a_float": len(large), "first_too_large_step": large[0]})
    return out


def get_series(source: Source, arguments: Mapping[str, Any], fits: Callable[[dict[str, Any]], bool] = lambda _: True) -> dict[str, Any]:
    name = arguments.get("name")
    if not isinstance(name, str) or not name:
        raise ToolError("`name` is required: a layer (module), parameter, router, logged scalar, or the probe's head or"
                        " attention module, as an incident's `where` names it.")
    kind = arguments.get("kind")
    kind = "layer" if kind == "module" else kind
    if kind is not None and kind not in SERIES_KINDS:
        raise ToolError(f"`kind` must be one of {', '.join(SERIES_KINDS)}, not {kind!r}.")
    stat = arguments.get("stat")
    if stat is not None and not isinstance(stat, str):
        raise ToolError("`stat` must be a string, such as abs_max or grad_norm.")
    rank = _integer(arguments, "rank", minimum=0)
    expert = _integer(arguments, "expert", minimum=1)
    start, end = _integer(arguments, "start"), _integer(arguments, "end")
    if start is not None and end is not None and start > end:
        raise ToolError(f"`start` ({start}) is after `end` ({end}).")
    max_points = _integer(arguments, "max_points", minimum=1, maximum=MAX_POINTS) or DEFAULT_POINTS

    held = source.rank_records(rank)
    origin: dict[str, Any]
    evidence: Mapping[str, Any] | None = None
    # Compact evidence that left samples out (`compact._omits`): with the records held, a router's
    # series is read from them and only its flags from this file's samples (CLI-06, Nov 18, 2026).
    window: Mapping[str, Any] | None = None
    compact_note: str | None = None
    other_note: str | None = None
    if held is None:
        opened = source.open(rank)
        if opened["shape"] != "rank":
            raise ToolError(
                "this all-ranks evidence file has no step records, and its run folder is not on this machine:"
                " start the server on the run folder (runs/<run_id>), or with --records, to read series."
            )
        evidence, names, origin = opened["evidence"], _sample_names(opened["evidence"]), {"records": None}
        omits = _compact._omits(evidence) is not None
        what = (
            f"the {'compact ' if omits else ''}evidence file {source.path}, which has only its routers' samples"
            f"{' of the steps it keeps' if omits else ''} (start the server on the .jsonl capture or the run folder,"
            " or with --records, for layers and parameters)"
        )
        if omits:
            sampled = evidence["compact"].get("sampled_steps") or [None, None]
            compact_note = (
                "This series is read from this file's samples alone, so its steps, recorded and over_range cover only"
                f" the steps it keeps, not the run, whose samples run from step {sampled[0]} to {sampled[-1]}. The"
                f" file is {_compact.describe(evidence)}. For every step, start the server on the .jsonl capture or"
                " the run folder, or with --records."
            )
    else:
        records, names, origin, scalar_only = held
        what = f"rank {origin['rank']}'s step records"
        if (kind or "router") == "router" and name in names["router"]:
            path_kind, path, _ = source.locate()
            if path_kind == "capture":
                evidence = source.capture_evidence(path, origin["rank"])
            elif path_kind == "evidence" and source.document(path).get("view") != ALL_RANKS_VIEW:
                document = source.document(path)
                same = _same_rank_and_run(document, origin["rank"], records)
                omits = same and _compact._omits(document) is not None
                changed = None if not same or omits else _router_steps_changed(name, document, records)
                if same and changed is None:
                    evidence = document
                    if omits:
                        window, evidence = evidence, None
                else:
                    # Another rank's or run's evidence, or one that does not say whose (lane U7,
                    # Nov 19, 2026: rank 0's samples were answered as rank 1's); or this rank's, of
                    # records that changed after it was written (BUG-25, wave AO: the capture grew
                    # and the file's older samples answered). This rank is read from its own
                    # records, and judged on them where they are a capture.
                    held_in = Path(origin["records"])
                    if held_in.is_file() and held_in.suffix == ".jsonl":
                        evidence = source.capture_evidence(held_in, origin["rank"])
                    judged = _judged_rank(document)
                    other_note = (
                        f"The evidence file {source.path} judged "
                        + (changed if changed is not None else
                           (f"rank {judged}" if judged is not None else "a rank it does not name")
                           + (f" of run {document.get('run_id')!r}" if document.get("run_id") else "")
                           + f", not rank {origin['rank']}'s records read here")
                        + ", so its samples and flags are not used:"
                        + (" this series and its flags are from the detectors run on those records."
                           if evidence is not None else " this series is from those records, with no flags.")
                    )
    asked = kind
    kind = _resolve(name, kind, names, what)
    also = [k for k in PROBE_KINDS if k != kind and name in names[k]] if asked is None else []
    if expert is not None and kind != "router":
        raise ToolError(f"`expert` is for a router's series; {name!r} is {_a(kind)}.")
    if evidence is not None and any(isinstance(s, dict) and s.get("router") == name for s in evidence.get("samples") or ()):
        rows = _sample_rows(name, evidence, expert)
    else:
        rows = _rows(kind, name, records if held is not None else (), expert)
        if kind == "scalar" and held is not None and scalar_only:
            rows = _merged_scalar_rows(name, rows, records, scalar_only)
        if window is not None and kind == "router":
            # The flags are the detectors' as this file has them: a rebuild from the records could
            # use other thresholds or detectors than the ones that wrote it.
            kept = dict(_sample_rows(name, window, None))
            for step, point in rows:
                if step in kept:
                    point.update(overloaded=kept[step]["overloaded"], dead=kept[step]["dead"])
            compact_note = (
                "The routing at every recorded step is read from the step records. overloaded, dead and"
                " first_flagged_step are the detectors' flags from the evidence file's own samples, so they cover only"
                " the steps it keeps, and first_flagged_step is the first flagged among those. The file is"
                f" {_compact.describe(window)}."
            )
    if not rows:
        raise ToolError(f"no step carries values for the {_word(kind)} {name!r} yet."
                        + _not_read(kind, name, records if held is not None else ()))
    columns = list(dict.fromkeys(column for _, point in rows for column in point))
    if expert is not None and all(point.get("expert_share") is None for _, point in rows):
        raise ToolError(f"the router {name!r} has no expert {expert}: its experts are numbered from 1.")
    if stat is not None:
        if stat not in columns:
            raise ToolError(f"the {_word(kind)} {name!r} has no `{stat}`. It has: {', '.join(columns)}.")
        columns = [stat]
    else:  # asked for by name only: the token count is the same at every step of most runs
        columns = [column for column in columns if column != "routed_tokens"]
    first, last = rows[0][0], rows[-1][0]
    inside = [row for row in rows if (start is None or row[0] >= start) and (end is None or row[0] <= end)]
    if not inside:
        raise ToolError(
            f"the {_word(kind)} {name!r} has no recorded step between {start if start is not None else first} and"
            f" {end if end is not None else last}: it is recorded at {len(rows)} steps, from {first} to {last}."
            + (f" {compact_note}" if compact_note is not None and held is None else "")
        )
    stride = math.ceil(len(inside) / max_points)
    while True:
        shown = inside[::stride]
        answer: dict[str, Any] = {
            "name": name, "kind": kind, **origin,
            "recorded": {"first_step": first, "last_step": last, "steps": len(rows)},
            "range": {"start": inside[0][0], "end": inside[-1][0], "steps": len(inside)},
            "stride": stride, "points": len(shown), "steps": [step for step, _ in shown],
            "values": {column: [_strict_number(point.get(column)) for _, point in shown] for column in columns},
            "over_range": {column: _over_range(inside, column) for column in columns if column != "largest_expert"},
        }
        if kind == "router":
            answer["experts_are_numbered_from"] = 1
            # How the routing reached the capture (part 7): said once, on the first record that knows.
            said = next((r["moe_source"] for r in (records if held is not None else ())
                         if isinstance(r.get("moe_source"), Mapping) and name in r["moe_source"]),
                        (evidence or {}).get("moe_source"))
            answer.update(_routing(said, [name]))
            flagged = [step for step, point in inside if point.get("overloaded") or point.get("dead")]
            if flagged:
                answer["first_flagged_step"] = flagged[0]
        if compact_note is not None:
            answer["compact"] = compact_note
        if other_note is not None:
            answer["evidence_file"] = other_note
        if also:
            answer["also"] = (f"{name!r} is also the name of {' and of '.join(map(_a, also))} of the probe: pass "
                              + " or ".join(f"kind={k!r}" for k in also) + " for that series")
        if stride > 1:
            answer["note"] = (
                f"every {stride}th recorded step is shown; over_range covers every recorded step in the range, so a"
                " one-step spike is in max and max_step even when its point is not shown. Narrow start and end to see each step."
            )
        if fits(answer) or len(shown) <= 1:
            return answer
        stride *= 2


# --- The all-ranks status -----------------------------------------------------------------


def get_all_ranks_status(source: Source, arguments: Mapping[str, Any]) -> dict[str, Any]:
    rank = _integer(arguments, "rank", minimum=0)
    kind, where, _ = source.locate()
    if kind == "capture" or (kind == "evidence" and source.document(where).get("view") != ALL_RANKS_VIEW):
        raise ToolError(
            f"{source.path} is one rank's {'capture' if kind == 'capture' else 'evidence file'}, so it has no all-ranks"
            " status: nothing is known here about the job's other ranks. list_incidents gives this rank's run state."
            " For every rank, start the server on the run folder in the server's store (runs/<run_id>)."
        )
    opened = source.open(None)
    evidence = opened["evidence"]
    entries = _entries(opened)
    answer: dict[str, Any] = {"source": opened["source"]}
    state = evidence.get("run_state")
    answer["run_state"] = state if isinstance(state, dict) else None
    answer["ranks"] = _rank_counts(evidence)
    by_state = evidence.get("ranks_by_state") if isinstance(evidence.get("ranks_by_state"), dict) else {}
    rows = {e.get("rank"): e for e in evidence.get("ranks") or () if isinstance(e, dict)}
    answer["reports"] = [
        {"state": name, "count": len(ranks), "ranks": _ranges(ranks),
         "reason": ((rows.get(ranks[0]) or {}).get("detection") or {}).get("reason") if ranks else None}
        for name, ranks in by_state.items() if isinstance(ranks, list)
    ]
    answer["last_step"] = evidence.get("last_step")
    answer["incidents"] = [_brief(entry) for entry in entries[:DEFAULT_INCIDENTS]]
    if len(entries) > DEFAULT_INCIDENTS:
        answer["incidents_omitted"] = len(entries) - DEFAULT_INCIDENTS
    answer["not_checked"] = _not_checked(opened)
    _kinds(opened, answer)
    if isinstance(evidence.get("watchdog"), dict):
        answer["watchdog"] = evidence["watchdog"]
    if isinstance(evidence.get("stopped_stacks"), dict):
        answer["stopped_stacks"] = evidence["stopped_stacks"]
    dropped = _dropped(opened)
    if dropped is not None:  # L22: no key when no rank dropped a row
        answer["dropped_rows"] = _dropped_shown(dropped)
    refused = _refused(opened)
    if refused is not None:  # RS10: no key when no rank refused a batch
        answer["refused_batches"] = _dropped_shown(refused)
    not_newer = _not_newer(opened)
    if not_newer is not None:  # L30: no key when no rank has rows not newer
        answer["rows_not_newer"] = _dropped_shown(not_newer)
    if rank is not None:
        answer["rank"] = _rank_entry(evidence, rank)
        mine = _rank_dropped(dropped, rank)
        if mine is not None:
            answer["rank"] = {**answer["rank"], "dropped_rows": mine}  # a copy: the entry is the kept evidence's
        theirs = _rank_dropped(refused, rank)
        if theirs is not None:
            answer["rank"] = {**answer["rank"], "refused_batches": theirs}
        late = _rank_not_newer(not_newer, rank)
        if late is not None:
            answer["rank"] = {**answer["rank"], "rows_not_newer": late}
    elif len(rows) <= PER_RANK_ROWS:
        answer["per_rank"] = [
            {"rank": number, "report": (entry.get("detection") or {}).get("state"),
             "judged": (entry.get("detection") or {}).get("judged"),
             "last_step": (entry.get("detection") or {}).get("report_last_step"),
             **({"incidents": [i.get("kind") for i in entry["incidents"] if isinstance(i, dict)]}
                if isinstance(entry.get("incidents"), list) else {}),
             # Of the kinds this rank's own report judges (`SERVER_KINDS`): where a rank differs from the others.
             **({"kinds": {key: entry["kind_counts"].get(key) for key in (
                 "checked", "not_enough_history", "not_configured", "unavailable")}}
                if isinstance(entry.get("kind_counts"), dict) else {}),
             **({"stream": entry["stream"]} if isinstance(entry.get("stream"), dict) else {}),
             **({"dropped_rows": mine["rows"]} if (mine := _rank_dropped(dropped, number)) is not None else {}),
             **({"refused_batches": theirs["batches"]} if (theirs := _rank_dropped(refused, number)) is not None
                else {}),
             # L30: each count as the status writes it, each only when the rank has it.
             **({"rows_not_newer": late["rows"],
                 **({"late_steps_confirmed": late["late_steps_confirmed"]} if "late_steps_confirmed" in late else {})}
                if (late := _rank_dropped(not_newer, number)) is not None else {})}
            for number, entry in sorted(rows.items(), key=lambda item: item[0] if isinstance(item[0], int) else -1)
        ]
    else:
        answer["per_rank_note"] = f"{len(rows)} ranks: pass rank= for one rank's entry"
    # L29 (lane AD5): each judged rank's skipped updates and checkpoint tensors, and what `kinds`
    # leaves out of a kind's entry for the run (replica coverage, the no-peer words).
    _carry_blocks(evidence, answer, ranks=[rank] if rank is not None else None)
    kind_keys = _kind_check_keys(evidence)
    if kind_keys is not None:
        answer["kind_checks"] = kind_keys
    # A rank whose capture stopped, as the server's status says it (DETECTOR_RULES_CONTRACT.md, part 7):
    # its `capture_stopped` as written, so `step` 0, `step` null and no `step` stay three things.
    stopped = _capture_stopped(opened)
    if rank is not None and rank in stopped:
        answer["rank"] = {**answer["rank"], "capture_stopped": stopped[rank]}
    for item in answer.get("per_rank") or ():
        if item.get("rank") in stopped:
            item["capture_stopped"] = stopped[item["rank"]]
    summary = _summary(evidence)
    if summary:
        answer["summary"] = summary
    return answer


# --- Other runs -----------------------------------------------------------------------------

_ROW_KEYS = ("project", "name", "run_id", "group", "tags", "has_manifest", "source", "started", "run_state", "last_step",
             "incidents", "first_incident", "kinds_checked", "ranks", "evidence", "rank_count", "ranks_with_incidents",
             "kinds_not_stated", "rank_count_not_known", "end", "dropped_rows", "refused_batches", "rows_not_newer",
             "freshness")
_BOUNDED_ROW_KEYS = ("dropped_rows", "refused_batches", "rows_not_newer")  # their ranks listed up to PER_RANK_ROWS


def _text_argument(arguments: Mapping[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is not None and (not isinstance(value, str) or not value):
        raise ToolError(f"`{name}` must be a string that is not empty, not {value!r}.")
    return value


def list_runs(source: Source, arguments: Mapping[str, Any]) -> dict[str, Any]:
    folder = source.runs_folder()
    project = _text_argument(arguments, "project")
    limit = _integer(arguments, "limit", minimum=1, maximum=MAX_RUNS) or DEFAULT_RUNS
    try:
        rows = _runs.list_runs(folder, None)
    except (OSError, ValueError) as error:
        raise ToolError(f"the runs at {folder} cannot be listed ({error}).") from None
    projects: dict[str, int] = {}
    for row in rows:
        projects[row["project"]] = projects.get(row["project"], 0) + 1
    if project is not None:
        if project not in projects:
            known = ", ".join(f"{name} ({count})" for name, count in projects.items()) or "none"
            raise ToolError(f"no run of project {project!r} at {folder}. Projects there: {known}.")
        rows = [row for row in rows if row["project"] == project]
    shown = rows[:limit]
    return {
        "folder": str(folder),
        "projects": projects,
        "count": len(rows),
        **({"shown": len(shown), "note": f"{len(rows) - len(shown)} more runs: pass project= or a larger limit"}
           if len(shown) < len(rows) else {}),
        # A row says nothing it did not read: `run_state` null with `evidence.state` "missing"
        # means no evidence file was written for that run, not that the run is clean.
        "runs": [{key: _dropped_shown(row[key]) if key in _BOUNDED_ROW_KEYS else row[key] for key in _ROW_KEYS if key in row}
                 for row in shown],
    }


def compare_runs(source: Source, arguments: Mapping[str, Any]) -> dict[str, Any]:
    folder = source.runs_folder()
    wanted = arguments.get("runs")
    if not isinstance(wanted, list) or not 2 <= len(wanted) <= MAX_COMPARED or not all(isinstance(w, str) and w for w in wanted):
        raise ToolError(f"`runs` must be a list of 2 to {MAX_COMPARED} run IDs or names, as list_runs gives them.")
    project = _text_argument(arguments, "project")
    rank = _integer(arguments, "rank", minimum=0)
    try:
        result = _compare.compare_runs(*wanted, source=folder, project=project, rank=rank, paths=False,
                                       read_parts=False)
    except MalformedRecordError as error:
        raise ToolError(str(error)) from None  # the error's own text (section 1)
    except (ValueError, OSError, RuntimeError) as error:
        # A run that is not there, a name that fits several runs (the message lists their IDs),
        # or records that cannot be read.
        raise ToolError(f"{error} (the runs are looked up at {folder}; list_runs shows them)") from None
    # L22 (lane AA5): a store run's dropped rows, and its refused batches (RS10, lane AB5), with the
    # ranks bounded as the other answers bound them.
    result["runs"] = [{**run, **{key: _dropped_shown(run[key]) for key in _BOUNDED_ROW_KEYS if key in run}}
                      for run in result["runs"]]
    return result


# --- The tools, as the protocol lists them --------------------------------------------------

# What the two counts of L30 are and are not (WAVE_AE_CONTRACT.md, section 6), in the page's words, said
# once for the tools that show dropped rows. `{where}` is where the tool puts the block.
_NOT_NEWER_WORDS = (
    " `{where}`, only when the server received rows of {whose} and did not store them because their step was not"
    " above a step it had already kept for their attempt: `rows` is how many such rows, and a row sent again is"
    " counted again, so it is not a number of steps lost. `late_steps_confirmed`, beside it only when above zero,"
    " is how many distinct steps among them are confirmed to have arrived out of order inside one stored part, and"
    " are not stored. It is a lower bound: a late step that crosses a part boundary is not in it, so the same rows"
    " can give another count when the parts are cut elsewhere. With no `late_steps_confirmed`, none is confirmed"
    " as a step that arrived out of order. The other rows, if any, are not told apart from rows sent again. `note` says"
    " this with the numbers. Neither count is added to `dropped_rows`. With no key, the server reported no such row."
)

# What the L29 blocks mean, and what their absence means, said once for the three tools (lane AD5).
_SKIPS_WORDS = (
    "`skipped_updates`, only where a step record says who skipped an update: the updates skipped by who"
    " reported them (`caller`, `scaler`, `both`), with `partial`, only when above zero, for the steps on which the scaler skipped some optimizers and applied others (counted there and under no other name; an update occurred on such a step) and, under `non_finite_gradients` (`steps`, `first_step`,"
    " `gradients`), the steps where a skip the caller reported had a gradient that was not finite. Such a"
    " gradient raises no incident (a skipped update is not judged on its gradient), so this is the only place it"
    " shows. With no key, no record says who skipped an update: none was skipped, or the capture was written"
    " before that was recorded."
)
_TENSORS_WORDS = (
    " `checkpoint_tensors`, only where a judged resume's load recorded what it found of the checkpoint's tensors:"
    " `resumes` counts those loads and `entries` gives the first five by step, each with its `step`, `rank`,"
    " `attempt` and `state`. `compared` means the checkpoint entry named by `entry` was matched with the model's"
    " tensors (`by` says how the entry was found; `model`, `trainable`, `checkpoint`, `matched` count them; `missing`, `missing_frozen`, `missing_buffers`,"
    " `unexpected` and `shape_differs` each give a `count` and `names`); `unavailable` means they could not be,"
    " and `why` says why (for example a folder, an ambiguous source, a model state not identified). No"
    " resume_incomplete incident means every trainable tensor was there only for a resume whose entry says"
    " `compared`. With no key, no load recorded its tensors, so nothing is said of them."
)
_BY_RANK_WORDS = (
    " For a run folder they are `skipped_updates_by_rank` and `checkpoint_tensors_by_rank`: one entry"
    " (`{\"rank\", ...}`) for each judged rank whose report has the block, listed up to"
    f" {PER_RANK_ROWS} (`skipped_updates_by_rank_omitted` or `checkpoint_tensors_by_rank_omitted` counts the rest),"
    " and with `rank`, that rank's entry only; a rank with no entry has no such block, and a rank that is not"
    " judged is in neither."
)
_KIND_KEYS_WORDS = (
    " `kind_checks`, only for kinds whose entry has one of them, gives what `kinds` leaves out of the entry:"
    " `stalled_weights.parameters_not_judged`, the parameters whose gradient could not be read, so no stall was"
    " looked for on them (the kind is still checked on the rest); `replica_drift.coverage`, how many of the ranks'"
    " replica step entries compared fingerprints (`steps`, `compared`), the ranks missing from any"
    " (`missing_ranks`), and the latest steps not compared with why (`not_compared`, `not_compared_omitted` the"
    f" rest); and `note` \"{NO_PEER_WORDS}\" on a cross-rank kind (`duplicate_batches`, `replica_drift`) that saw"
    " one rank and had no other to compare it with, which is why it reads not_configured. With no `kind_checks`,"
    " no kind's entry has any of these."
)
# The keys the detector rules contract adds (DETECTOR_RULES_CONTRACT.md, wave AD), said once (lane AD5).
_MEASUREMENTS_WORDS = (
    " `gradient_trend`, `attention_trends`, `update_trends` and `weight_jumps`, each only where the records hold"
    " what it reads, are descriptions of the run: numbers with no threshold, which no rule reads and which change"
    " no state. They are not findings or warnings, and a large `growth`, a low `update_trends` ratio or a large"
    " `weight_jumps` ratio is not by itself a fault (a healthy run can rise more than a faulty one). With no key the"
    " records do not hold what it reads (`gradient_trend` needs six points)."
)
_RULES_WORDS = (
    " A `moe_imbalance` incident with `since_start` true began at `step` and became reportable only at"
    " `reportable_step` (`when` says \"collapsed since step S, reportable at step R\"): it was not found at `step`."
    " `kind_checks.moe_imbalance.starting_state` lists routers still flagged on every sample since step 0 and"
    " not yet reportable (`{\"router\", \"samples\"}`); the kind then has not enough history for them."
    " `kind_checks.<kind>.modules_not_judged` names modules an activation kind did not judge because the capture"
    " could not read them (the `layers` entry of not_checked, with `coverage` `partial`; with `coverage` `none` the"
    " kinds are unavailable). `stalled_weights.parameters_not_judged_by_rank` gives each rank's list when ranks"
    " differ. `moe_source`, only where the capture said how it read the routing, gives each router's `form`"
    " (scores, indices or map: the representation only) and `point`; `moe_source_words` says what the assignments"
    " are: \"the dispatched assignments\" only for `point` dispatched, \"the assignments the router asked for\""
    " for requested, and for unknown that whether a capacity limit came first is not known. With no `moe_source`,"
    " nothing is said of where the routing was taken."
)
_SCORES_WORDS = (
    " `scores` lists each `ratio` of the incident that has a `score` beside it (the incident's own, then each"
    " parameter's or module's, named under `at`), with `display`: \"score 31.2\", \"no score: fewer than 10 earlier"
    " values\" or \"no score: the earlier values had zero median absolute deviation\". A null score is no score,"
    " never 0. The score is how far the value is from the median of the earlier values, in units of their spread: a"
    " description, from which nothing is judged. An incident written before the score has no `scores`. A"
    " `moe_imbalance` incident with `since_start` has `when` (\"collapsed since step S, reportable at step R\");"
    " an `update_spike` incident may carry `lr` and `lr_before`, the learning rates at the step and at the step it"
    " was compared with; a router's incident carries its `moe_source` and `moe_source_words` where the capture said."
)

_INCIDENT_BLOCKS_WORDS = (
    " By its kind, and only where the evidence has them, it also carries what bears on it, as list_incidents"
    " describes each: `skipped_updates` for non_finite, grad_spike, update_spike, stalled_weights and"
    " loss_scale_trouble, `checkpoint_tensors` for resume_incomplete (for a run folder the `_by_rank` lists, of the"
    " ranks that show the incident), and `kind_checks` with its own kind's `parameters_not_judged`, `coverage` or"
    " the no-peer `note`."
)

_RANK = {
    "type": "integer", "minimum": 0,
    "description": "The rank to read, when the path holds several. Default: the path's own rank, else the lowest.",
}
_READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_incidents",
        "title": "List incidents",
        "description": (
            "Call this first. Lists what TensorQuill's detectors found in the run: each incident's id, kind, class"
            " (invariant, anomaly or observation), first step and `where` (the MoE layer and router with expert"
            " numbers, or the parameters or modules it names), in the order the evidence lists them. Also gives the"
            " run state with its reasons, what was not checked, and a text summary. An empty list is not by itself"
            " \"no failure\": read the run state's reasons and `kinds`. With no incidents, a state of healthy means"
            " the recorded steps were judged and nothing was found, for the kinds listed under `kinds.checked`"
            " only. A kind under `kinds.not_enough_history` is checked but its rule is still warming up: it has"
            " seen `records` of the `records_needed` it needs, could not have fired yet, and nothing found is not a"
            " verdict for it, though the state stays healthy (with `reported`, one of its faster rules did find the"
            " listed incident, which stands). A kind under `kinds.not_configured` was never looked"
            " for, because the capture does not record what it needs; one under `kinds.unavailable` was asked for"
            " and could not be judged, and makes the state degraded. Kinds under `kinds.history_not_stated` come"
            " from a file written before this was recorded: treat them as not enough history. When the answer has"
            " no `kinds` (a file from an older version, or a run folder with `kinds_not_stated`, which gives the"
            " reason), what was checked is not stated: do not infer it. A not_checked entry or an unknown rank"
            " means that part was not judged."
            " For a run folder the incidents are grouped across ranks, and `rank` keeps one rank's; `kinds` is"
            " then the whole run's, read as get_all_ranks_status describes; `rank.dropped_rows`, only when the server"
            " dropped rows of that rank (for a null key or a schema version it does not support), counts them (an"
            " upper bound); `rank.refused_batches`, only when it refused whole batches of that rank because a key"
            " column was missing or of another type, counts those batches apart (never added to the rows)."
            + _NOT_NEWER_WORDS.format(where="rank.rows_not_newer", whose="that rank") +
            " For a local .jsonl file, `run_end` says whether its session closed it: any state but `closed` means not"
            " known to have finished (still running, killed or never closed look the same)."
            " Where one rank's evidence has them, `head_trends` gives the probe's language-model head numbers over"
            " the run: values with no threshold, never an incident (get_series with kind head gives each probe's). "
            + _SKIPS_WORDS + _TENSORS_WORDS + _BY_RANK_WORDS + _KIND_KEYS_WORDS + _MEASUREMENTS_WORDS + _RULES_WORDS
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "rank": _RANK,
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_INCIDENTS,
                          "description": f"Most incidents to list (default {DEFAULT_INCIDENTS})."},
            },
            "additionalProperties": False,
        },
        "annotations": {"title": "List incidents", **_READ_ONLY},
    },
    {
        "name": "get_incident",
        "title": "Get one incident",
        "description": (
            "One incident in full, by the id list_incidents gave (for example grad_spike@310), or by the `mcp_id` a"
            " trigger handed over (`TENSORQUILL_INCIDENT_MCP_ID`, for example my-run/rank0/grad_spike@310, which names"
            " its run and ranks and needs no `rank`; a stall's is resolved only on the evidence file the trigger"
            " named, and when the stall listed now cannot be shown to be the one that fired, the answer is"
            " `original_incident_no_longer_identifiable` with the current stall's own id): every number the"
            " detector recorded, what its class means, the ranks that show it, for an MoE incident the sample it"
            " points at (largest expert shares, and the sample before; from compact evidence that left the steps"
            " between out, the last earlier one it keeps is `previous_kept_sample` instead), and the get_series"
            " calls that show the data behind it, with the rank to read." + _INCIDENT_BLOCKS_WORDS + _SCORES_WORDS
            + " By its kind it also carries the measurement that bears on it (`gradient_trend` for grad_spike,"
            " `update_trends` and `weight_jumps` for update_spike, `attention_trends` for the attention kinds): a"
            " description, never a finding."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The incident's id from list_incidents, as <kind>@<step>."},
                "rank": _RANK,
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        "annotations": {"title": "Get one incident", **_READ_ONLY},
    },
    {
        "name": "get_series",
        "title": "Get a series",
        "description": (
            "The recorded values of one layer (module: mean, std, abs_max, zero_frac), parameter (grad_norm, update_rms,"
            " weight_norm), MoE router (diversity, largest share and expert, experts with no tokens; `expert` adds"
            " that expert's share), logged scalar (such as the loss, or grad_norm and lr for the whole model), or,"
            " from the probe, its language-model head (kind head: norm, mean, log_z, gap, gap_relative, positions)"
            " or an attention module (kind attention: entropy_share, the mean of its heads' finite shares; max_logit;"
            " heads; the rows counted) over a step range. A name that is also a layer's is the layer unless"
            " kind says otherwise, and the answer's `also` says so."
            f" At most `max_points` points come back (default {DEFAULT_POINTS}): a longer range is"
            " thinned to every nth recorded step, with the stride reported, and `over_range` gives the minimum,"
            " maximum (with their steps) and median over every recorded step in the range, so a one-step spike is"
            " not lost. Narrow `start` and `end` to see each step. Numbers have six significant digits; a NaN or"
            ' Inf is the string "nan", "inf" or "-inf". An integer too large for a float is shown as recorded and'
            " counted in `over_range` as `too_large_for_a_float`, out of its minimum, maximum and median."
            " A router's series from compact evidence that left steps"
            " out says under `compact` which steps it covers. A rank's records that hold more than one run ID are"
            " refused with the IDs, as list_incidents refuses them: a series is of one run. A router's series carries"
            " `moe_source` and `moe_source_words` where the capture said how it read the routing: \"the dispatched"
            " assignments\" only when `point` is dispatched."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "The layer (module), parameter, router, scalar, head or attention module name,"
                                        " as an incident's `where` gives it."},
                "kind": {"type": "string", "enum": list(SERIES_KINDS), "description": "Only needed when one name is used for two kinds."},
                "stat": {"type": "string", "description": "One column only, such as abs_max or grad_norm. Default: all of them."},
                "expert": {"type": "integer", "minimum": 1,
                           "description": "For a router: also give this expert's share. Experts are numbered from 1."},
                "start": {"type": "integer", "description": "First step of the range (default: the first recorded)."},
                "end": {"type": "integer", "description": "Last step of the range, inclusive (default: the last recorded)."},
                "max_points": {"type": "integer", "minimum": 1, "maximum": MAX_POINTS,
                               "description": f"Most points to return (default {DEFAULT_POINTS})."},
                "rank": _RANK,
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        "annotations": {"title": "Get a series", **_READ_ONLY},
    },
    {
        "name": "get_all_ranks_status",
        "title": "Get the all-ranks status",
        "description": (
            "For a run folder in a tensorquill-server store, or an all-ranks evidence file: the run state, how many"
            " ranks are judged, unknown, clean and with incidents, each report state with its reason, the incidents"
            " grouped across ranks and those of the run as a whole (a stall, a silent rank), and the server's"
            " progress status. A rank whose report is not current is unknown and is never counted as clean."
            " `kinds` says what was checked for the run, over the judged ranks only (`kinds.judged_ranks`; those"
            " under `kinds.ranks_not_counted` are in no value). A kind under `kinds.checked` was checked on every"
            " judged rank with enough history, and only for those does nothing found mean anything. A kind takes"
            " its worst judged rank's value, so one under `kinds.not_configured` or `kinds.unavailable` may still"
            " be checked on other ranks: `kinds.differs_by_rank` names each such kind's ranks by value (with"
            " `not_enough_history` for those of its checked ranks still warming up, where nothing found is no"
            " verdict either), and an incident of that kind stands for the ranks that show it. `kinds.not_enough_history` gives the fewest"
            " `records` any judged rank has and the ranks still warming up. When a judged rank's report does not"
            " say what was checked, or no rank is judged, there is no `kinds`: `kinds_not_stated` gives the reason,"
            f" and nothing is to be inferred. `per_rank[].kinds` counts each rank's own {len(SERVER_KINDS)} kinds. While ranks are"
            " stopped, `stopped_stacks` groups those that sent a stack by where their main thread was and names an odd"
            " group only when one is clearly apart (`verdict`); it is an observation, never the cause (see its"
            " `caveats`), and the ranks with no stack are listed, not compared. `dropped_rows`, only when the"
            " server dropped rows of the run (never stored) because a key of theirs was null or their schema version"
            " is one it does not support: how many in all, by key (`unsupported_schema_version` for the version) and"
            " by rank (`per_rank[].dropped_rows`, and in the `rank` entry). It is an upper bound: a batch sent again"
            " after a lost acknowledgement is counted again. With no key, none was dropped. `refused_batches`, only"
            " when the server refused whole batches of the run because a key column was missing or of another type"
            " (no row of them stored): how many batches in all, by key column and by rank (`per_rank[].refused_batches`,"
            " and in the `rank` entry). A count of batches, not rows, never added to `dropped_rows`; the same batch"
            " sent again is not counted again. With no key, none was refused."
            + _NOT_NEWER_WORDS.format(where="rows_not_newer", whose="the run (by rank in `ranks`, in"
                                      " `per_rank[].rows_not_newer` and `per_rank[].late_steps_confirmed`, and in the"
                                      " `rank` entry)") +
            " `rank` gives one rank's entry in full. On one rank's capture or evidence file this is an error that says so."
            " Each judged rank's skipped updates and checkpoint tensors, as list_incidents describes them for one rank:"
            + _BY_RANK_WORDS + " " + _SKIPS_WORDS.replace("`skipped_updates`", "A rank's `skipped_updates`", 1)
            + _TENSORS_WORDS.replace(" `checkpoint_tensors`", " A rank's `checkpoint_tensors`", 1) + _KIND_KEYS_WORDS
            + " `per_rank[].capture_stopped` (and the `rank` entry's), only for a rank whose capture stopped, is the"
            " server's status as written: `reason`, and `step`, the last step the session committed before it"
            " stopped. `step` 0 is step 0; `step` null means it committed none; no `step` key means the sender is older"
            " than the key and the step is not known."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"rank": {"type": "integer", "minimum": 0, "description": "Give this rank's entry in full."}},
            "additionalProperties": False,
        },
        "annotations": {"title": "Get the all-ranks status", **_READ_ONLY},
    },
    {
        "name": "list_runs",
        "title": "List runs",
        "description": (
            "The runs kept in the same folder or store as this one, by project: each run's name, run ID, run state,"
            " last step, incidents by class, first incident (kind and step), how many of the kinds were checked, and"
            " start time. The run ID is the identity; a name is a label and may repeat. It reads each run's manifest,"
            " its small evidence or report files and the ends of its .jsonl files (one over a megabyte is scanned"
            " for the attempts its lines name), and runs no detector: a local run whose evidence file was"
            " never written has `run_state` null and `evidence.state` \"missing\", which is not \"clean\"."
            " `evidence.state` \"several_run_ids\" is a run one of whose .jsonl files holds, for a rank, the records"
            " of more than one run ID (`evidence.run_ids`, and `evidence.files` with each file's ranks and IDs): its"
            " rows are not one run's, so the row gives no run state, last step or incident, whatever evidence file"
            " lies beside it, and compare_runs on another of those IDs names the file. A run"
            " with `has_manifest` false was recorded before runs had a project, and is listed under"
            " `uncategorized` with its run ID as its name. `rank_count` is how many ranks the run has, or null when the"
            " listing cannot count them without reading every record (`rank_count_not_known` says why). For a run of"
            " several ranks the row is the whole run's, by the all-ranks rules: the incidents of every judged rank"
            " (`ranks_with_incidents` names them), and a kind counts as checked only when every judged rank checked"
            " it. A rank without a current report or evidence file is unknown (`evidence.unknown_ranks`), never"
            " clean; `kinds_not_stated` says why there is no count. `dropped_rows`, on a store run's row only when"
            " the server dropped rows of it (never stored) because a key of theirs was null or their schema version is"
            " one it does not support: how many in all, by key and by rank; an upper bound, since a batch sent again"
            " after a lost acknowledgement is counted again. No key means none was dropped. `refused_batches`, on a"
            " store run's row only when the server refused whole batches of it because a key column was missing or of"
            " another type: how many batches in all, by key column and by rank, never added to `dropped_rows`. No key"
            " means none was refused."
            + _NOT_NEWER_WORDS.format(where="rows_not_newer", whose="a store run, on its row, by rank in `ranks`") +
            " `freshness`, on a local run's row only when its evidence files do not cover the"
            " newest step its records reach: `current` false, `judged_to_step` (the step the evidence says it judged to,"
            " null when the file does not name one, as a file the command line writes does not), `records_to_step` and"
            " `why`. With `why` `records_newer` the row's state and incidents are the evidence's, as of that judgment, and"
            " say nothing of the steps after it; with `nothing_judged` (the evidence judged no step) or"
            " `thresholds_changed` no earlier result is shown and the state is unknown. No key means the evidence covers"
            " the newest step (a store's row never has it: a rank whose report is behind its stored parts is unknown)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Only this project's runs."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RUNS,
                          "description": f"How many runs to list (default {DEFAULT_RUNS})."},
            },
            "additionalProperties": False,
        },
        "annotations": {"title": "List runs", **_READ_ONLY},
    },
    {
        "name": "compare_runs",
        "title": "Compare runs",
        "description": (
            "Compare two or more of the runs list_runs gives, each named by run ID or by name (a name that fits"
            " several runs is refused with their IDs; a path is never taken). The answer has four parts, to be read in"
            " this order. `setup`: the project, group, world size, versions and config keys that differ. `checked`:"
            " the kinds one run checked and another did not configure, could not judge, or has not had enough"
            " history for. `incidents`: by kind and what the incident names, with the runs that show it (`in`), each"
            " run's `first_step`, and `absent_from`, which says for each other run whether it looked: only"
            " `checked` there means that run looked, with enough history, and found none. `divergence`: for two"
            " paired runs, the first step and the layer or parameter where they part (`state` `diverged`), or"
            " `no_divergence`, or `no_answer` with the reason in `detail` (runs that never agreed early); any other"
            " `state` says why it was not asked, such as a run in a server's store or runs whose ranks differ"
            " (`ranks_differ`). Every rank of each run is read: its incidents come from all of its ranks (an entry's"
            " `ranks` names the ranks that show it), and in `checked` a kind is `checked` for the run only when every"
            " judged rank checked it with enough history (`history_not_stated`: some rank does not say). For two runs"
            " of several ranks, `divergence` compares each rank with the same rank of the other run and gives the"
            " earliest, with its `rank` and every rank's answer in `by_rank`. `rank` reads that one rank of every run"
            " instead. `incidents.other_ranks_not_compared` lists a run's ranks whose incidents are not in the answer:"
            " those not read, or, when every rank is read, those without a current report (not clean). Local files"
            " are judged here. A run in a server's store is read from the server's reports and no part is opened; a"
            " rank whose report is not current is unknown, and when no rank's is, the tool says so and compares"
            " nothing. `runs[].dropped_rows`, only for a run in a server's store whose server dropped rows with a"
            " null key or a schema version it does not support (with `rank`, that rank's): the rows in all, by key"
            " and by rank, the ranks listed up to"
            f" {PER_RANK_ROWS} (`ranks_omitted` counts the rest). It is an upper bound: a batch sent again after a"
            " lost acknowledgement is counted again. A run that dropped none has no such key."
            " `runs[].refused_batches`, only for such a run whose server refused whole batches because a key column"
            " was missing or of another type: the batches in all, by key column and by rank, bounded the same way;"
            " a count of batches, never added to `dropped_rows`. A run that refused none has no such key."
            + _NOT_NEWER_WORDS.format(where="runs[].rows_not_newer", whose="such a run (with `rank`, that rank's)")
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "runs": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": MAX_COMPARED,
                         "description": "Run IDs or names, as list_runs gives them. The first is A, the second B."},
                "project": {"type": "string", "description": "Look names up in this project only."},
                "rank": {"type": "integer", "minimum": 0, "description": "Read this one rank of every run (default: every rank)."},
            },
            "required": ["runs"],
            "additionalProperties": False,
        },
        "annotations": {"title": "Compare runs", **_READ_ONLY},
    },
)
_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "list_incidents": list_incidents, "get_incident": get_incident, "get_series": get_series,
    "get_all_ranks_status": get_all_ranks_status, "list_runs": list_runs, "compare_runs": compare_runs,
}


def _integer(arguments: Mapping[str, Any], name: str, *, minimum: int | None = None, maximum: int | None = None) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"`{name}` must be a whole number, not {value!r}.")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        bounds = f"between {minimum} and {maximum}" if minimum is not None and maximum is not None else f"{minimum} or more"
        raise ToolError(f"`{name}` must be {bounds}, not {value}.")
    return value


# --- Replies ------------------------------------------------------------------------------


def _strict(value: Any) -> Any:
    """`value` as strict JSON: a NaN or Inf becomes the string the capture writes for it."""
    if isinstance(value, float):
        return value if math.isfinite(value) else "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return value if value is None or isinstance(value, (str, int, bool)) else str(value)


def _line(message: Any) -> bytes:
    """One reply as a line of ASCII: `json.dumps` escapes every other character."""
    return json.dumps(message, separators=(",", ":"), allow_nan=False).encode("ascii")


def _tool_reply(request_id: Any, text: str, is_error: bool) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": {"content": [{"type": "text", "text": text}], "isError": is_error}}


def _text(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def _largest(value: Any, path: str, found: list[tuple[int, str, Any, Any]], parent: Any = None, key: Any = None) -> int:
    """The length of `value` as JSON, noting in `found` every part that can be cut: a list of
    more than four items, a string of more than 256 characters, and a map of more than sixteen
    entries unless one entry is most of it (then that entry is what to cut)."""
    if isinstance(value, dict):
        sizes = [
            len(json.dumps(k)) + 2 + _largest(v, f"{path}.{k}" if path else str(k), found, value, k)
            for k, v in value.items()
        ]
        size = 2 + sum(sizes)
        if parent is not None and len(value) > 16 and max(sizes) <= size / 2:
            found.append((size, path, parent, key))
        return size
    if isinstance(value, list):
        size = 2 + sum(1 + _largest(v, f"{path}[{i}]", found, value, i) for i, v in enumerate(value))
        if len(value) > 4:
            found.append((size, path, parent, key))
        return size
    size = len(json.dumps(value))
    if isinstance(value, str) and len(value) > 256:
        found.append((size, path, parent, key))
    return size


def fit(payload: dict[str, Any], fits: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    """`payload`, cut until `fits` accepts it: the largest list, map or string loses its last
    quarter, again and again, and `truncated` says what is shown of each. The original is not
    changed. A `ToolError` when nothing is left to cut."""
    if fits(payload):
        return payload
    payload = copy.deepcopy(payload)
    cut: dict[str, tuple[int, int, str]] = {}
    for _ in range(400):
        notes = [f"{path}: {shown} of {total} {unit} shown" for path, (shown, total, unit) in cut.items()]
        trial = {**payload, "truncated": notes}
        if fits(trial):
            return trial
        found: list[tuple[int, str, Any, Any]] = []
        _largest(payload, "", found)
        if not found:
            break
        _, path, parent, key = max(found, key=lambda item: item[0])
        value = parent[key]
        if isinstance(value, str):
            keep, unit = max(len(value) * 3 // 4, 128), "characters"
            parent[key] = value[:keep] + " …"
        elif isinstance(value, dict):
            keep, unit = max(len(value) * 3 // 4, 8), "entries"
            parent[key] = dict(list(value.items())[:keep])
        else:
            keep, unit = max(len(value) * 3 // 4, 4), "items"
            parent[key] = value[:keep]
        cut[path] = (keep, cut[path][1] if path in cut else len(value), unit)
    raise ToolError(
        f"the answer is longer than {MAX_ANSWER_BYTES} bytes and could not be shortened further"
        + (f" (already cut: {'; '.join(notes)[:1000]})" if notes else "") + "."
    )


def call_tool(source: Source, request_id: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """Answer `tools/call`. A tool that cannot answer replies with `isError` and the reason; an
    unknown tool is a protocol error, as the specification has it."""
    name = params.get("name")
    if not isinstance(name, str) or name not in _HANDLERS:
        return _error(request_id, INVALID_PARAMS, f"Unknown tool: {name!r}. The tools are: {', '.join(_HANDLERS)}.")
    arguments = params.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _error(request_id, INVALID_PARAMS, "`arguments` must be an object.")

    def fits(payload: dict[str, Any]) -> bool:
        return len(_line(_tool_reply(request_id, _text(_strict(payload)), False))) + 1 <= MAX_ANSWER_BYTES

    try:
        allowed = next(tool for tool in TOOLS if tool["name"] == name)["inputSchema"]["properties"]
        unknown = [key for key in arguments if key not in allowed]
        if unknown:
            raise ToolError(f"{name} takes no argument named {', '.join(map(repr, unknown))}. It takes: {', '.join(allowed) or 'none'}.")
        handler = _HANDLERS[name]
        answer = handler(source, arguments, fits) if handler is get_series else handler(source, arguments)
        return _tool_reply(request_id, _text(fit(_strict(answer), fits)), False)
    except ToolError as error:
        return _tool_reply(request_id, str(error)[:4000], True)
    except (Exception, SystemExit) as error:  # a request must never take the server down
        traceback.print_exc(file=sys.stderr)
        text = f"{name} failed inside the server ({type(error).__name__}: {str(error)[:1000]}). The server is still running."
        return _tool_reply(request_id, text, True)


def _is_request_id(value: Any) -> bool:
    """Whether `value` is an id a reply carries: a string, or a finite number that is not a bool.
    JSON-RPC 2.0 allows a number with a fraction (it SHOULD NOT have one) and allows null, which
    MCP forbids and this server refuses; NaN and Infinity, which Python's JSON reader takes, are no
    JSON, so no reply could carry them. The one rule of `handle`'s check and of every error reply
    that echoes an id (wave AP, AO-05: those took a bool and dropped a fraction)."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)  # never `isfinite` of an int: 10**400 raises


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(source: Source, message: Any) -> dict[str, Any] | None:
    """The reply to one JSON-RPC message, or None for a notification or a client's own reply."""
    if not isinstance(message, dict):
        return _error(None, INVALID_REQUEST, "A message must be a JSON object.")
    request_id = message.get("id")
    has_id = "id" in message
    method = message.get("method")
    if not isinstance(method, str):
        if has_id and ("result" in message or "error" in message):
            return None  # a reply to a request; this server sends none
        return _error(request_id if _is_request_id(request_id) else None, INVALID_REQUEST, "A request needs a `method`.")
    if not has_id:
        return None  # a notification, `notifications/initialized` among them: never answered
    if not _is_request_id(request_id):
        return _error(None, INVALID_REQUEST, "A request's `id` must be a string or a finite number.")
    if message.get("jsonrpc") != "2.0":
        return _error(request_id, INVALID_REQUEST, 'A request needs "jsonrpc": "2.0".')
    params = message.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return _error(request_id, INVALID_PARAMS, "`params` must be an object.")
    if method == "initialize":
        asked = params.get("protocolVersion")
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": asked if asked in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "title": "TensorQuill evidence (read-only)", "version": __version__},
            "instructions": INSTRUCTIONS,
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": list(TOOLS)}}
    if method == "tools/call":
        return call_tool(source, request_id, params)
    return _error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}. This server has initialize, ping, tools/list and tools/call.")


def _fits_line(line: bytes) -> bool:
    return len(line) + 1 <= MAX_ANSWER_BYTES  # the newline is part of the reply line


def _cut_to_fit(text: str, build: Callable[[str], dict[str, Any]]) -> bytes | None:
    """The line of `build(text)` with `text` cut, and marked cut, until the line fits; None when
    not even an empty text fits. Measured on the line as written, where a character outside
    ASCII is six or twelve bytes."""
    if _fits_line(line := _line(build(text))):
        return line
    low, high = 0, len(text)  # `low` characters and the mark fit, `high` do not (or are the whole)
    if not _fits_line(_line(build(" …"))):
        return None
    while high - low > 1:
        middle = (low + high) // 2
        if _fits_line(_line(build(text[:middle] + " …"))):
            low = middle
        else:
            high = middle
    return _line(build(text[:low] + " …"))


def _reply_line(reply: Any) -> bytes:
    """One reply as its line, within `MAX_ANSWER_BYTES` whatever it holds (BUG-25, wave AO). A tool's
    answer is already fitted (`fit`); what can still be too long is an error's or a tool error's
    text, which is cut, and the request's id, which every reply carries: an answer that cannot fit
    with its id says so to that id, and one that cannot fit any reply with it is said with no id,
    as JSON-RPC answers a request whose id it could not read."""
    if isinstance(reply, list):  # a batch: the cap is per answer
        return b"[" + b",".join(_reply_line(item) for item in reply) + b"]"
    try:
        line = _line(reply)
    except ValueError:  # a value strict JSON has no form for, which `_strict` did not see
        traceback.print_exc(file=sys.stderr)
        kept = reply.get("id") if isinstance(reply, dict) else None
        reply = _error(kept if _is_request_id(kept) else None, -32603,
                       "Internal error (the reply is not strict JSON). The server is still running.")
        line = _line(reply)
    if _fits_line(line):
        return line
    request_id = reply.get("id")
    cut: bytes | None = None
    if isinstance(reply.get("error"), dict) and isinstance(reply["error"].get("message"), str):
        error = reply["error"]
        cut = _cut_to_fit(error["message"], lambda text: {**reply, "error": {**error, "message": text}})
    elif isinstance(reply.get("result"), dict) and reply["result"].get("isError") is True:
        text = reply["result"]["content"][0]["text"]
        cut = _cut_to_fit(text, lambda shown: _tool_reply(request_id, shown, True))
    if cut is not None:
        return cut
    size = len(_line(request_id))
    message = (f"The answer to this request is longer than {MAX_ANSWER_BYTES} bytes with its id, which is {size}"
               " bytes as JSON. Send the request again with a shorter id.")
    line = _line(_error(request_id, INVALID_REQUEST, message))
    return line if _fits_line(line) else _line(_error(None, INVALID_REQUEST, message))


def handle_line(source: Source, line: bytes) -> bytes | None:
    """The reply line to one line of input, without its newline, or None when there is none. It is
    at most `MAX_ANSWER_BYTES` long with its newline (`_reply_line`); a batch's, an answer each."""
    try:
        message = json.loads(line.decode("utf-8"))
    except ValueError as error:
        return _reply_line(_error(None, PARSE_ERROR, f"Parse error: the line is not JSON ({error})."))
    try:
        if isinstance(message, list):  # a batch, which revision 2025-03-26 lets a client send (the cap is per answer)
            if not message:
                return _reply_line(_error(None, INVALID_REQUEST, "An empty batch."))
            replies = [reply for reply in (handle(source, item) for item in message) if reply is not None]
            return _reply_line(replies) if replies else None
        reply = handle(source, message)
    except (Exception, SystemExit) as error:
        traceback.print_exc(file=sys.stderr)
        request_id = message.get("id") if isinstance(message, dict) and _is_request_id(message.get("id")) else None
        reply = _error(request_id, -32603, f"Internal error ({type(error).__name__}). The server is still running.")
    return _reply_line(reply) if reply is not None else None


def serve(source: Source, reader: Any, writer: Any) -> None:
    """Answer requests from `reader`, a line each, on `writer`, until `reader` ends."""
    while True:
        line = reader.readline()
        if not line:
            return
        if not line.strip():
            continue
        reply = handle_line(source, line)
        if reply is not None:
            writer.write(reply + b"\n")
            writer.flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        # The installed command (`tensorquill-mcp`) names itself; the module run with -m is named here.
        prog=None if Path(sys.argv[0]).name.startswith("tensorquill-mcp") else "python -m tensorquill.mcp_server",
        description="A read-only MCP server (stdio) over one run's TensorQuill evidence, and the runs kept beside it: "
        "six tools, no write action.",
    )
    parser.add_argument(
        "path", type=Path,
        help="an evidence.json, a .jsonl capture from tensorquill.init, or a run folder in a tensorquill-server "
        "store (runs/<run_id>). It is read again at every call and need not exist yet",
    )
    parser.add_argument(
        "--records", type=Path,
        help="for an evidence.json: the .jsonl capture or run folder its step records are in, for get_series "
        "(default: x.jsonl beside x.evidence.json or x.evidence.compact.json, or an all-ranks view's own run folder)",
    )
    parser.add_argument(
        "--runs", type=Path,
        help="the folder of local files or the store that list_runs and compare_runs read (default: the folder "
        "PATH is in, or the runs/ folder of the store for a run folder)",
    )
    args = parser.parse_args(argv)
    # Only protocol messages may reach the client: anything a library prints goes to stderr.
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    try:
        writer = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass
    sys.stdout = sys.stderr
    try:
        serve(Source(args.path, args.records, args.runs), reader, writer)
    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
