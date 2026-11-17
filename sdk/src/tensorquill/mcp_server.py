"""A read-only MCP server over TensorQuill's evidence files, for a team's coding agent.

    python -m tensorquill.mcp_server PATH [--records CAPTURE] [--runs FOLDER]

`PATH` is an `evidence.json`, a `.jsonl` capture from `tensorquill.init`, or a run folder in a
`tensorquill-server` store (`runs/<run_id>`). The server speaks the Model Context Protocol
over stdio and offers six tools:

    list_incidents         what the detectors found, with the run state and what was not checked
    get_incident           one incident in full
    get_series             one layer's, parameter's, router's or logged scalar's values over a step range
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
from . import compare as _compare
from . import runs as _runs
from .evidence import (
    ALL_RANKS_VIEW,
    _discover_rank_folders,
    all_ranks_evidence,
    all_ranks_kinds,
    build_evidence,
    incident_class,
    incident_summary,
    kinds_by_value,
    load_records,
    read_server_report,
    read_watchdog,
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

SERIES_KINDS = ("layer", "parameter", "router", "scalar")
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
    " and get_series for the layer, parameter or router it names. For a run folder with several ranks,"
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


def _rank_number(name: str) -> int | None:
    key, _, value = name.partition("=")
    return int(value) if key == "rank" and value.isdecimal() else None


def _rank_of(record: Mapping[str, Any]) -> int:
    rank = record.get("rank", 0)
    return rank if isinstance(rank, int) and not isinstance(rank, bool) else 0


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
    for line in data[: data.rfind(b"\n") + 1].splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if isinstance(record, dict) and "step" in record:
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
            own = _rank_number(path.resolve().name)
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
                folder = path.parent.parent if _rank_number(path.name) is not None else path.parent
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
        """Every record of a `.jsonl` capture, all ranks and attempts, and the lines left out."""

        def read() -> tuple[list[dict[str, Any]], int]:
            try:
                try:
                    return load_records(path, all_attempts=True), 0
                except ValueError:  # a job that is still running may be in the middle of a line
                    records, skipped = _complete_lines(path)
                    return load_records(records, all_attempts=True), skipped
            except OSError as error:
                raise ToolError(f"{path} cannot be read ({error}).") from None
            except (TypeError, KeyError) as error:
                raise ToolError(f"{path} does not hold TensorQuill step records ({error!r}).") from None

        return self._keep(("capture", path), _stamp(path), read)

    def capture_evidence(self, path: Path, rank: int) -> dict[str, Any]:
        records, _ = self.capture(path)

        def build() -> dict[str, Any]:
            try:
                return build_evidence([r for r in records if _rank_of(r) == rank], run_manifest=path)
            except (KeyError, TypeError, ValueError) as error:
                raise ToolError(f"the detectors could not read the records in {path} ({error!r}).") from None

        return self._keep(("evidence", path, rank), _stamp(path), build)

    # One rank's step records, for series ------------------------------------------------

    def rank_records(self, rank: int | None) -> tuple[list[dict[str, Any]], dict[str, dict[str, None]], dict[str, Any]] | None:
        """One rank's step records in step order, the names they carry by kind, and where they
        came from; None when the path is an evidence file with no records beside it."""
        kind, where, own_rank = self.locate()
        default_rank = own_rank
        if kind == "evidence":
            document = self.document(where)
            where = self._records_beside(where, document)
            if where is None:
                return None
            kind = "store" if where.is_dir() else "capture"
            if kind == "store" and _rank_number(where.resolve().name) is not None:
                default_rank, where = _rank_number(where.resolve().name), where.resolve().parent
            detection = document.get("detection")
            if default_rank is None and isinstance(detection, dict) and isinstance(detection.get("rank"), int):
                default_rank = detection["rank"]
        if kind == "capture":
            every, _ = self.capture(where)
            ranks = sorted({_rank_of(r) for r in every})
            picked = _pick_rank(rank, default_rank, ranks, f"the capture {where}")
            stamp: Any = _stamp(where)

            def read() -> list[dict[str, Any]]:
                return load_records(every, rank=picked)
        else:
            ranks = sorted(_discover_rank_folders(where))
            if not ranks:
                raise ToolError(f"{where} holds no rank=<n> folder yet, so there are no step records to read.")
            picked = _pick_rank(rank, default_rank, ranks, f"the run folder {where}")
            stamp = read_server_report(where / f"rank={picked}")["stored"]

            def read() -> list[dict[str, Any]]:
                try:
                    return load_records(where, rank=picked)
                except ImportError:
                    raise ToolError(
                        "reading a store's step records needs pyarrow (the SDK's `flight` extra), which is not"
                        " installed here. The incidents and the all-ranks status do not need it."
                    ) from None
                except Exception as error:  # pyarrow's own errors are not built-in ones
                    raise ToolError(f"rank {picked}'s stored parts under {where} could not be read ({error}).") from None

        def index() -> tuple[list[dict[str, Any]], dict[str, dict[str, None]]]:
            try:
                records = sorted(read(), key=lambda r: int(r["step"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ToolError(f"the records under {where} are not TensorQuill step records ({error!r}).") from None
            return records, _names(records)

        records, names = self._keep(("records", where, picked), stamp, index)
        origin = {"records": str(where), "rank": picked}
        if len(ranks) > 1:
            origin["ranks_here"] = _ranges(ranks)
        return records, names, origin

    def _records_beside(self, path: Path, document: Mapping[str, Any]) -> Path | None:
        """Where an evidence file's step records are: `--records`, the capture the command line
        writes it next to (`x.jsonl` for `x.evidence.json`), or an all-ranks view's run folder."""
        if self.records_path is not None:
            if not self.records_path.exists():
                raise ToolError(f"{self.records_path}, given as --records, does not exist.")
            return self.records_path
        if path.name.endswith(".evidence.json"):
            capture = path.with_name(path.name[: -len(".evidence.json")] + ".jsonl")
            if capture.is_file():
                return capture
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
            evidence = self.capture_evidence(where, picked)
            source["rank"] = picked
            steps = [r["step"] for r in records if _rank_of(r) == picked and isinstance(r.get("step"), int)]
            if steps:
                source["steps"] = {"first": min(steps), "last": max(steps), "records": len(steps)}
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
                detection = evidence.get("detection")
                own = detection.get("rank") if isinstance(detection, dict) else None
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


def _brief(entry: Mapping[str, Any]) -> dict[str, Any]:
    incident = entry["incident"]
    out: dict[str, Any] = {
        "id": entry["id"], "kind": incident.get("kind"), "class": _class_of(incident), "step": incident.get("step"),
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
    for entry in evidence.get("ranks") or ():
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
    summary = _summary(evidence)
    if summary:
        answer["summary"] = summary
    return answer


def _sample_context(evidence: Mapping[str, Any], incident: Mapping[str, Any]) -> dict[str, Any] | None:
    """The MoE sample an incident points at: the largest shares, and the sample before it."""
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
        out["previous_sample"] = {
            "step": previous["step"], "diversity": previous.get("diversity"),
            "largest_share": _strict_number(max(previous.get("counts") or [0]) / previous["routed_tokens"]),
        }
    return out


def _look_next(incident: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The `get_series` calls that show the data behind an incident."""
    calls: list[dict[str, Any]] = []

    def add(**arguments: Any) -> None:
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
        raise ToolError("`id` is required: the id of an incident, as list_incidents gives it (for example grad_spike@310).")
    rank = _integer(arguments, "rank", minimum=0)
    opened = source.open(rank)
    entries = _entries(opened)
    entry = next((e for e in entries if e["id"] == wanted), None)
    if entry is None:
        known = ", ".join(e["id"] for e in entries[:20]) or "none"
        more = f" and {len(entries) - 20} more" if len(entries) > 20 else ""
        raise ToolError(
            f"no incident has the id {wanted!r}. The incidents here are: {known}{more}."
            " Call list_incidents for the current list."
        )
    incident = entry["incident"]
    found = _class_of(incident)
    answer: dict[str, Any] = {
        "source": opened["source"], "id": entry["id"], "kind": incident.get("kind"), "class": found,
        "class_meaning": CLASS_MEANINGS.get(found, CLASS_MEANINGS["unclassified"]), "step": incident.get("step"),
        "where": _where(incident),
    }
    group = entry.get("group")
    if group is not None:
        ranks = group.get("ranks") or []
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
    if incident.get("kind") == "moe_imbalance":
        sample = _sample_context(opened["evidence"], incident)
        if sample is not None:
            answer["sample"] = sample
    calls = _look_next(incident)
    if calls:
        answer["look_next"] = calls
    return answer


# --- Series -------------------------------------------------------------------------------


def _names(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, None]]:
    """The names step records carry, by kind, in the order they first appear."""
    found: dict[str, dict[str, None]] = {kind: {} for kind in SERIES_KINDS}
    fields = (("layers", "layer"), ("grad_norms", "parameter"), ("weight_norms", "parameter"), ("moe", "router"), ("scalars", "scalar"))
    for record in records:
        for field, kind in fields:
            value = record.get(field)
            if isinstance(value, dict):
                found[kind].update(dict.fromkeys(value))
        for field in RUN_LEVEL:
            if record.get(field) is not None:
                found["scalar"].setdefault(field)
    return found


def _sample_names(evidence: Mapping[str, Any]) -> dict[str, dict[str, None]]:
    found: dict[str, dict[str, None]] = {kind: {} for kind in SERIES_KINDS}
    for sample in evidence.get("samples") or ():
        if isinstance(sample, dict) and isinstance(sample.get("router"), str):
            found["router"].setdefault(sample["router"])
    return found


def _resolve(name: str, kind: str | None, names: Mapping[str, Mapping[str, None]], what: str) -> str:
    """The kind of series `name` is, or a `ToolError` that says what the names here are."""
    holders = [k for k in SERIES_KINDS if name in names[k]]
    if kind is not None and kind in holders:
        return kind
    if kind is None and len(holders) == 1:
        return holders[0]
    if kind is None and holders:
        raise ToolError(f"{name!r} is the name of a {' and of a '.join(holders)} here: pass kind= to choose.")
    if holders:
        raise ToolError(f"{name!r} is a {holders[0]} here, not a {kind}: pass kind={holders[0]!r}, or leave kind out.")
    pool = {n: k for k in ([kind] if kind else SERIES_KINDS) for n in names[k]}
    close = difflib.get_close_matches(name, list(pool), n=5, cutoff=0.5)
    counts = ", ".join(f"{len(names[k])} {k}{'s' if len(names[k]) != 1 else ''}" for k in SERIES_KINDS)
    text = f"no {kind or 'layer, parameter, router or scalar'} is named {name!r} in {what}, which has {counts}."
    if close:
        text += " Closest names: " + ", ".join(f"{n} ({pool[n]})" for n in close) + "."
    else:
        first = [f"{n} ({k})" for k in ([kind] if kind else SERIES_KINDS) for n in list(names[k])[:4]]
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
    if not isinstance(counts, list) or not counts:
        return None
    tokens = sum(counts)
    if tokens <= 0:
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
            stats = (record.get("layers") or {}).get(name)
            point = _columns(stats) if isinstance(stats, Mapping) else None
        elif kind == "parameter":
            point = {
                column: (record.get(field) or {})[name]
                for field, column in (("grad_norms", "grad_norm"), ("weight_norms", "weight_norm"))
                if isinstance(record.get(field), Mapping) and name in record[field]
            } or None
        elif kind == "router":
            stats = (record.get("moe_stats") or {}).get(name)
            diversity = stats.get("diversity") if isinstance(stats, Mapping) else None
            point = _router_point((record.get("moe") or {}).get(name), diversity, expert)
        else:
            logged = record.get("scalars")
            value = logged[name] if isinstance(logged, Mapping) and name in logged else record.get(name) if name in RUN_LEVEL else None
            point = _columns(value) if value is not None else None
        if point:
            rows.append((step, point))
    return rows


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


def _finite(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _over_range(rows: list[tuple[int, dict[str, Any]]], column: str) -> dict[str, Any]:
    """What a column did over every recorded step in the range, sampled or not."""
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
    bad = [step for step, value in held if _finite(value) is None]
    if bad:
        out.update({"not_finite": len(bad), "first_not_finite_step": bad[0]})
    return out


def get_series(source: Source, arguments: Mapping[str, Any], fits: Callable[[dict[str, Any]], bool] = lambda _: True) -> dict[str, Any]:
    name = arguments.get("name")
    if not isinstance(name, str) or not name:
        raise ToolError("`name` is required: a layer (module), parameter, router or logged scalar, as an incident's `where` names it.")
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
    if held is None:
        opened = source.open(rank)
        if opened["shape"] != "rank":
            raise ToolError(
                "this all-ranks evidence file has no step records, and its run folder is not on this machine:"
                " start the server on the run folder (runs/<run_id>), or with --records, to read series."
            )
        evidence, names, origin = opened["evidence"], _sample_names(opened["evidence"]), {"records": None}
        what = (
            f"the evidence file {source.path}, which has only its routers' samples (start the server on the"
            " .jsonl capture or the run folder, or with --records, for layers and parameters)"
        )
    else:
        records, names, origin = held
        what = f"rank {origin['rank']}'s step records"
        if (kind or "router") == "router" and name in names["router"]:
            path_kind, path, _ = source.locate()
            if path_kind == "capture":
                evidence = source.capture_evidence(path, origin["rank"])
            elif path_kind == "evidence" and source.document(path).get("view") != ALL_RANKS_VIEW:
                evidence = source.document(path)
    kind = _resolve(name, kind, names, what)
    if expert is not None and kind != "router":
        raise ToolError(f"`expert` is for a router's series; {name!r} is a {kind}.")
    if evidence is not None and any(isinstance(s, dict) and s.get("router") == name for s in evidence.get("samples") or ()):
        rows = _sample_rows(name, evidence, expert)
    else:
        rows = _rows(kind, name, records if held is not None else (), expert)
    if not rows:
        raise ToolError(f"no step carries values for the {kind} {name!r} yet.")
    columns = list(dict.fromkeys(column for _, point in rows for column in point))
    if expert is not None and all(point.get("expert_share") is None for _, point in rows):
        raise ToolError(f"the router {name!r} has no expert {expert}: its experts are numbered from 1.")
    if stat is not None:
        if stat not in columns:
            raise ToolError(f"the {kind} {name!r} has no `{stat}`. It has: {', '.join(columns)}.")
        columns = [stat]
    else:  # asked for by name only: the token count is the same at every step of most runs
        columns = [column for column in columns if column != "routed_tokens"]
    first, last = rows[0][0], rows[-1][0]
    inside = [row for row in rows if (start is None or row[0] >= start) and (end is None or row[0] <= end)]
    if not inside:
        raise ToolError(
            f"the {kind} {name!r} has no recorded step between {start if start is not None else first} and"
            f" {end if end is not None else last}: it is recorded at {len(rows)} steps, from {first} to {last}."
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
            flagged = [step for step, point in inside if point.get("overloaded") or point.get("dead")]
            if flagged:
                answer["first_flagged_step"] = flagged[0]
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
    if rank is not None:
        answer["rank"] = _rank_entry(evidence, rank)
    elif len(rows) <= PER_RANK_ROWS:
        answer["per_rank"] = [
            {"rank": number, "report": (entry.get("detection") or {}).get("state"),
             "judged": (entry.get("detection") or {}).get("judged"),
             "last_step": (entry.get("detection") or {}).get("report_last_step"),
             **({"incidents": [i.get("kind") for i in entry["incidents"] if isinstance(i, dict)]}
                if isinstance(entry.get("incidents"), list) else {}),
             # Of the 25 kinds this rank's own report judges: where a rank differs from the others.
             **({"kinds": {key: entry["kind_counts"].get(key) for key in (
                 "checked", "not_enough_history", "not_configured", "unavailable")}}
                if isinstance(entry.get("kind_counts"), dict) else {}),
             **({"stream": entry["stream"]} if isinstance(entry.get("stream"), dict) else {})}
            for number, entry in sorted(rows.items(), key=lambda item: item[0] if isinstance(item[0], int) else -1)
        ]
    else:
        answer["per_rank_note"] = f"{len(rows)} ranks: pass rank= for one rank's entry"
    summary = _summary(evidence)
    if summary:
        answer["summary"] = summary
    return answer


# --- Other runs -----------------------------------------------------------------------------

_ROW_KEYS = ("project", "name", "run_id", "group", "tags", "has_manifest", "source", "started", "run_state", "last_step",
             "incidents", "first_incident", "kinds_checked", "ranks", "evidence")


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
        "runs": [{key: row[key] for key in _ROW_KEYS if key in row} for row in shown],
    }


def compare_runs(source: Source, arguments: Mapping[str, Any]) -> dict[str, Any]:
    folder = source.runs_folder()
    wanted = arguments.get("runs")
    if not isinstance(wanted, list) or not 2 <= len(wanted) <= MAX_COMPARED or not all(isinstance(w, str) and w for w in wanted):
        raise ToolError(f"`runs` must be a list of 2 to {MAX_COMPARED} run IDs or names, as list_runs gives them.")
    project = _text_argument(arguments, "project")
    rank = _integer(arguments, "rank", minimum=0)
    try:
        return _compare.compare_runs(*wanted, source=folder, project=project, rank=rank, paths=False,
                                     read_parts=False)
    except (ValueError, OSError, RuntimeError) as error:
        # A run that is not there, a name that fits several runs (the message lists their IDs),
        # or records that cannot be read.
        raise ToolError(f"{error} (the runs are looked up at {folder}; list_runs shows them)") from None


# --- The tools, as the protocol lists them --------------------------------------------------

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
            " then the whole run's, read as get_all_ranks_status describes."
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
            "One incident in full, by the id list_incidents gave (for example grad_spike@310): every number the"
            " detector recorded, what its class means, the ranks that show it, for an MoE incident the sample it"
            " points at (largest expert shares, and the sample before), and the get_series calls that show the data"
            " behind it."
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
            "The recorded values of one layer (module: mean, std, abs_max, zero_frac), parameter (grad_norm,"
            " weight_norm), MoE router (diversity, largest share and expert, experts with no tokens; `expert` adds"
            " that expert's share) or logged scalar (such as the loss, or grad_norm and lr for the whole model) over"
            f" a step range. At most `max_points` points come back (default {DEFAULT_POINTS}): a longer range is"
            " thinned to every nth recorded step, with the stride reported, and `over_range` gives the minimum,"
            " maximum (with their steps) and median over every recorded step in the range, so a one-step spike is"
            " not lost. Narrow `start` and `end` to see each step. Numbers have six significant digits; a NaN or"
            ' Inf is the string "nan", "inf" or "-inf".'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "The layer (module), parameter, router or scalar name, as an incident's `where` gives it."},
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
            " and nothing is to be inferred. `per_rank[].kinds` counts each rank's own 25 kinds. `rank`"
            " gives one rank's entry in full. On one rank's capture or evidence file this is an error that says so."
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
            " start time. The run ID is the identity; a name is a label and may repeat. It reads each run's manifest"
            " and its small evidence or report files only, and runs no detector: a local run whose evidence file was"
            " never written has `run_state` null and `evidence.state` \"missing\", which is not \"clean\". A run"
            " with `has_manifest` false was recorded before runs had a project, and is listed under"
            " `uncategorized` with its run ID as its name."
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
            " `state` says why it was not asked, such as a run in a server's store. One rank of each run is read (`rank`, default the"
            " lowest); `incidents.other_ranks_not_compared` lists a run's ranks that were not read. Local files are"
            " judged here. A run in a server's store is read from the server's report and no part is opened; when"
            " that report is not current the tool says so and compares nothing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "runs": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": MAX_COMPARED,
                         "description": "Run IDs or names, as list_runs gives them. The first is A, the second B."},
                "project": {"type": "string", "description": "Look names up in this project only."},
                "rank": {"type": "integer", "minimum": 0, "description": "The rank read of every run (default: each run's lowest)."},
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
        return _error(request_id if isinstance(request_id, (str, int)) else None, INVALID_REQUEST, "A request needs a `method`.")
    if not has_id:
        return None  # a notification, `notifications/initialized` among them: never answered
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int, float)):
        return _error(None, INVALID_REQUEST, "A request's `id` must be a string or a number.")
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


def handle_line(source: Source, line: bytes) -> bytes | None:
    """The reply line to one line of input, without its newline, or None when there is none."""
    try:
        message = json.loads(line.decode("utf-8"))
    except ValueError as error:
        return _line(_error(None, PARSE_ERROR, f"Parse error: the line is not JSON ({error})."))
    try:
        if isinstance(message, list):  # a batch, which revision 2025-03-26 lets a client send (the cap is per answer)
            if not message:
                return _line(_error(None, INVALID_REQUEST, "An empty batch."))
            replies = [reply for reply in (handle(source, item) for item in message) if reply is not None]
            return _line(replies) if replies else None
        reply = handle(source, message)
    except (Exception, SystemExit) as error:
        traceback.print_exc(file=sys.stderr)
        request_id = message.get("id") if isinstance(message, dict) and isinstance(message.get("id"), (str, int)) else None
        reply = _error(request_id, -32603, f"Internal error ({type(error).__name__}). The server is still running.")
    return _line(reply) if reply is not None else None


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
        prog="python -m tensorquill.mcp_server",
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
        "(default: x.jsonl beside x.evidence.json, or an all-ranks view's own run folder)",
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
