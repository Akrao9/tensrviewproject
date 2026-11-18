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
  evidence file `<file>.evidence.json`;
- a server's store: `runs/<run_id>/` with `run.json`, `watchdog.json` and a `rank=<n>` folder
  for each rank. `<folder or store>` is the folder that holds `runs/`, or `runs/` itself.

The listing reads the manifest and the small evidence or report files only. It never opens a
Parquet part and never runs a detector, so a local run whose evidence file was never written
is listed with its state unknown and a note saying how to write it. A local run of several
ranks reads each rank's evidence file. Of a `.jsonl` file it
reads the first line and the last one (for the run ID, the start time, the last step and the
ranks they name).

How many ranks: a file named `<run_id>.rank<N>.jsonl` holds rank N alone. A file with a name of
its own may hold several ranks, and only reading it whole would name them all, so its ranks are
counted from its first and last records and the manifest's `world_size`: when the world size
equals the ranks seen, that is the count; otherwise `rank_count` is None (printed as the ranks
seen and "+") and `rank_count_not_known` says why. The evidence file beside a file of several
ranks (its ends name two, or the manifest's `world_size` is above 1) is one rank's and does not
say which, so such a row is `ranks_unknown`: its incidents are
listed, and what was checked is not counted for the run. `compare` reads every rank of it.

A run without a manifest (every run recorded before the manifest existed) is listed under
`uncategorized` with its run ID as its name.

`compare` (`compare.py`) finds runs by ID or name through `find_run` here.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _last_record(path: Path) -> dict[str, Any] | None:
    """The last whole line of a `.jsonl` file, read from its last megabyte only."""
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            file.seek(max(0, size - _TAIL_BYTES))
            tail = file.read()
    except OSError:
        return None
    lines = tail.split(b"\n")
    if size > _TAIL_BYTES:
        lines = lines[1:]  # the first piece starts mid-line
    for line in reversed(lines[:-1]):  # what follows the last newline is not finished
        if not line.strip():
            continue
        try:
            found = json.loads(line)
        except ValueError:
            return None
        return found if isinstance(found, dict) else None
    return None


def _rank_of_file(path: Path, first: Mapping[str, Any] | None) -> int:
    named = _named_rank(path)
    if named is not None:
        return named
    value = (first or {}).get("rank")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _is_run_folder(folder: Path) -> bool:
    try:
        return (folder / _manifest.STORE_MANIFEST).is_file() or (folder / "watchdog.json").is_file() or any(
            entry.is_dir() and entry.name.startswith("rank=") for entry in os.scandir(folder)
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
        return by_id.setdefault(run_id, {"run_id": run_id, "source": "local", "path": None, "files": {},
                                         "manifest": None})

    for path in sorted(folder.glob("*.jsonl")) if files is None else files:
        if not path.is_file():
            continue
        first = None
        run_id = _manifest.run_id_of_file(path)
        if run_id is None:  # a file with a name of its own: its records say whose it is
            first = _first_record(path)
            run_id = str(first["run_id"]) if first and first.get("run_id") else path.name[: -len(".jsonl")]
        found = entry(run_id)
        found["files"].setdefault(_rank_of_file(path, first), path)
    if files is None:
        for path in sorted(folder.glob(f"*{_manifest.MANIFEST_SUFFIX}")):
            # A manifest whose rank 0 has not written a record yet is still a run that started.
            entry(path.name[: -len(_manifest.MANIFEST_SUFFIX)])
    for run_id, found in by_id.items():
        ranks = sorted(found["files"])
        found["path"] = found["files"][ranks[0]] if ranks else folder / f"{run_id}{_manifest.MANIFEST_SUFFIX}"
        found["manifest"] = _manifest.find_manifest(folder / f"{run_id}{_manifest.MANIFEST_SUFFIX}", run_id)
    return [by_id[run_id] for run_id in sorted(by_id)]


def discover_runs(source: str | Path) -> list[dict[str, Any]]:
    """Every run at `source`, each as `{"run_id", "source", "path", "manifest"}` (and `files`,
    a `.jsonl` file per rank, for local files). `source` is a folder of local files, a
    server's store (the folder that holds `runs/`, or `runs/` itself), one run folder of a
    store, or one `.jsonl` file. A folder may hold local files and a store both."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist: nothing to list")
    if source.is_file():
        if source.suffix != ".jsonl":
            raise ValueError(f"{source} is not a folder of runs or a .jsonl capture")
        return _local_runs(source.parent, [source])
    if (source / "runs").is_dir() and not _is_run_folder(source):
        return _local_runs(source) + _store_runs(source / "runs")
    if _is_run_folder(source):  # one run of a store
        folder = source.resolve()
        return [{"run_id": folder.name, "source": "store", "path": source,
                 "manifest": _manifest.find_manifest(source, folder.name)}]
    return _local_runs(source) + _store_runs(source)


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


def _local_ranks(files: Mapping[int, Path], manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ranks of a local run, from what the listing reads anyway: the file names, the first
    and the last record of each file, and the manifest's `world_size`. No file is read whole.

    A file named `<run_id>.rank<N>.jsonl` holds rank N alone (`tensorquill.init` writes one
    such file a rank). A file with a name of its own may hold several ranks: its first and last
    records name some of them, and only reading it whole would name them all. Its count is
    settled when the manifest's `world_size` equals the ranks seen; otherwise it is not known,
    and `why` says so (the row prints it rather than a count that may be wrong).
    """
    seen: set[int] = set(files)
    several: list[str] = []
    own_name: list[str] = []
    read: dict[Path, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
    for path in files.values():
        if _named_rank(path) is not None:
            continue
        own_name.append(path.name)
        read[path] = first, last = _first_record(path), _last_record(path)  # kept: the row reads them too
        ends = {r for r in (_record_rank(first), _record_rank(last)) if r is not None}
        if len(ends) > 1:
            several.append(path.name)
        seen |= ends
    world = (manifest or {}).get("world_size")
    world = world if isinstance(world, int) and not isinstance(world, bool) and world > 0 else None
    # `shared`: a lone file read as one run of several ranks, for its ends or for the world size.
    # Not on which rank wrote the last line alone, which changes from one listing to the next.
    found: dict[str, Any] = {"ranks": sorted(seen), "several_in": several, "ends": read,
                             "shared": len(files) == 1 and bool(own_name) and (bool(several) or (world or 1) > 1)}
    if own_name and world != len(seen):
        names = ", ".join(own_name[:4]) + (f" and {len(own_name) - 4} more" if len(own_name) > 4 else "")
        said = (f"the run manifest says world size {world}" if world is not None
                else "no run manifest says the world size" if manifest is None
                else "the run manifest does not say the world size")
        found["why"] = (f"{names} {'has a name' if len(own_name) == 1 else 'have names'} of {'its' if len(own_name) == 1 else 'their'} "
                        f"own, so may hold several ranks; the first and last records name "
                        f"{_rank_list(sorted(seen))}, and {said}. How many ranks the run has is not known without "
                        "reading every record (python -m tensorquill compare reads them)")
    return found


def _rank_list(ranks: list[int]) -> str:
    return f"rank{'s' if len(ranks) != 1 else ''} {', '.join(map(str, ranks))}"


def _named_rank(path: Path) -> int | None:
    """N for a file named `<anything>.rank<N>.jsonl`, as `tensorquill.init` names them."""
    _, _, rank = path.name[: -len(".jsonl")].rpartition(".rank")
    return int(rank) if rank.isdecimal() else None


def _local_row(run: Mapping[str, Any]) -> dict[str, Any]:
    files: Mapping[int, Path] = run.get("files") or {}
    row: dict[str, Any] = {"ranks": sorted(files)}
    if not files:
        row["evidence"] = {"state": "no_records", "detail": "the manifest is written and no rank has written a record yet"}
        return row
    known = _local_ranks(files, run.get("manifest"))
    row["ranks"] = known["ranks"]
    if "why" in known:
        row["rank_count_not_known"] = known["why"]
    if len(files) > 1:
        return _local_all_ranks_row(files, row)
    rank = min(files)
    records = files[rank]
    if known["shared"]:
        return _local_shared_file_row(records, known, row)
    first, last = known["ends"].get(records) or (_first_record(records), _last_record(records))
    if isinstance((last or {}).get("step"), int):
        row["last_step"] = last["step"]
    row["started_from_records"] = _iso((first or {}).get("ts"))
    path = records.with_suffix(".evidence.json")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        written = path.stat().st_mtime
    except FileNotFoundError:
        row["evidence"] = {
            "state": "missing", "rank": rank,
            "detail": f"no evidence file beside {records.name}: write it with python -m tensorquill {records}",
        }
        return row
    except (OSError, ValueError) as error:
        row["evidence"] = {"state": "unreadable", "rank": rank, "path": str(path), "detail": str(error)}
        return row
    if not isinstance(evidence, dict):
        row["evidence"] = {"state": "unreadable", "rank": rank, "path": str(path), "detail": "not a JSON object"}
        return row
    stale = written < records.stat().st_mtime
    row["evidence"] = {
        "state": "older_than_records" if stale else "current", "rank": rank, "path": str(path),
        **({"detail": f"{records.name} has grown since {path.name} was written; the row is as of that file"} if stale else {}),
        **({"compact": True} if isinstance(evidence.get("compact"), Mapping) else {}),
    }
    incidents = evidence.get("incidents") or []
    row["run_state"] = (evidence.get("run_state") or {}).get("state")
    row["incidents"] = _by_class(incidents)
    row["first_incident"] = _first(incidents)
    row["kinds_checked"] = _kinds_checked(evidence.get("kind_checks"))
    return row


def _local_shared_file_row(records: Path, known: Mapping[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """One `.jsonl` file that holds several ranks. Its evidence file, if any, is one rank's:
    the command line judges one rank of such a file (`--rank N`) and does not write which. So
    its incidents are listed as found, the run state is `unhealthy` if it shows any and
    otherwise unknown (the other ranks are not judged: the all-ranks rule), and what was
    checked is not counted for the run."""
    first, last = known["ends"][records]
    if isinstance((last or {}).get("step"), int):
        row["last_step"] = last["step"]  # the last line's rank's; another rank may be a step behind
    row["started_from_records"] = _iso((first or {}).get("ts"))
    holds = (f"may hold several ranks (its first and last records name {_rank_list(known['ranks'])})" if "why" in known
             else f"holds {_rank_list(known['ranks'])}")
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
    row["evidence"] = {
        "state": "ranks_unknown", "rank": None, "path": look.get("path"), "judged_ranks": 1,
        "unknown_ranks": len(known["ranks"]) - 1 if "why" not in known else None,
        **({"compact": True} if isinstance(evidence.get("compact"), Mapping) else {}),
        "detail": f"{records.name} {holds}, and its evidence file is one "
                  "rank's, without saying which: the row's incidents are that rank's, the other ranks are not judged, "
                  "and what was checked is not counted for the run (python -m tensorquill compare reads every rank)"
                  + ("; the records have grown since it was written" if look["state"] == "older_than_records" else ""),
    }
    return row


def _read_evidence_file(records: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """A rank's evidence file beside its `.jsonl` file, and what the row says of it."""
    path = records.with_suffix(".evidence.json")
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


def _local_all_ranks_row(files: Mapping[int, Path], row: dict[str, Any]) -> dict[str, Any]:
    """A local run of several ranks: every rank's evidence file, put together by the
    all-ranks view's rules (`combine_rank_evidence`). No record is judged here: a rank without
    an evidence file is unknown, never clean."""
    from .evidence import _rank_ranges, combine_rank_evidence

    per_rank: dict[int, dict[str, Any]] = {}
    unknown: dict[int, str] = {}
    looks: dict[int, dict[str, Any]] = {}
    steps: list[int] = []
    for number in sorted(files):
        records = files[number]
        last = _last_record(records)
        step = (last or {}).get("step")
        if isinstance(step, int) and not isinstance(step, bool):
            steps.append(step)
        evidence, look = _read_evidence_file(records)
        looks[number] = look
        if evidence is None:
            unknown[number] = f"{look['state']}: {look['detail']}"
        else:
            per_rank[number] = {**evidence, "last_step": step if isinstance(step, int) else None}
    if steps:
        row["last_step"] = max(steps)
    first = _first_record(files[min(files)])
    row["started_from_records"] = _iso((first or {}).get("ts"))
    stale = [n for n, look in looks.items() if look["state"] == "older_than_records"]
    if not per_rank:
        row["evidence"] = {"state": "missing", "judged_ranks": 0, "unknown_ranks": len(files),
                           "detail": f"no evidence file beside any of its {len(files)} ranks' .jsonl files: write each "
                                     f"with python -m tensorquill <the rank's .jsonl file>, as for {files[min(files)].name}"}
        return row
    combined = combine_rank_evidence(per_rank, unknown_ranks=unknown)
    state = "ranks_unknown" if unknown else "older_than_records" if stale else "current"
    notes = []
    if unknown:
        notes.append(f"rank{'s' if len(unknown) != 1 else ''} {_rank_ranges(sorted(unknown), 8)} "
                     f"{'have' if len(unknown) != 1 else 'has'} no evidence file (write it with python -m tensorquill "
                     f"<the rank's .jsonl file>), so nothing is said about incidents there")
    if stale:
        notes.append(f"the records of rank{'s' if len(stale) != 1 else ''} {_rank_ranges(stale, 8)} have grown since "
                     "their evidence files were written; the row is as of those files")
    row["evidence"] = {
        "state": state, "from": "each rank's evidence file", "judged_ranks": len(per_rank), "unknown_ranks": len(unknown),
        **({"detail": "; ".join(notes)} if notes else {}),
    }
    _all_ranks_cells(row, combined)
    return row


def _all_ranks_cells(row: dict[str, Any], view: Mapping[str, Any]) -> None:
    """The run state, incidents and kinds checked of a run of several ranks, from the
    all-ranks view or `combine_rank_evidence`: the whole run's, with no rank named."""
    row["run_state"] = (view.get("run_state") or {}).get("state")
    incidents = [group["incident"] for group in view.get("incident_groups") or () if isinstance(group.get("incident"), Mapping)]
    incidents += [i for i in view.get("run_incidents") or () if isinstance(i, Mapping)]
    row["incidents"] = _by_class(incidents)
    row["first_incident"] = _first(incidents)
    row["ranks_with_incidents"] = list(view.get("ranks_with_incidents") or ())
    row["kinds_checked"] = _kinds_checked(view.get("kind_checks"))
    if row["kinds_checked"] is None and isinstance(view.get("kind_checks_not_stated"), Mapping):
        row["kinds_not_stated"] = view["kind_checks_not_stated"].get("detail")


def _store_row(run: Mapping[str, Any]) -> dict[str, Any]:
    """From each rank's `incidents.json`, the names of the newest parts and `watchdog.json`."""
    from .evidence import _report_evidence, all_ranks_evidence, read_server_report, read_watchdog

    folder = Path(run["path"])
    row: dict[str, Any] = {}
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
    incidents = [group["incident"] for group in view.get("incident_groups") or () if isinstance(group.get("incident"), Mapping)]
    incidents += [i for i in view.get("run_incidents") or () if isinstance(i, Mapping)]
    row["incidents"] = _by_class(incidents)
    row["first_incident"] = _first(incidents)
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
    row.update(_local_row(run) if run["source"] == "local" else _store_row(run))
    if run["source"] == "local" and run.get("files"):
        row["end"] = _local_end(run["files"])
    # None when the files cannot say without being read whole; `rank_count_not_known` says why.
    row["rank_count"] = None if row.get("rank_count_not_known") else len(row.get("ranks") or ())
    if row["started"] is None:
        row["started"] = row.get("started_from_records")
    row.pop("started_from_records", None)
    return row


def _local_end(files: Mapping[int, Path]) -> dict[str, Any]:
    """Whether every rank's session closed its file (`sink.read_end_marker`). `closed` only when
    each file has a marker that matches it; otherwise `not_closed`, with the ranks: still running,
    killed or never closed look the same from the files. A store's run has no such cell: the
    server's status says how each stream ended."""
    ends = {number: read_end_marker(path) for number, path in files.items()}
    closed = sorted(n for n, end in ends.items() if end["state"] == "closed")
    raised = sorted(n for n, end in ends.items() if end["state"] == "closed" and end.get("outcome") == "raised")
    other = sorted(n for n in ends if n not in closed)
    found: dict[str, Any] = {"state": "closed" if not other else "not_closed", "closed_ranks": closed}
    if raised:
        found["raised_ranks"] = raised
    if other:
        found["not_closed_ranks"] = other
        found["detail"] = ends[other[0]]["detail"]
    return found


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
    return ", ".join(f"{counts[name]} {name}" for name in names)


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
            if row.get("rank_count_not_known"):
                notes.append(f"{row['run_id']}: ranks not counted: {row['rank_count_not_known']}")
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
