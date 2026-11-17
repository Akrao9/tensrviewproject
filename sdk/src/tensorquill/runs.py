"""List the runs in a folder of local files or in a server's store, by project.

    python -m tensorquill runs <folder or store> [--project P] [--json]

    import tensorquill
    tensorquill.list_runs("tensorquill-out", project="granite")

One row a run: its name, run ID, run state, last step, incidents by class, the first incident,
how many kinds were checked, and its start time. A run is found by its files:

- local files, as `tensorquill.init` writes them: `<run_id>.rank<N>.jsonl`, with the manifest
  `<run_id>.run.json` beside them and, once `python -m tensorquill <file>` has run, the
  evidence file `<file>.evidence.json`;
- a server's store: `runs/<run_id>/` with `run.json`, `watchdog.json` and a `rank=<n>` folder
  for each rank. `<folder or store>` is the folder that holds `runs/`, or `runs/` itself.

The listing reads the manifest and the small evidence or report files only. It never opens a
Parquet part and never runs a detector, so a local run whose evidence file was never written
is listed with its state unknown and a note saying how to write it. Of a `.jsonl` file it
reads the first line and the last one (for the run ID, the start time and the last step).

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

DEFAULT_SOURCE = "tensorquill-out"  # where `tensorquill.init` writes without a sink
VERBS = ("runs", "compare")
_TAIL_BYTES = 1 << 20  # the most of a `.jsonl` file's end read for its last step
_HEAD_BYTES = 1 << 20


class RunLookupError(ValueError):
    """A run named by ID or name was not found, or the name fits several runs."""


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
    stem = path.name[: -len(".jsonl")]
    _, _, rank = stem.rpartition(".rank")
    if rank.isdecimal():
        return int(rank)
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
    several runs raises `RunLookupError` with their IDs. A run kept both as local files and in
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
        raise RunLookupError(
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


def _local_row(run: Mapping[str, Any]) -> dict[str, Any]:
    files: Mapping[int, Path] = run.get("files") or {}
    row: dict[str, Any] = {"ranks": sorted(files)}
    if not files:
        row["evidence"] = {"state": "no_records", "detail": "the manifest is written and no rank has written a record yet"}
        return row
    rank = min(files)
    records = files[rank]
    first, last = _first_record(records), _last_record(records)
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
    row["run_state"] = (view.get("run_state") or {}).get("state")
    last = view.get("last_step")
    if isinstance(last, Mapping):
        row["last_step"] = last.get("max")
    incidents = [group["incident"] for group in view.get("incident_groups") or () if isinstance(group.get("incident"), Mapping)]
    incidents += [i for i in view.get("run_incidents") or () if isinstance(i, Mapping)]
    row["incidents"] = _by_class(incidents)
    row["first_incident"] = _first(incidents)
    # Until the all-ranks view carries `kind_checks`, the count is the lowest judged rank's.
    if isinstance(view.get("kind_checks"), Mapping):
        row["kinds_checked"] = _kinds_checked(view["kind_checks"])
    elif judged:
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
    if row["started"] is None:
        row["started"] = row.get("started_from_records")
    row.pop("started_from_records", None)
    return row


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
        "?" if row.get("last_step") is None else str(row["last_step"]),
        _incidents_text(row),
        f"{first['kind']} at step {first['step']}" if first else ("-" if row.get("incidents") is not None else "?"),
        f"{checked['checked']} of {checked['of']}" if checked else "?",
        str(row.get("started") or "?").replace("T", " ").rstrip("Z"),
    ]


def listing_text(rows: list[Mapping[str, Any]], source: str | Path, project: str | None = None) -> str:
    if not rows:
        return f"no runs at {source}" + (f" in project {project!r}" if project is not None else "")
    header = ["name", "run ID", "state", "last step", "incidents", "first incident", "kinds checked", "started (UTC)"]
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
