"""Compare two or more runs: their setup, what was checked, and their incidents.

    python -m tensorquill compare A B [C ...] [--source FOLDER] [--project P] [-o compare.json]

    import tensorquill
    result = tensorquill.compare_runs("baseline", "after-the-fix", source="tensorquill-out")

Each run is given by its run ID or its name, looked up at `--source` (a folder of local files
or a server's store, as `python -m tensorquill runs` lists it; default `tensorquill-out`), or
as a path: a `.jsonl` capture, a run folder of a store, or one rank's `evidence.json`. An
exact run ID wins; a name that fits several runs is refused with their IDs.

The result has four parts, in this order, and the printed summary keeps the order:

1. `setup`: what differs between the run manifests: project, group, world size, the SDK and
   PyTorch versions, and each `config` key that differs.
2. `checked`: the runs' `kind_checks` side by side. A kind checked in one run and not
   configured, unavailable or not yet with enough history in another is listed here, before
   any incident, so "no incident" in one run is never set against "not looked for" in another.
3. `incidents`: by kind and what the incident names (the layer and router, the parameters,
   the modules; for the run's own incidents, such as a stall or a stuck collective, how it
   stopped or which collective, with each run's ranks in `where`), with the runs that show it
   and each run's first step. For a run that does not show it, the entry says whether that
   run looked.
4. `divergence`: where two runs part (`where_runs_part`, which calls `divergence.first_divergence`).

Every rank of each run is read, unless `rank=` names one rank to read of every run. A run of
several ranks is compared as a whole, by the all-ranks view's rules: its incidents are every
judged rank's, grouped across ranks (an entry's `ranks` names the ranks that show it), and a
kind is `checked` for the run only when every judged rank checked it with enough history. A
run in a server's store is read through the all-ranks view (`all_ranks_evidence`: each rank's
`incidents.json` and the status; no part is opened); a rank whose report is not current is
unknown there, and the command line judges it from its parts, as for one rank, when they can
be read. When they cannot (no `pyarrow`, a torn part), the rank stays unknown with why
(`read_from.unknown_why`), for one rank read alone too: the comparison is still made, and
never counts that rank as clean. Local files are judged here by the Python detectors with the default thresholds,
each rank on its own, and put together by the same rules (`combine_rank_evidence`). Ranks
that are not judged are named (`other_ranks_not_compared`, `read_from.unknown_ranks`), never
counted as clean. A run of one rank is read as before.

Part 4 compares rank r of one run with rank r of the other, for each rank, and gives the
earliest divergence with its rank. Runs whose ranks differ are not compared there
(`ranks_differ`); nothing is ever made across ranks.

`compare_runs` returns the result and writes nothing. The command line writes `compare.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from . import manifest as _manifest
from . import runs as _runs

COMPARE_VIEW = "compare"
COMPARE_VIEW_VERSION = 1
DEFAULT_OUTPUT = "compare.json"
SETUP_FIELDS = ("project", "group", "world_size", "sdk_version", "torch_version")
_WHERE_NAMES = 8  # the most names of one list an incident's `where` carries


def _labels(count: int) -> list[str]:
    return [chr(ord("A") + i) if i < 26 else f"R{i + 1}" for i in range(count)]


# --- Opening a run --------------------------------------------------------------------------


def _is_step(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_local(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Every record of a `.jsonl` file, and whether a line that is not whole was left out."""
    from .evidence import _Tail, load_records

    try:
        return load_records(path, all_attempts=True), False
    except ValueError:
        # A line the job is still writing, or one torn by a killed process: the whole lines are
        # read, as the watcher reads them, and the result says so.
        tail = _Tail(Path(path), None)
        tail.poll()
        return tail.records, True


def _rank_of(record: Mapping[str, Any]) -> int:
    return int(record.get("rank", 0))


def _open_local(run: Mapping[str, Any], rank: int | None) -> dict[str, Any]:
    from .evidence import build_evidence

    files: Mapping[int, Path] = run.get("files") or {}
    if not files:
        raise _runs.RunLookupError(f"run {run['run_id']} has a manifest and no records yet: nothing to compare")
    path = files.get(rank) if rank is not None and rank in files else files[min(files)]
    records, unfinished = _read_local(path)
    if rank is None and len(files) > 1:
        # Every rank's file is read, whether or not the lowest rank's holds a whole record yet:
        # a rank whose process has written none is unknown there, never a reason to compare
        # nothing while the other ranks have records.
        return _open_local_all_ranks(run, files, path, records, unfinished)
    if not records:
        raise _runs.RunLookupError(f"run {run['run_id']} has no whole record in {path} yet: nothing to compare")
    held = sorted({_rank_of(r) for r in records})
    ranks = sorted(set(files) | set(held))
    if rank is None and len(ranks) > 1:
        return _open_local_all_ranks(run, files, path, records, unfinished)
    # The rank asked for; a run with one rank is read on the rank it has, and says which.
    picked = rank if rank in held else held[0] if held and (rank is None or len(ranks) == 1) else None
    if picked is None:
        raise _runs.RunLookupError(f"run {run['run_id']} has no rank {rank} here: it has rank{'s' if len(ranks) != 1 else ''} "
                                   f"{', '.join(map(str, ranks))}")
    records = [r for r in records if _rank_of(r) == picked]
    evidence = build_evidence(records, run_manifest=run.get("manifest") or {})
    steps = [r["step"] for r in records if _is_step(r.get("step"))]
    return {
        "evidence": evidence, "last_step": max(steps) if steps else None,
        "read_from": {"path": str(path), "kind": "local records", "rank": picked, "ranks": ranks,
                      "judged_by": "the Python detectors, here, with the default thresholds",
                      **({"unfinished_lines_left_out": True} if unfinished else {})},
        "records": {"path": str(path), "rank": picked},
    }


def _all_ranks_run(combined: Mapping[str, Any]) -> dict[str, Any]:
    """The run-level evidence parts 2 and 3 read, from the all-ranks view or
    `combine_rank_evidence`: the view's keys, with `incidents` the grouped incidents (one for
    each group, whatever its ranks) and the run's own."""
    groups = [g for g in combined.get("incident_groups") or () if isinstance(g, Mapping) and isinstance(g.get("incident"), Mapping)]
    run_incidents = [i for i in combined.get("run_incidents") or () if isinstance(i, Mapping)]
    return {**combined, "incidents": [g["incident"] for g in groups] + run_incidents}


def _open_local_all_ranks(run: Mapping[str, Any], files: Mapping[int, Path], first: Path,
                          records: list[dict[str, Any]], unfinished: bool) -> dict[str, Any]:
    """Every rank of a run kept as local files: each rank judged here by `build_evidence`, as
    one rank's is, and the ranks put together by the all-ranks view's rules
    (`combine_rank_evidence`)."""
    from .evidence import build_evidence, combine_rank_evidence

    by_rank: dict[int, list[dict[str, Any]]] = {}
    paths: dict[int, str] = {}
    unfinished_in: list[str] = [str(first)] if unfinished else []
    for path in [first, *(files[n] for n in sorted(files) if files[n] != first)]:
        found, torn = (records, unfinished) if path == first else _read_local(path)
        if torn and str(path) not in unfinished_in:
            unfinished_in.append(str(path))
        for record in found:
            number = _rank_of(record)
            if paths.setdefault(number, str(path)) == str(path):  # a rank is read from the first file that holds it
                by_rank.setdefault(number, []).append(record)
    if not by_rank:
        names = ", ".join(sorted({p.name for p in files.values()}))
        raise _runs.RunLookupError(f"run {run['run_id']} has no whole record in any of its files ({names}) yet: nothing to compare")
    unknown = {n: f"its file {files[n].name} holds no whole record yet" for n in files if n not in by_rank}
    per_rank = {}
    for number, own in sorted(by_rank.items()):
        steps = [r["step"] for r in own if _is_step(r.get("step"))]
        per_rank[number] = {**build_evidence(own, run_manifest=run.get("manifest") or {}),
                            "last_step": max(steps) if steps else None}
    combined = combine_rank_evidence(per_rank, unknown_ranks=unknown)
    last = combined.get("last_step")
    ranks = sorted(set(per_rank) | set(unknown))
    return {
        "evidence": _all_ranks_run(combined), "last_step": last.get("max") if isinstance(last, Mapping) else None,
        "read_from": {"path": str(first.parent if len(set(files.values())) > 1 else first), "kind": "local records",
                      "rank": None, "ranks": ranks, "all_ranks": True,
                      "judged_ranks": combined["judged_ranks"], "unknown_ranks": combined["unknown_ranks"],
                      **({"unknown_why": unknown} if unknown else {}),
                      "judged_by": "the Python detectors, here, with the default thresholds, on each rank; the ranks put "
                                   "together by the all-ranks view's rules",
                      **({"unfinished_lines_left_out": True, "unfinished_in": unfinished_in} if unfinished_in else {})},
        "records": {"path": str(first), "rank": None, "ranks": {n: paths[n] for n in sorted(per_rank)}},
    }


def _open_store(run: Mapping[str, Any], rank: int | None, read_parts: bool) -> dict[str, Any]:
    from .evidence import (
        _discover_rank_folders, _ranks_without_folder, _report_evidence, applied_thresholds, build_evidence,
        load_records, read_server_report, read_watchdog,
    )

    folder = Path(run["path"])
    folders = _discover_rank_folders(folder)
    if not folders:
        raise _runs.RunLookupError(f"run {run['run_id']} at {folder} has no rank=<n> folder yet: nothing to compare")
    # Several ranks, as the all-ranks view counts them: the folders, and the ranks the server's
    # status expects that stored nothing (unknown, never clean).
    expected = _ranks_without_folder(folders, read_watchdog(folder / "watchdog.json")) if rank is None else {}
    if rank is None and len(folders) + len(expected) > 1:
        return _open_store_all_ranks(run, folder, folders, read_parts)
    picked = rank if rank in folders else min(folders) if rank is None or len(folders) == 1 else None
    if picked is None:
        raise _runs.RunLookupError(f"run {run['run_id']} has no rank {rank}: its store holds ranks "
                                   f"{', '.join(map(str, sorted(folders)))}")
    look = read_server_report(folders[picked])
    if look["state"] == "current":
        evidence = _report_evidence(folders[picked], look, read_watchdog(folder / "watchdog.json"))
        last = look["report"].get("last_step")
        judged = "the server's report (incidents.json), with the thresholds the server applied; no part was read"
    elif not read_parts:
        raise _runs.RunLookupError(
            f"run {run['run_id']} rank {picked} has no current report from its server: it is {look['state']} "
            f"({look['reason']}). Nothing current to compare without reading its parts and judging them here, which "
            f"`python -m tensorquill compare` does")
    else:
        # The report cannot be used (see `read_server_report`), so the parts are read and judged
        # here, as the watcher does: with the thresholds the server applied, when a report says
        # them, so the incidents do not change with who judged them.
        applied = applied_thresholds(look["report"]) if isinstance(look.get("report"), Mapping) else {}
        try:
            records = load_records(folder, rank=picked, all_attempts=True)
        except (OSError, ValueError, ImportError) as error:  # ImportError: the parts need pyarrow
            return _unread_store_rank(run, folder, folders, picked,
                                      f"{look['state']}: {look['reason']}; its parts could not be read here ({error})")
        evidence = build_evidence(records, watchdog=folder / "watchdog.json", run_manifest=run.get("manifest") or {},
                                  **applied)
        steps = [r["step"] for r in records if _is_step(r.get("step"))]
        last = max(steps) if steps else None
        judged = ("the Python detectors, here, on the stored parts, with "
                  + ("the thresholds the server's last report applied" if applied else "the default thresholds")
                  + f": the server's report is {look['state']} ({look['reason']})")
    return {
        "evidence": evidence, "last_step": last if _is_step(last) else None,
        "read_from": {"path": str(folder), "kind": "server store", "rank": picked, "ranks": sorted(folders),
                      "judged_by": judged},
        "records": {"path": str(folder), "rank": picked},
    }


def _unread_store_rank(run: Mapping[str, Any], folder: Path, folders: Mapping[int, Path], picked: int,
                       why: str) -> dict[str, Any]:
    """One rank of a store whose report is not current and whose parts cannot be read here (no
    `pyarrow`, a torn part): unknown, with why, as the all-ranks path leaves such a rank. Its
    evidence is the all-ranks rules' over no judged rank: no incident of its own, what was
    checked not stated, the run state unknown; the server's status still counts."""
    from .evidence import combine_rank_evidence, read_watchdog

    combined = combine_rank_evidence({}, status=read_watchdog(folder / "watchdog.json"), status_expected=True,
                                     unknown_ranks={picked: why})
    return {
        "evidence": _all_ranks_run(combined), "last_step": None,
        "read_from": {"path": str(folder), "kind": "server store", "rank": picked, "ranks": sorted(folders),
                      "judged_ranks": [], "unknown_ranks": [picked], "unknown_why": {picked: why},
                      "judged_by": f"nothing: rank {picked} is not judged ({why})"},
        "records": {"path": str(folder), "rank": picked},
    }


def _open_store_all_ranks(run: Mapping[str, Any], folder: Path, folders: Mapping[int, Path],
                          read_parts: bool) -> dict[str, Any]:
    """Every rank of a run in a server's store: the all-ranks view (`all_ranks_evidence`),
    which reads each rank's report and the status and opens no part. A rank whose report is
    not current is in no value there. With `read_parts`, such a rank's parts are read and
    judged here, as for one rank, and the ranks are put together by the same rules
    (`combine_rank_evidence`); a rank whose parts cannot be read (no `pyarrow`, a torn part)
    stays unknown, with why."""
    from .evidence import (
        _report_evidence, all_ranks_evidence, applied_thresholds, build_evidence, combine_rank_evidence,
        load_records, read_server_report, read_watchdog,
    )

    view = all_ranks_evidence(folder)
    reasons = {entry["rank"]: f"{entry['detection'].get('state')}: {entry['detection'].get('reason')}"
               for entry in view.get("ranks") or () if entry.get("rank") in view.get("unknown_ranks", ())}
    stale = [n for n in view.get("unknown_ranks") or () if n in folders]
    if not view.get("judged_ranks") and not (read_parts and stale):
        raise _runs.RunLookupError(
            f"run {run['run_id']} has no rank with a current report from its server ("
            + "; ".join(f"rank {n} {why}" for n, why in sorted(reasons.items())[:4])
            + "). Nothing current to compare without reading its parts and judging them here, which "
              "`python -m tensorquill compare` does")
    judged_here: list[int] = []
    if read_parts and stale:
        status = read_watchdog(folder / "watchdog.json")
        per_rank: dict[int, dict[str, Any]] = {}
        memory: dict[int, Any] = {}
        unknown = {n: why for n, why in reasons.items() if n not in folders}
        for number in sorted(folders):
            look = read_server_report(folders[number])
            if look["state"] == "current":
                report = look["report"]
                per_rank[number] = {**_report_evidence(folders[number], look, status), "last_step": report.get("last_step")}
                memory[number] = report.get("memory")
                continue
            applied = applied_thresholds(look["report"]) if isinstance(look.get("report"), Mapping) else {}
            try:
                records = load_records(folder, rank=number, all_attempts=True)
            except (OSError, ValueError, ImportError) as error:  # ImportError: the parts need pyarrow
                unknown[number] = f"{look['state']}: {look['reason']}; its parts could not be read here ({error})"
                continue
            steps = [r["step"] for r in records if _is_step(r.get("step"))]
            per_rank[number] = {**build_evidence(records, watchdog=folder / "watchdog.json",
                                                 run_manifest=run.get("manifest") or {}, **applied),
                                "last_step": max(steps) if steps else None}
            judged_here.append(number)
        combined = combine_rank_evidence(per_rank, status=status, status_expected=True, unknown_ranks=unknown,
                                         memory=memory)
    else:
        combined, unknown = view, reasons
    last = combined.get("last_step")
    judged = ("the all-ranks view: each rank's report from its server (incidents.json), with the thresholds the "
              "server applied, and the server's status; no part was read")
    if judged_here:
        judged += (f", but for rank{'s' if len(judged_here) != 1 else ''} {', '.join(map(str, judged_here))}, whose "
                   "reports are not current: the Python detectors judged their stored parts here, with the thresholds "
                   "the server's last report applied")
    unknown_now = list(combined.get("unknown_ranks") or ())
    return {
        "evidence": _all_ranks_run(combined), "last_step": last.get("max") if isinstance(last, Mapping) else None,
        "read_from": {"path": str(folder), "kind": "server store", "rank": None,
                      "ranks": sorted(set(folders) | set(unknown_now)), "all_ranks": True,
                      "judged_ranks": list(combined.get("judged_ranks") or ()), "unknown_ranks": unknown_now,
                      **({"judged_here": judged_here} if judged_here else {}),
                      **({"unknown_why": {n: unknown.get(n) for n in unknown_now}} if unknown_now else {}),
                      "judged_by": judged},
        "records": {"path": str(folder), "rank": None, "ranks": {n: str(folder) for n in sorted(folders)}},
    }


def _open_evidence_file(path: Path, rank: int | None) -> dict[str, Any]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _runs.RunLookupError(f"{path} cannot be read as an evidence file ({error})") from None
    if isinstance(evidence, dict) and evidence.get("view") == "all_ranks" and isinstance(evidence.get("incident_groups"), list):
        if rank is not None:
            raise _runs.RunLookupError(f"{path} is an all-ranks view; to read rank {rank} alone, give the run folder")
        return _open_all_ranks_file(path, evidence)
    if not isinstance(evidence, dict) or "incidents" not in evidence:
        raise _runs.RunLookupError(f"{path} is not one rank's evidence or an all-ranks view")
    own = (evidence.get("detection") or {}).get("rank") if isinstance(evidence.get("detection"), Mapping) else None
    if rank is not None and _is_step(own) and own != rank:
        raise _runs.RunLookupError(f"{path} is rank {own}'s evidence, not rank {rank}'s")
    run_id = evidence.get("run_id")
    block = evidence.get("run") if isinstance(evidence.get("run"), Mapping) else None
    return {
        "run_id": run_id, "manifest": None,
        "block": {**_manifest.run_block(run_id, block), "run_id": run_id},
        "evidence": evidence, "last_step": None,
        "read_from": {"path": str(path), "kind": "evidence file", "rank": own if _is_step(own) else None, "ranks": None,
                      "judged_by": "whatever wrote the file" + (" (compact)" if isinstance(evidence.get("compact"), Mapping) else "")},
        "records": None,
    }


def _open_all_ranks_file(path: Path, view: Mapping[str, Any]) -> dict[str, Any]:
    """An all-ranks view as the watcher wrote it (`--rank all`): read as it stands."""
    run_id = view.get("run_id")
    block = view.get("run") if isinstance(view.get("run"), Mapping) else None
    last = view.get("last_step")
    ranks = sorted(e["rank"] for e in view.get("ranks") or () if isinstance(e, Mapping) and _is_step(e.get("rank")))
    return {
        "run_id": run_id, "manifest": None,
        "block": {**_manifest.run_block(run_id, block), "run_id": run_id},
        "evidence": _all_ranks_run(view),
        "last_step": last.get("max") if isinstance(last, Mapping) and _is_step(last.get("max")) else None,
        "read_from": {"path": str(path), "kind": "evidence file", "rank": None, "ranks": ranks, "all_ranks": True,
                      "judged_ranks": list(view.get("judged_ranks") or ()),
                      "unknown_ranks": list(view.get("unknown_ranks") or ()),
                      "judged_by": "whatever wrote the file (the all-ranks view)"},
        "records": None,
    }


def open_run(wanted: str | Path, source: str | Path | None = None, project: str | None = None,
             rank: int | None = None, *, paths: bool = True, read_parts: bool = True) -> dict[str, Any]:
    """One run, ready to compare: its manifest, one rank's evidence and where both were read.

    `wanted` is a run ID or name at `source`, or (with `paths`) a path. The lookup at `source`
    comes first, so an ID never changes meaning because a file of that name appears.
    `read_parts=False` refuses a run in a store whose report is not current, where the parts
    would otherwise be opened and judged here.
    """
    source = Path(source) if source is not None else Path(_runs.DEFAULT_SOURCE)
    run: dict[str, Any] | None = None
    lookup_error: Exception | None = None
    if isinstance(wanted, str):
        try:
            run = _runs.find_run(source, wanted, project)
        except FileNotFoundError as error:
            lookup_error = _runs.RunLookupError(str(error))
        except _runs.AmbiguousRunName:  # several runs of that name: never guessed around
            raise
        except _runs.RunLookupError as error:
            lookup_error = error
    if run is None:
        path = Path(wanted)
        if not paths or not path.exists():
            if lookup_error is not None:
                raise lookup_error
            raise _runs.RunLookupError(f"{path} does not exist")
        if path.is_file() and path.suffix != ".jsonl":
            return _open_evidence_file(path, rank)
        found = _runs.discover_runs(path)
        if len(found) != 1:
            raise _runs.RunLookupError(
                f"{path} holds {len(found)} runs; name one by its ID or name with source={str(path)!r}")
        run = found[0]
    opened = _open_local(run, rank) if run["source"] == "local" else _open_store(run, rank, read_parts)
    return {"run_id": run["run_id"], "manifest": run.get("manifest"),
            "block": _manifest.run_block(run["run_id"], run.get("manifest")), **opened}


# --- Part 1: the setup ----------------------------------------------------------------------


def _differ(values: list[Any]) -> bool:
    return any(json.dumps(v, sort_keys=True, default=str) != json.dumps(values[0], sort_keys=True, default=str)
               for v in values[1:])


def compare_setup(opened: list[Mapping[str, Any]], labels: list[str]) -> dict[str, Any]:
    known = [(label, run["manifest"]) for label, run in zip(labels, opened) if isinstance(run.get("manifest"), Mapping)]
    differs: list[dict[str, Any]] = []
    same: list[str] = []
    if len(known) > 1:
        for field in SETUP_FIELDS:
            values = [m.get(field) for _, m in known]
            if _differ(values):
                differs.append({"field": field, "values": {label: value for (label, _), value in zip(known, values)}})
            else:
                same.append(field)
    # A config that was too large for the stream is not in the store's copy of the manifest
    # (`manifest.manifest_to_send`): not known, which is not the same as "no keys set".
    config_unknown = [label for label, m in known if m.get("config_left_out")]
    with_config = [(label, m) for label, m in known if label not in config_unknown]
    same_config: list[str] = []
    if len(with_config) > 1:
        configs = [m.get("config") if isinstance(m.get("config"), Mapping) else {} for _, m in with_config]
        keys = sorted({str(key) for config in configs for key in config})
        for key in keys:
            values = [config.get(key) for config in configs]
            missing = [label for (label, _), config in zip(with_config, configs) if key not in config]
            if missing or _differ(values):
                differs.append({
                    "field": f"config.{key}",
                    "values": {label: value for (label, _), value in zip(with_config, values)},
                    **({"not_set_in": missing} if missing else {}),
                })
        same_config = [key for key in keys if not any(d["field"] == f"config.{key}" for d in differs)]
    return {
        "differs": differs,
        "same": same,
        "config_keys_same": same_config,
        **({"config_not_known": config_unknown} if config_unknown else {}),
        # A run without a manifest says nothing of its setup; it is left out of the differences.
        "no_manifest": [label for label, run in zip(labels, opened) if not isinstance(run.get("manifest"), Mapping)],
    }


# --- Part 2: what was checked ---------------------------------------------------------------


def _check_of(evidence: Mapping[str, Any], kind: str) -> Mapping[str, Any] | None:
    checks = evidence.get("kind_checks")
    found = checks.get(kind) if isinstance(checks, Mapping) else None
    return found if isinstance(found, Mapping) else None


def _looked(evidence: Mapping[str, Any], kind: Any) -> str:
    """Whether one run's evidence looked for `kind`: `checked`, `not_enough_history` (checked,
    but its slowest rule has not had the records it needs), `not_configured`, `unavailable`,
    or `not_said` for evidence without `kind_checks` or a kind it does not list. For a run of
    several ranks read whole, also `history_not_stated`: checked on every judged rank, and
    some rank does not say whether it had the records it needs."""
    check = _check_of(evidence, kind) if isinstance(kind, str) else None
    if check is None:
        return "not_said"
    value = check.get("value")
    if value == "checked" and "judged_ranks" in evidence:
        # The whole run's (the all-ranks rule): `checked` on every judged rank, and its history
        # is enough only when every one of them says so. A rank that does not say is not read
        # as enough.
        from .evidence import kind_history

        return {"enough": "checked", "not_enough": "not_enough_history"}.get(kind_history(check), "history_not_stated")
    if value == "checked":
        return "checked" if check.get("enough_history", True) else "not_enough_history"
    return value if isinstance(value, str) else "not_said"


def compare_checked(opened: list[Mapping[str, Any]], labels: list[str]) -> dict[str, Any]:
    from .evidence import INCIDENT_CLASSES

    kinds = list(INCIDENT_CLASSES)
    for run in opened:  # kinds a newer writer lists and this version does not know
        checks = run["evidence"].get("kind_checks")
        kinds += [kind for kind in (checks if isinstance(checks, Mapping) else ()) if kind not in kinds]
    not_said = [label for label, run in zip(labels, opened) if not isinstance(run["evidence"].get("kind_checks"), Mapping)]
    said = [(label, run) for label, run in zip(labels, opened) if label not in not_said]
    differs: list[dict[str, Any]] = []
    same = 0
    for kind in kinds:
        states = {label: _looked(run["evidence"], kind) for label, run in said}
        if len(set(states.values())) <= 1:
            same += 1
            continue
        entry: dict[str, Any] = {"kind": kind, "looked": states}
        for field in ("records", "records_needed"):
            values = {label: (_check_of(run["evidence"], kind) or {}).get(field) for label, run in said}
            if any(value is not None for value in values.values()):
                entry[field] = values
        needs = next((c["needs"] for _, run in said if (c := _check_of(run["evidence"], kind)) and c.get("needs")), None)
        if needs:
            entry["needs"] = needs
        # In a run read on every rank, the ranks by value when they differ within that run.
        by_rank = {label: c["ranks"] for label, run in said
                   if (c := _check_of(run["evidence"], kind)) and isinstance(c.get("ranks"), Mapping)}
        if by_rank:
            entry["ranks"] = by_rank
        differs.append(entry)
    whole = {label: run["read_from"].get("judged_ranks") or [] for label, run in zip(labels, opened)
             if run["read_from"].get("all_ranks")}
    return {
        "ranks_read": {label: run["read_from"].get("rank") for label, run in zip(labels, opened)},
        # Runs read on every rank (`ranks_read` is None for them): the judged ranks, which are
        # the ones the run's `kind_checks` count.
        **({"all_ranks_read": whole} if whole else {}),
        "differs": differs,
        "same": same if len(said) > 1 else 0,
        "not_said_by": not_said,
        **({"not_said_why": {label: (run["evidence"].get("kind_checks_not_stated") or {}).get("detail")
                             for label, run in zip(labels, opened)
                             if label in not_said and isinstance(run["evidence"].get("kind_checks_not_stated"), Mapping)}}
           if any(isinstance(run["evidence"].get("kind_checks_not_stated"), Mapping) for run in opened) else {}),
    }


# --- Part 3: the incidents ------------------------------------------------------------------


# The run's own kinds: from the server's status, the checks across ranks, and the sources read
# beside the records. The all-ranks view never groups them across ranks, so
# `evidence._incident_identity` does not know them; their identity here is `_run_level_parts`.
RUN_LEVEL_KINDS = ("job_stalled", "rank_silent", "duplicate_batches", "memory_imbalance", "collective_hang",
                   "collective_desync", "traincheck_violation")
_STALL_KINDS = ("job_stalled", "rank_silent")
_COLLECTIVE_KINDS = ("collective_hang", "collective_desync")
# A hang is suspected from the status alone and confirmed when Flight Recorder dumps given beside
# it show a stuck collective: the same stall, whether or not dumps were given.
_HANG_ASSESSMENTS = ("suspected_hang", "confirmed_hang")


def _run_level_parts(incident: Mapping[str, Any]) -> list[Any]:
    """What names one of the run's own incidents and stays the same whenever the run is read."""
    kind = incident.get("kind")
    if kind in _STALL_KINDS:
        assessment = incident.get("assessment")
        return ["hang" if assessment in _HANG_ASSESSMENTS else assessment]
    if kind in _COLLECTIVE_KINDS:
        group = incident.get("group") if isinstance(incident.get("group"), Mapping) else {}
        return [incident.get("collective"), group.get("id"), group.get("desc"), group.get("size")]
    if kind == "traincheck_violation":
        return sorted({str(v.get("display_name")) for v in incident.get("violations") or () if isinstance(v, Mapping)})
    return []  # duplicate_batches, memory_imbalance: the ranks are each run's own, in `where`


def incident_identity(incident: Mapping[str, Any]) -> str:
    """What makes two runs' incidents "the same": the kind and what it names, without the step.

    It is the identity the all-ranks view groups one incident across ranks by
    (`evidence._incident_identity`), with the step taken out, since two runs meet the same
    fault at different steps. For the two MoE kinds it is cut down to the layer and router:
    which experts are overloaded or starved differs from run to run on the same router, and
    each run's own experts are listed in the entry.

    The run's own kinds (`RUN_LEVEL_KINDS`) carry readings of the moment they were read (times,
    seconds since progress, severity, steps, byte counts, sequence numbers), so their identity is
    what names the fault and stays the same whenever the run is read, and the ranks they name are
    each run's own, listed in the entry (`incident_where`), as which ranks show a per-rank
    incident is: a stall (`job_stalled`, `rank_silent`) by how it stopped, a hang (suspected or
    confirmed) or a lost connection; a stuck collective (`collective_hang`, `collective_desync`)
    by the collective and its process group; `traincheck_violation` by the invariants violated;
    `duplicate_batches` and `memory_imbalance` by the kind alone.
    """
    from .evidence import ATTN_KINDS, _incident_identity

    if incident.get("kind") in RUN_LEVEL_KINDS:
        return json.dumps([incident["kind"], _run_level_parts(incident)], sort_keys=True, default=str)
    kind, _, parts = json.loads(_incident_identity(incident))
    if kind == "moe_imbalance" and isinstance(parts, list):
        parts = parts[:2]
    elif kind == "starved_experts" and isinstance(parts, list):
        parts = sorted({json.dumps(part[:2]) for part in parts if isinstance(part, list)})
    elif kind in ATTN_KINDS and isinstance(parts, list) and len(parts) == 2:
        parts = parts[1]  # without the step the excursion was confirmed at
    elif isinstance(parts, dict):  # a kind this version does not know
        parts = {k: v for k, v in parts.items() if k not in ("step", "class")}
    return json.dumps([kind, parts], sort_keys=True, default=str)


def _name(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    for key in ("name", "module", "rank", "display_name"):
        if key in item:
            return item[key]
    if "expert" in item:
        return {key: item[key] for key in ("layer", "router", "expert") if key in item}
    return dict(item)


_WHERE_LISTS = ("overloaded", "dead", "parameters", "modules", "layers", "gradients", "experts", "ranks", "missing_ranks",
                "violations")


def incident_where(incident: Mapping[str, Any]) -> dict[str, Any]:
    """What an incident names: the MoE layer and router with its experts, and the parameters,
    modules, layers, gradients or experts. For the run's own kinds: the collective and its
    process group, how a stall stopped (`assessment`), the ranks, the ranks missing (never
    connected, or not in the stuck collective) and the invariants violated. At most eight
    names of each list, with the count beyond."""
    kind = incident.get("kind")
    where: dict[str, Any] = {}
    for key in ("layer", "router"):
        if incident.get(key) is not None:
            where[key] = incident[key]
    if kind in _COLLECTIVE_KINDS:
        group = incident.get("group") if isinstance(incident.get("group"), Mapping) else {}
        for key, value in (("collective", incident.get("collective")), ("group", group.get("desc") or group.get("id"))):
            if value is not None:
                where[key] = value
    if kind in _STALL_KINDS and incident.get("assessment") is not None:
        where["assessment"] = incident["assessment"]
    for key in _WHERE_LISTS:
        value = incident.get(key)
        names = list(value) if isinstance(value, Mapping) else [_name(i) for i in value] if isinstance(value, list) else None
        if names is None or not (names or key in ("overloaded", "dead")):
            continue
        where[key] = names[:_WHERE_NAMES]
        if len(names) > _WHERE_NAMES:
            where[f"{key}_count"] = len(names)
    return where


def compare_incidents(opened: list[Mapping[str, Any]], labels: list[str]) -> dict[str, Any]:
    from .evidence import INCIDENT_CLASSES, incident_class

    groups: dict[str, dict[str, Any]] = {}
    for label, run in zip(labels, opened):
        for incident, ranks in _shown_incidents(run["evidence"]):
            group = groups.setdefault(incident_identity(incident), {"kind": incident["kind"], "runs": {}})
            step = incident.get("step") if _is_step(incident.get("step")) else None
            shown = group["runs"].get(label)
            if shown is None:
                shown = group["runs"][label] = {"first_step": step, "count": 1, "where": incident_where(incident), "ranks": set()}
            else:
                shown["count"] += 1
                if step is not None and (shown["first_step"] is None or step < shown["first_step"]):
                    shown.update(first_step=step, where=incident_where(incident))
            if ranks is not None:
                shown["ranks"].update(ranks)
    order = {kind: index for index, kind in enumerate(INCIDENT_CLASSES)}
    entries: list[dict[str, Any]] = []
    for group in groups.values():
        shown = group["runs"]
        first_steps = [s["first_step"] for s in shown.values() if s["first_step"] is not None]
        absent: dict[str, str] = {}
        for label, run in zip(labels, opened):
            if label in shown:
                continue
            looked = _looked(run["evidence"], group["kind"])
            # A run that looked and stopped before the step the others first show it at never
            # reached the place where it could have shown it.
            if looked == "checked" and first_steps and run.get("last_step") is not None and run["last_step"] < min(first_steps):
                looked = "ended_before_step"
            absent[label] = looked
        own_class = incident_class(group["kind"])
        entries.append({
            "kind": group["kind"],
            "class": own_class or "unclassified",
            "in": [label for label in labels if label in shown],
            "first_step": {label: shown[label]["first_step"] for label in labels if label in shown},
            "where": {label: shown[label]["where"] for label in labels if label in shown},
            **({"count": {label: s["count"] for label, s in shown.items()}} if any(s["count"] > 1 for s in shown.values()) else {}),
            # For a run read on every rank: the ranks that show it there.
            **({"ranks": {label: sorted(shown[label]["ranks"]) for label in labels if label in shown and shown[label]["ranks"]}}
               if any(s["ranks"] for s in shown.values()) else {}),
            # For each run that does not show it: `checked` (it looked, with enough history, and
            # found none), or why its silence is not a "none": `not_enough_history`,
            # `not_configured`, `unavailable`, `not_said`, `ended_before_step`.
            "absent_from": absent,
        })
    entries.sort(key=lambda e: (min((s for s in e["first_step"].values() if s is not None), default=float("inf")),
                                order.get(e["kind"], len(order)), json.dumps(e["where"], sort_keys=True, default=str)))
    only = {label: [i for i, e in enumerate(entries) if e["in"] == [label]] for label in labels}
    # The ranks of each run whose incidents are not here: the others, when one rank was read
    # (`rank=`), or those with no current report or evidence, when every rank was.
    other: dict[str, list[int]] = {}
    for label, run in zip(labels, opened):
        read = run["read_from"]
        if read.get("all_ranks"):
            if read.get("unknown_ranks"):
                other[label] = list(read["unknown_ranks"])
        elif (read.get("ranks") and len(read["ranks"]) > 1) or read.get("unknown_ranks"):
            # One rank read (`rank=`): the others, and that rank itself when it could not be judged.
            other[label] = sorted({r for r in read.get("ranks") or () if r != read["rank"]} | set(read.get("unknown_ranks") or ()))
    return {
        "entries": entries,
        "only_in": only,
        "in_all": [i for i, e in enumerate(entries) if len(e["in"]) == len(labels)],
        "counts": {label: sum(1 for e in entries if label in e["in"]) for label in labels},
        **({"other_ranks_not_compared": other} if other else {}),
    }


def _shown_incidents(evidence: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], list[int] | None]]:
    """A run's incidents, each with the ranks that show it: from the incident groups for a run
    read on every rank (the run's own incidents, such as a stall, have no ranks of their own
    here), and with None for one rank's evidence."""
    found: list[tuple[Mapping[str, Any], list[int] | None]] = []
    if "incident_groups" in evidence:
        for group in evidence.get("incident_groups") or ():
            if isinstance(group, Mapping) and isinstance(group.get("incident"), Mapping) and "kind" in group["incident"]:
                found.append((group["incident"], [r for r in group.get("ranks") or () if _is_step(r)]))
        found += [(i, None) for i in evidence.get("run_incidents") or () if isinstance(i, Mapping) and "kind" in i]
        return found
    return [(i, None) for i in evidence.get("incidents") or () if isinstance(i, Mapping) and "kind" in i]


# --- Part 4: where two runs part -------------------------------------------------------------


def where_runs_part(opened: list[Mapping[str, Any]], labels: list[str], *, read_parts: bool = True) -> dict[str, Any]:
    """Part 4: the first step, and the layer or parameter, where a series leaves two runs' own
    early agreement and stays out (`divergence.first_divergence`, which reads both runs' step
    records). The block always carries a `state` and a `detail`, which `summary_text` prints.

    `state` is the rule's own answer (`diverged`, `no_divergence`, `no_answer`), or says why it
    was not asked: `not_asked` (it is for two runs only), `no_records` (a run was given as an
    evidence file, or its records could not be read), `not_read` (`read_parts=False` and a run
    is in a server's store, whose parts are not opened then) or `ranks_differ`. It is for paired
    runs, so `no_answer` is common and correct, and no step is ever given without `diverged`.
    Of the rule's per-series table only the series that do not agree are kept: on a 700-series
    model the whole table is half a megabyte. A `.jsonl` file the job is still writing is read on
    its whole lines, as parts 2 and 3 read it; `unfinished_lines_left_out` names the runs (and
    ranks) where a line was left out, and `detail` ends by saying so.
    """
    if len(opened) != 2:
        return {"state": "not_asked", "detail": f"where runs part is answered for two runs only, and {len(opened)} were given"}
    without = [label for run, label in zip(opened, labels) if not run.get("records")]
    if without:
        return {"state": "no_records", "detail": f"where the two runs part needs both runs' step records, and {' and '.join(without)} "
                f"{'was' if len(without) == 1 else 'were'} given as an evidence file"}
    stored = [label for run, label in zip(opened, labels) if (run.get("read_from") or {}).get("kind") == "server store"]
    if stored and not read_parts:
        return {"state": "not_read", "detail": f"where the two runs part reads every step record, and the stored parts of "
                f"{' and '.join(stored)} are not opened here: use the command line"}
    if any(run["records"].get("rank") is None for run in opened):
        return _where_ranks_part(opened, labels)
    ranks = [run["records"]["rank"] for run in opened]
    if ranks[0] != ranks[1]:
        return {"state": "ranks_differ", "detail": f"where the two runs part compares one rank of each run, and rank {ranks[0]} of "
                f"{labels[0]} and rank {ranks[1]} of {labels[1]} were read"}
    from .divergence import divergence_summary, first_divergence

    try:
        read = [_step_records(run["records"]["path"], run["records"]["rank"]) for run in opened]
    except (OSError, ValueError, ImportError) as error:  # ImportError: a store's parts need pyarrow
        return {"state": "no_records", "detail": f"the step records could not be read for where the runs part ({error})"}
    left_out = {label: [rank] for label, rank, (_, torn) in zip(labels, ranks, read) if torn}
    found = first_divergence(read[0][0], read[1][0], rank=ranks[0])
    lines = divergence_summary(found)
    series = found.get("series") or {}
    kept = {name: entry for name, entry in series.items() if entry.get("status") != "agrees"}
    return {
        "state": found["answer"], "detail": found["reason"] + _left_out_text(left_out, of_ranks=False), "lines": lines,
        "labels": {"a": labels[0], "b": labels[1]},
        **{key: value for key, value in found.items() if key not in ("answer", "reason", "series")},
        "series": kept, "series_agreeing": len(series) - len(kept),
        **({"unfinished_lines_left_out": left_out} if left_out else {}),
    }


def _step_records(path: str | Path, rank: int) -> tuple[list[dict[str, Any]], bool]:
    """One rank's step records for part 4, as `load_records(path, rank=rank)` reads them, and
    whether a line that is not whole was left out: a `.jsonl` file the job is still writing (or
    one with a line torn by a killed process) is read on its whole lines, as `_open_local` reads
    it, with the attempts resolved as `load_records` resolves them."""
    from .evidence import _Tail, load_records, resolve_attempts

    try:
        return load_records(path, rank=rank), False
    except ValueError:
        source = Path(path)
        if source.is_dir() or source.suffix == ".parquet":
            raise
        tail = _Tail(source, rank)
        tail.poll()
        return resolve_attempts(tail.records), True


def _left_out_text(left_out: Mapping[str, list[int]], *, of_ranks: bool) -> str:
    """The sentence part 4's `detail` ends with when a line that is not whole was left out."""
    if not left_out:
        return ""
    whose = " and ".join(f"{label}'s records" + (f" ({_rank_text(ranks)})" if of_ranks else "")
                         for label, ranks in left_out.items())
    return f" A line that is not whole was left out of {whose}: the run{'s' if len(left_out) > 1 else ''} may still be going."


def _where_ranks_part(opened: list[Mapping[str, Any]], labels: list[str]) -> dict[str, Any]:
    """Part 4 for runs read on every rank: rank r of one run against rank r of the other, for
    each rank, by the same rule as for one rank. Nothing is made across ranks. Runs whose
    ranks differ are not compared: a rank's share of the data depends on how many there are.

    The answer is the earliest divergence over the ranks (the lowest rank on a tie), with the
    keys one rank's answer has, read from that rank; `no_divergence` only when every rank
    says so; else `no_answer`, naming the ranks without one. `by_rank` has each rank's answer
    in brief: a rank without an answer carries its reason when it is not the chosen rank's.
    """
    from .divergence import divergence_summary, first_divergence

    # Each run's step records by rank, and its ranks: all of them, whether or not a rank has
    # records yet, so that a run of two ranks is never taken for a run of one.
    paths: list[dict[int, str]] = []
    sets: list[list[int]] = []
    for run in opened:
        records = run["records"]
        own = dict(records["ranks"]) if records.get("rank") is None else {records["rank"]: records["path"]}
        paths.append(own)
        listed = run["read_from"].get("ranks") if records.get("rank") is None else None
        sets.append(sorted(set(listed or ()) | set(own)))
    if sets[0] != sets[1]:
        return {"state": "ranks_differ", "ranks": {labels[0]: sets[0], labels[1]: sets[1]},
                "detail": f"where the two runs part compares each rank with the same rank of the other run, and {labels[0]} "
                          f"has {_rank_text(sets[0])} where {labels[1]} has {_rank_text(sets[1])}: with other ranks a rank reads another share of the data, so no rank is "
                          "compared"}
    without = {rank: [label for label, own in zip(labels, paths) if rank not in own] for rank in sets[0]}
    without = {rank: missing for rank, missing in without.items() if missing}
    compared = [rank for rank in sets[0] if rank not in without]
    if not compared:
        return {"state": "no_records", "ranks_without_records": without,
                "detail": "where the two runs part needs both runs' step records of a rank, and no rank has them in both"}
    answers: dict[int, dict[str, Any]] = {}
    summaries: dict[int, str] = {}
    torn_in: dict[str, list[int]] = {}
    for rank in compared:
        try:
            read = [_step_records(own[rank], rank) for own in paths]
        except (OSError, ValueError, ImportError) as error:  # ImportError: a store's parts need pyarrow
            return {"state": "no_records", "detail": f"the step records of rank {rank} could not be read for where the runs "
                                                     f"part ({error})"}
        for label, (_, torn) in zip(labels, read):
            if torn:
                torn_in.setdefault(label, []).append(rank)
        pair = [records for records, _ in read]
        del read
        found = first_divergence(pair[0], pair[1], rank=rank)
        summaries[rank] = divergence_summary(found)
        series = found.get("series") or {}
        answers[rank] = {**found, "series": {name: e for name, e in series.items() if e.get("status") != "agrees"},
                         "series_agreeing": sum(1 for e in series.values() if e.get("status") == "agrees")}
        del pair
    diverged = [r for r, found in answers.items() if found["answer"] == "diverged"]
    unanswered = [r for r, found in answers.items() if found["answer"] == "no_answer"]

    def step(rank: int) -> float:
        value = (answers[rank].get("divergence") or {}).get("step")
        return value if _is_step(value) else float("inf")

    if diverged:
        picked = min(diverged, key=lambda r: (step(r), r))
        state = "diverged"
        detail = (f"on rank {picked}, earliest of the {len(diverged)} of {len(answers)} ranks that part: "
                  f"{answers[picked]['reason']}")
    elif unanswered:
        picked = unanswered[0]
        state = "no_answer"
        detail = (f"no answer on {_rank_text(unanswered)} of the {len(answers)} compared"
                  + (f", and no divergence on the others" if len(unanswered) < len(answers) else "")
                  + f"; rank {picked}: {answers[picked]['reason']}")
    else:
        picked = min(answers)
        state = "no_divergence"
        detail = f"no divergence on any of the {len(answers)} ranks compared; rank {picked}: {answers[picked]['reason']}"
    if without:
        detail += (f" Not compared: {_rank_text(sorted(without))}, with no step records yet in "
                   + "; ".join(f"rank {rank}: {' and '.join(missing)}" for rank, missing in sorted(without.items())[:8]) + ".")
    left_out = {label: torn_in[label] for label in labels if label in torn_in}
    detail += _left_out_text(left_out, of_ranks=True)
    chosen = answers[picked]
    brief = []
    for rank, found in answers.items():
        item: dict[str, Any] = {"rank": rank, "answer": found["answer"]}
        if found["answer"] == "diverged":
            item.update(step=(found.get("divergence") or {}).get("step"), series=(found.get("divergence") or {}).get("series"))
        elif found["answer"] == "no_answer" and (rank == picked or found["reason"] != chosen["reason"]):
            item["reason"] = found["reason"]  # the same paragraph as the chosen rank's is given once
        brief.append(item)
    return {
        "state": state, "detail": detail,
        "lines": f"On rank {picked}: {summaries[picked]}",
        "labels": {"a": labels[0], "b": labels[1]},
        **{key: value for key, value in chosen.items() if key not in ("answer", "reason")},
        "ranks_compared": sorted(answers),
        **({"ranks_without_records": without} if without else {}),
        "by_rank": brief,
        **({"unfinished_lines_left_out": left_out} if left_out else {}),
    }


# --- The whole comparison ---------------------------------------------------------------------


def compare_runs(*runs: str | Path, source: str | Path | None = None, project: str | None = None,
                 rank: int | None = None, paths: bool = True, read_parts: bool = True) -> dict[str, Any]:
    """Compare two or more runs and return the result (see the module's text). Nothing is written.

    Each of `runs` is a run ID or a name at `source` (default `tensorquill-out`), or a path.
    `project` narrows a name to one project. `rank` is the one rank read of every run
    (default: every rank of each run; a run of one rank is read on the rank it has). `paths=False` takes IDs and names only. `read_parts=False`
    refuses a run in a server's store whose report is not current; otherwise its parts are read
    and judged here.
    Raises `runs.RunLookupError` (a `ValueError`) for a run that is not found, a name that fits
    several runs, or fewer than two runs.
    """
    if len(runs) < 2:
        raise _runs.RunLookupError("compare needs at least two runs")
    opened = [open_run(wanted, source, project, rank, paths=paths, read_parts=read_parts) for wanted in runs]
    seen: dict[tuple[str, str], int] = {}
    for index, run in enumerate(opened):
        key = (str(run["run_id"]), run["read_from"]["path"])
        if key in seen:
            raise _runs.RunLookupError(
                f"{runs[seen[key]]!r} and {runs[index]!r} are the same run ({run['run_id']}): nothing to compare")
        seen[key] = index
    labels = _labels(len(opened))
    result: dict[str, Any] = {
        "view": COMPARE_VIEW,
        "view_version": COMPARE_VIEW_VERSION,
        "written_at_ms": int(time.time() * 1000),
        "runs": [{
            "label": label, "asked_for": str(wanted), **run["block"],
            "has_manifest": isinstance(run.get("manifest"), Mapping),
            "run_state": (run["evidence"].get("run_state") or {}).get("state"),
            "last_step": run["last_step"],
            "incidents": len([i for i in run["evidence"].get("incidents") or () if isinstance(i, Mapping)]),
            "rank_count": len(run["read_from"]["ranks"]) if run["read_from"].get("ranks") else None,
            "read_from": run["read_from"],
        } for label, wanted, run in zip(labels, runs, opened)],
        # The order of the four parts is part of the format: what was checked is read before
        # the incidents.
        "setup": compare_setup(opened, labels),
        "checked": compare_checked(opened, labels),
        "incidents": compare_incidents(opened, labels),
        "divergence": where_runs_part(opened, labels, read_parts=read_parts),
    }
    result["summary"] = summary_text(result)
    return result


# --- The printed summary ----------------------------------------------------------------------

_LOOKED_TEXT = {
    "checked": "checked",
    "not_enough_history": "checked, not yet with enough history",
    "not_configured": "not configured",
    "unavailable": "unavailable",
    "not_said": "not said",
    "history_not_stated": "checked, without saying whether it had enough history on every rank",
}
def _rank_text(ranks: list[int]) -> str:
    from .evidence import _rank_ranges

    return f"rank{'s' if len(ranks) != 1 else ''} {_rank_ranges(ranks, 8)}"


_ABSENT_TEXT = {
    "checked": "{run} checked this kind and found none",
    "not_enough_history": "{run} checked this kind without enough history yet, so its silence is not a \"none\"",
    "not_configured": "{run} did not look: the kind is not configured there",
    "unavailable": "{run} did not look: the kind was unavailable there",
    "not_said": "{run}'s evidence does not say whether it looked",
    "ended_before_step": "{run} ended before this step",
    "history_not_stated": "{run} checked this kind on every rank, and some rank does not say whether it had enough "
                          "history, so its silence is not a \"none\"",
}


def _value(value: Any) -> str:
    return "not set" if value is None else json.dumps(value, default=str) if not isinstance(value, str) else repr(value)


def _where_text(where: Mapping[str, Any]) -> str:
    parts = []
    if "layer" in where or "router" in where:
        parts.append(" ".join(str(x) for x in (f"layer {where['layer']}" if "layer" in where else None,
                                               f"router {where['router']}" if "router" in where else None) if x))
    for key in ("collective", "group", "assessment"):
        if where.get(key) is not None:
            parts.append(f"{key} {where[key]}")
    for key in _WHERE_LISTS:
        if where.get(key):
            more = where.get(f"{key}_count")
            names = ", ".join(str(n) if not isinstance(n, Mapping) else json.dumps(n, sort_keys=True) for n in where[key])
            parts.append(f"{key.replace('_', ' ')} {names}" + (f" and {more - len(where[key])} more" if more else ""))
    return "; ".join(parts)


def summary_text(result: Mapping[str, Any]) -> str:
    runs = result["runs"]
    labels = [run["label"] for run in runs]
    lines = [f"Comparing {len(runs)} runs"]
    for run in runs:
        read = run["read_from"]
        if read.get("all_ranks"):
            judged, unknown = read.get("judged_ranks") or [], read.get("unknown_ranks") or []
            rank = (f"all {len(read['ranks'])} ranks" if not unknown
                    else f"{len(judged)} of {len(read['ranks'])} ranks ({_rank_text(unknown)} not judged)")
            of = ""
        else:
            rank = f"rank {read['rank']}" if read.get("rank") is not None else "one rank"
            of = f" of {len(read['ranks'])}" if read.get("ranks") and len(read["ranks"]) > 1 else ""
            if read.get("unknown_ranks"):  # the one rank read could not be judged
                of += f" (not judged: {(read.get('unknown_why') or {}).get(read['rank'], 'no current report')})"
        lines.append(
            f"  {run['label']}  {run['name']} (run ID {run['run_id']}, project {run['project']}): "
            f"{run['run_state'] or 'state unknown'}, "
            + (f"last step {run['last_step']}, " if run["last_step"] is not None else "")
            + f"{run['incidents']} incident{'s' if run['incidents'] != 1 else ''}; {rank}{of} from {read['path']}"
            + (" (a line that is not whole was left out: the run may still be going)" if read.get("unfinished_lines_left_out") else "")
        )

    setup = result["setup"]
    lines.append(f"1. Setup: {len(setup['differs'])} difference{'s' if len(setup['differs']) != 1 else ''}")
    for item in setup["differs"]:
        lines.append(f"   {item['field']}: " + ", ".join(f"{label} {_value(value)}" for label, value in item["values"].items()))
    if setup.get("config_not_known"):
        lines.append(f"   the config of {', '.join(setup['config_not_known'])} was too large to send to its server and is "
                     "not in the store: left out of the differences (the run's local manifest has it)")
    if setup["no_manifest"]:
        lines.append(f"   no run manifest for {', '.join(setup['no_manifest'])}: "
                     "its setup is not known and is left out of the differences")

    checked = result["checked"]
    whole = checked.get("all_ranks_read") or {}
    ranks = ", ".join(f"{label} rank {rank}" if rank is not None else
                      f"{label} {_rank_text(whole[label])}" if label in whole else f"{label} one rank"
                      for label, rank in checked["ranks_read"].items())
    lines.append(f"2. What was checked: {len(checked['differs'])} kind{'s' if len(checked['differs']) != 1 else ''} differ"
                 f"{'s' if len(checked['differs']) == 1 else ''} ({ranks})")
    for item in checked["differs"]:
        states = []
        for label, state in item["looked"].items():
            text = _LOOKED_TEXT.get(state, state)
            records, needed = (item.get("records") or {}).get(label), (item.get("records_needed") or {}).get(label)
            if state in ("checked", "not_enough_history") and records is not None and needed is not None:
                text += f" ({records} of the {needed} records its slowest rule needs)" if state == "not_enough_history" else f" ({records} records)"
            within = (item.get("ranks") or {}).get(label)
            if within:
                text += " (" + ", ".join(f"{_LOOKED_TEXT.get(value, value)} on {_rank_text(ranks)}" for value, ranks in within.items()) + ")"
            states.append(f"{text} in {label}")
        lines.append(f"   {item['kind']}: " + "; ".join(states))
    if checked["not_said_by"]:
        lines.append(f"   the evidence of {', '.join(checked['not_said_by'])} does not say what was checked")
        for label, why in (checked.get("not_said_why") or {}).items():
            if why:
                lines.append(f"      {label}: {why}")

    incidents = result["incidents"]
    entries = incidents["entries"]
    counts = [f"{len(incidents['only_in'][label])} in {label} only" for label in labels]
    some = len(entries) - sum(len(v) for v in incidents["only_in"].values()) - len(incidents["in_all"])
    counts.append(f"{len(incidents['in_all'])} in {'both' if len(labels) == 2 else 'all'}")
    if some:
        counts.append(f"{some} in some")
    lines.append("3. Incidents: " + ", ".join(counts))
    for entry in entries:
        shown = entry["in"]
        if len(shown) == len(labels):
            head = "both" if len(labels) == 2 else "all"
        else:
            head = (", ".join(shown) + " only") if len(shown) == 1 else ", ".join(shown)
        on = entry.get("ranks") or {}
        steps = ", ".join((f"{label} step {step}" if step is not None else f"{label} (no step)")
                          + (f" (on {_rank_text(on[label])})" if label in on else "")
                          for label, step in entry["first_step"].items())
        first = entry["where"][shown[0]]
        where = _where_text(first)
        lines.append(f"   {head}: {entry['kind']} ({entry['class']}), first at {steps}" + (f": {where}" if where else ""))
        for label in shown[1:]:
            if entry["where"][label] != first:
                lines.append(f"      in {label}: {_where_text(entry['where'][label])}")
        for label, why in entry["absent_from"].items():
            lines.append("      " + _ABSENT_TEXT.get(why, "{run}: " + why).format(run=label))
    other = incidents.get("other_ranks_not_compared")
    if other:
        lines.append("   incidents on other ranks were not compared: " + "; ".join(
            f"{label} ranks {', '.join(map(str, ranks))}" for label, ranks in other.items())
            + (" (not judged: no current report or evidence)" if any(
                run["read_from"].get("all_ranks") or run["read_from"].get("unknown_ranks")
                for run in runs if run["label"] in other) else ""))
    lines.append(f"4. Where the runs part: {result['divergence']['detail']}")
    return "\n".join(lines)


# --- The command line -------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tensorquill compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="RUN", help="a run ID or name at --source, or a path "
                        "(a .jsonl capture, a run folder of a server's store, or one rank's evidence.json)")
    parser.add_argument("--source", type=Path, default=Path(_runs.DEFAULT_SOURCE),
                        help=f"where IDs and names are looked up: a folder of local files or a server's store "
                        f"(default: {_runs.DEFAULT_SOURCE})")
    parser.add_argument("--project", help="look a name up in this project only")
    parser.add_argument("--rank", type=int, help="read this one rank of every run (default: every rank)")
    parser.add_argument("-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT),
                        help=f"where the comparison is written (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)
    try:
        result = compare_runs(*args.runs, source=args.source, project=args.project, rank=args.rank)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    from .evidence import write_evidence

    written = write_evidence(result, args.output)  # by a temporary file and a rename
    print(result["summary"])
    print(f"wrote comparison -> {written}")
