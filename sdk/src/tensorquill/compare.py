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
   the modules), with the runs that show it and each run's first step. For a run that does
   not show it, the entry says whether that run looked.
4. `divergence`: where two runs part. Not built in this version (`where_runs_part`).

One rank of each run is read: rank 0 unless `rank=` names another, and the result says which.
Incidents on a run's other ranks are not compared, and it says that too. Local files are
judged here by the Python detectors with the default thresholds; a run in a server's store is
read from the rank's `incidents.json` when that report is current (no part is opened), and
from its parts otherwise.

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


def _open_local(run: Mapping[str, Any], rank: int | None) -> dict[str, Any]:
    from .evidence import _Tail, build_evidence, load_records

    files: Mapping[int, Path] = run.get("files") or {}
    if not files:
        raise _runs.RunLookupError(f"run {run['run_id']} has a manifest and no records yet: nothing to compare")
    path = files.get(rank) if rank is not None and rank in files else files[min(files)]
    unfinished = False
    try:
        records = load_records(path, all_attempts=True)
    except ValueError:
        # A line the job is still writing, or one torn by a killed process: the whole lines are
        # read, as the watcher reads them, and the result says so.
        tail = _Tail(Path(path), None)
        tail.poll()
        records, unfinished = tail.records, True
    held = sorted({int(r.get("rank", 0)) for r in records})
    ranks = sorted(set(files) | set(held))
    # The rank asked for; a run with one rank is read on the rank it has, and says which.
    picked = rank if rank in held else held[0] if held and (rank is None or len(ranks) == 1) else None
    if picked is None:
        raise _runs.RunLookupError(f"run {run['run_id']} has no rank {rank} here: it has rank{'s' if len(ranks) != 1 else ''} "
                                   f"{', '.join(map(str, ranks))}")
    records = [r for r in records if int(r.get("rank", 0)) == picked]
    evidence = build_evidence(records, run_manifest=run.get("manifest") or {})
    steps = [r["step"] for r in records if _is_step(r.get("step"))]
    return {
        "evidence": evidence, "last_step": max(steps) if steps else None,
        "read_from": {"path": str(path), "kind": "local records", "rank": picked, "ranks": ranks,
                      "judged_by": "the Python detectors, here, with the default thresholds",
                      **({"unfinished_lines_left_out": True} if unfinished else {})},
        "records": {"path": str(path), "rank": picked},
    }


def _open_store(run: Mapping[str, Any], rank: int | None) -> dict[str, Any]:
    from .evidence import _discover_rank_folders, _report_evidence, build_evidence, load_records, read_server_report, read_watchdog

    folder = Path(run["path"])
    folders = _discover_rank_folders(folder)
    if not folders:
        raise _runs.RunLookupError(f"run {run['run_id']} at {folder} has no rank=<n> folder yet: nothing to compare")
    picked = rank if rank in folders else min(folders) if rank is None or len(folders) == 1 else None
    if picked is None:
        raise _runs.RunLookupError(f"run {run['run_id']} has no rank {rank}: its store holds ranks "
                                   f"{', '.join(map(str, sorted(folders)))}")
    look = read_server_report(folders[picked])
    if look["state"] == "current":
        evidence = _report_evidence(folders[picked], look, read_watchdog(folder / "watchdog.json"))
        last = look["report"].get("last_step")
        judged = "the server's report (incidents.json), with the thresholds the server applied; no part was read"
    else:
        # The report cannot be used (see `read_server_report`), so the parts are read and judged here.
        records = load_records(folder, rank=picked, all_attempts=True)
        evidence = build_evidence(records, watchdog=folder / "watchdog.json", run_manifest=run.get("manifest") or {})
        steps = [r["step"] for r in records if _is_step(r.get("step"))]
        last = max(steps) if steps else None
        judged = (f"the Python detectors, here, on the stored parts with the default thresholds: the server's "
                  f"report is {look['state']} ({look['reason']})")
    return {
        "evidence": evidence, "last_step": last if _is_step(last) else None,
        "read_from": {"path": str(folder), "kind": "server store", "rank": picked, "ranks": sorted(folders),
                      "judged_by": judged},
        "records": {"path": str(folder), "rank": picked},
    }


def _open_evidence_file(path: Path, rank: int | None) -> dict[str, Any]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _runs.RunLookupError(f"{path} cannot be read as an evidence file ({error})") from None
    if not isinstance(evidence, dict) or "incidents" not in evidence:
        what = "an all-ranks view, which lists no kind_checks: give the run folder" if isinstance(evidence, dict) and evidence.get("view") else "not one rank's evidence"
        raise _runs.RunLookupError(f"{path} is {what}")
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


def open_run(wanted: str | Path, source: str | Path | None = None, project: str | None = None,
             rank: int | None = None, *, paths: bool = True) -> dict[str, Any]:
    """One run, ready to compare: its manifest, one rank's evidence and where both were read.

    `wanted` is a run ID or name at `source`, or (with `paths`) a path. The lookup at `source`
    comes first, so an ID never changes meaning because a file of that name appears.
    """
    source = Path(source) if source is not None else Path(_runs.DEFAULT_SOURCE)
    run: dict[str, Any] | None = None
    lookup_error: Exception | None = None
    if isinstance(wanted, str):
        try:
            run = _runs.find_run(source, wanted, project)
        except FileNotFoundError as error:
            lookup_error = _runs.RunLookupError(str(error))
        except _runs.RunLookupError as error:
            if "fits" in str(error):  # several runs of that name: never guessed around
                raise
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
    opened = _open_local(run, rank) if run["source"] == "local" else _open_store(run, rank)
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
        configs = [m.get("config") if isinstance(m.get("config"), Mapping) else {} for _, m in known]
        keys = sorted({str(key) for config in configs for key in config})
        for key in keys:
            values = [config.get(key) for config in configs]
            missing = [label for (label, _), config in zip(known, configs) if key not in config]
            if missing or _differ(values):
                differs.append({
                    "field": f"config.{key}",
                    "values": {label: value for (label, _), value in zip(known, values)},
                    **({"not_set_in": missing} if missing else {}),
                })
        same_config = [key for key in keys if not any(d["field"] == f"config.{key}" for d in differs)]
    else:
        same_config = []
    return {
        "differs": differs,
        "same": same,
        "config_keys_same": same_config,
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
    or `not_said` for evidence without `kind_checks` or a kind it does not list."""
    check = _check_of(evidence, kind) if isinstance(kind, str) else None
    if check is None:
        return "not_said"
    value = check.get("value")
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
        differs.append(entry)
    return {
        "ranks_read": {label: run["read_from"].get("rank") for label, run in zip(labels, opened)},
        "differs": differs,
        "same": same if len(said) > 1 else 0,
        "not_said_by": not_said,
    }


# --- Part 3: the incidents ------------------------------------------------------------------


def incident_identity(incident: Mapping[str, Any]) -> str:
    """What makes two runs' incidents "the same": the kind and what it names, without the step.

    It is the identity the all-ranks view groups one incident across ranks by
    (`evidence._incident_identity`), with the step taken out, since two runs meet the same
    fault at different steps. For the two MoE kinds it is cut down to the layer and router:
    which experts are overloaded or starved differs from run to run on the same router, and
    each run's own experts are listed in the entry.
    """
    from .evidence import ATTN_KINDS, _incident_identity

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
    for key in ("name", "module", "rank"):
        if key in item:
            return item[key]
    if "expert" in item:
        return {key: item[key] for key in ("layer", "router", "expert") if key in item}
    return dict(item)


def incident_where(incident: Mapping[str, Any]) -> dict[str, Any]:
    """What an incident names: the MoE layer and router with its experts, and the parameters,
    modules, layers, gradients or experts. At most eight names of each, with the count beyond."""
    where: dict[str, Any] = {}
    for key in ("layer", "router"):
        if incident.get(key) is not None:
            where[key] = incident[key]
    for key in ("overloaded", "dead", "parameters", "modules", "layers", "gradients", "experts"):
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
        for incident in run["evidence"].get("incidents") or ():
            if not isinstance(incident, Mapping) or "kind" not in incident:
                continue
            group = groups.setdefault(incident_identity(incident), {"kind": incident["kind"], "runs": {}})
            step = incident.get("step") if _is_step(incident.get("step")) else None
            shown = group["runs"].get(label)
            if shown is None:
                group["runs"][label] = {"first_step": step, "count": 1, "where": incident_where(incident)}
            else:
                shown["count"] += 1
                if step is not None and (shown["first_step"] is None or step < shown["first_step"]):
                    shown.update(first_step=step, where=incident_where(incident))
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
            # For each run that does not show it: `checked` (it looked, with enough history, and
            # found none), or why its silence is not a "none": `not_enough_history`,
            # `not_configured`, `unavailable`, `not_said`, `ended_before_step`.
            "absent_from": absent,
        })
    entries.sort(key=lambda e: (min((s for s in e["first_step"].values() if s is not None), default=float("inf")),
                                order.get(e["kind"], len(order)), json.dumps(e["where"], sort_keys=True, default=str)))
    only = {label: [i for i, e in enumerate(entries) if e["in"] == [label]] for label in labels}
    multi = {label: run["read_from"]["ranks"] for label, run in zip(labels, opened)
             if run["read_from"].get("ranks") and len(run["read_from"]["ranks"]) > 1}
    return {
        "entries": entries,
        "only_in": only,
        "in_all": [i for i, e in enumerate(entries) if len(e["in"]) == len(labels)],
        "counts": {label: sum(1 for e in entries if label in e["in"]) for label in labels},
        # One rank of each run was read. A run of several ranks may show more on the others.
        **({"other_ranks_not_compared": {
            label: [r for r in ranks if r != opened[labels.index(label)]["read_from"]["rank"]] for label, ranks in multi.items()
        }} if multi else {}),
    }


# --- Part 4: where two runs part -------------------------------------------------------------


def where_runs_part(opened: list[Mapping[str, Any]], labels: list[str]) -> dict[str, Any]:
    """PART 4, NOT BUILT HERE. The first step, and the layer or parameter, where a series
    leaves two runs' own early agreement and stays out.

    This is the one place the rule is joined to `compare`: it is written in its own module
    (`divergence.py`, plan row C3), which reads two runs' records. Each opened run carries
    `records` (`{"path", "rank"}`: what `evidence.load_records(path, rank=rank)` reads), or
    None for a run given as an evidence file. Whatever is returned here becomes the result's
    `divergence` block and must carry a `state` and a `detail`; `summary_text` prints the
    `detail`. Until the join the answer is `not_built`, never a step.
    """
    if len(opened) != 2:
        return {"state": "not_asked", "detail": f"where runs part is answered for two runs only, and {len(opened)} were given"}
    return {"state": "not_built", "detail": "where the two runs part is not built in this version"}


# --- The whole comparison ---------------------------------------------------------------------


def compare_runs(*runs: str | Path, source: str | Path | None = None, project: str | None = None,
                 rank: int | None = None, paths: bool = True) -> dict[str, Any]:
    """Compare two or more runs and return the result (see the module's text). Nothing is written.

    Each of `runs` is a run ID or a name at `source` (default `tensorquill-out`), or a path.
    `project` narrows a name to one project. `rank` is the rank read of every run (default:
    each run's lowest, rank 0). `paths=False` takes IDs and names only.
    Raises `runs.RunLookupError` (a `ValueError`) for a run that is not found, a name that fits
    several runs, or fewer than two runs.
    """
    if len(runs) < 2:
        raise _runs.RunLookupError("compare needs at least two runs")
    opened = [open_run(wanted, source, project, rank, paths=paths) for wanted in runs]
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
            "read_from": run["read_from"],
        } for label, wanted, run in zip(labels, runs, opened)],
        # The order of the four parts is part of the format: what was checked is read before
        # the incidents.
        "setup": compare_setup(opened, labels),
        "checked": compare_checked(opened, labels),
        "incidents": compare_incidents(opened, labels),
        "divergence": where_runs_part(opened, labels),
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
}
_ABSENT_TEXT = {
    "checked": "{run} checked this kind and found none",
    "not_enough_history": "{run} checked this kind without enough history yet, so its silence is not a \"none\"",
    "not_configured": "{run} did not look: the kind is not configured there",
    "unavailable": "{run} did not look: the kind was unavailable there",
    "not_said": "{run}'s evidence does not say whether it looked",
    "ended_before_step": "{run} ended before this step",
}


def _value(value: Any) -> str:
    return "not set" if value is None else json.dumps(value, default=str) if not isinstance(value, str) else repr(value)


def _where_text(where: Mapping[str, Any]) -> str:
    parts = []
    if "layer" in where or "router" in where:
        parts.append(" ".join(str(x) for x in (f"layer {where['layer']}" if "layer" in where else None,
                                               f"router {where['router']}" if "router" in where else None) if x))
    for key in ("overloaded", "dead", "parameters", "modules", "layers", "gradients", "experts"):
        if where.get(key):
            more = where.get(f"{key}_count")
            names = ", ".join(str(n) if not isinstance(n, Mapping) else json.dumps(n, sort_keys=True) for n in where[key])
            parts.append(f"{key} {names}" + (f" and {more - len(where[key])} more" if more else ""))
    return "; ".join(parts)


def summary_text(result: Mapping[str, Any]) -> str:
    runs = result["runs"]
    labels = [run["label"] for run in runs]
    lines = [f"Comparing {len(runs)} runs"]
    for run in runs:
        read = run["read_from"]
        rank = f"rank {read['rank']}" if read.get("rank") is not None else "one rank"
        of = f" of {len(read['ranks'])}" if read.get("ranks") and len(read["ranks"]) > 1 else ""
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
    if setup["no_manifest"]:
        lines.append(f"   no run manifest for {', '.join(setup['no_manifest'])}: "
                     "its setup is not known and is left out of the differences")

    checked = result["checked"]
    ranks = ", ".join(f"{label} rank {rank}" if rank is not None else f"{label} one rank" for label, rank in checked["ranks_read"].items())
    lines.append(f"2. What was checked: {len(checked['differs'])} kind{'s' if len(checked['differs']) != 1 else ''} differ"
                 f"{'s' if len(checked['differs']) == 1 else ''} ({ranks})")
    for item in checked["differs"]:
        states = []
        for label, state in item["looked"].items():
            text = _LOOKED_TEXT.get(state, state)
            records, needed = (item.get("records") or {}).get(label), (item.get("records_needed") or {}).get(label)
            if state in ("checked", "not_enough_history") and records is not None and needed is not None:
                text += f" ({records} of the {needed} records its slowest rule needs)" if state == "not_enough_history" else f" ({records} records)"
            states.append(f"{text} in {label}")
        lines.append(f"   {item['kind']}: " + "; ".join(states))
    if checked["not_said_by"]:
        lines.append(f"   the evidence of {', '.join(checked['not_said_by'])} does not say what was checked")

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
        steps = ", ".join(f"{label} step {step}" if step is not None else f"{label} (no step)" for label, step in entry["first_step"].items())
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
            f"{label} ranks {', '.join(map(str, ranks))}" for label, ranks in other.items()))
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
    parser.add_argument("--rank", type=int, help="the rank read of every run (default: each run's lowest)")
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
