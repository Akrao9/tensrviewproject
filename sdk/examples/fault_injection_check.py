"""Compare what TensorQuill reported against lm-resiliency's fault-injection ground truth.

Lane AI-3, Nov 22, 2026. The author's terms: "retain `injection.json` as independent ground truth and
compare TensorQuill's reported rank and step against it." The campaign is run with
`fault_injection_check_entry.py`, which leaves TensorQuill's captures in `<artifact-dir>/tensorquill/`
beside the campaign's `injection.json`. This script reads both and writes
`<artifact-dir>/tensorquill_comparison.json`. It never reads the campaign's `localization.json` or
`evaluation.json` (SCOUT's answers) to decide anything.

Ground truth, from `injection.json` (written by the campaign's rank 0; format in lm-resiliency's
`docs/fault_injection.md` and `FaultInjectionRecord.to_dict`): every record of `injections` with
`injection_succeeded` true is one injected action, at global rank `execution_rank` and training
iteration `iteration`, belonging to occurrence `occurrence_id`. Records skipped by probability
(`status: skipped_probability`) are left out, as the campaign's own comparison leaves them out.
Records not injected (`injection_succeeded` false: pending, failed, cancelled) are listed under
`not_injected` and scored neither way. An action is active from its iteration for its incident's
`lifetime.iterations` in the embedded manifest (1 when the lifetime is given in calls or as "until"
a boundary: the campaign allows `matching_calls=1` only, and calls its recovery and replacement
boundaries at the trigger iteration).

TensorQuill's reports, from the captures alone (fixed before any run, Nov 22, 2026, and not chosen
from results):

- every incident of every kind that the Python detectors list in each rank's evidence
  (`tensorquill.build_evidence` on that rank's file), with that rank, at its `step` and at each step
  of any list of steps it holds under a key ending in `_steps` (`incident_steps`);
- every `attributed` entry of the cross-rank replica comparison (`tensorquill.evidence.compare_replicas`
  over all ranks' records), once for each rank in its `outliers`, at its step;
- a `replica_drift` incident names a step and no rank: listed under `unattributed`, never credited.

TensorQuill's step N is the campaign's iteration N - start_step + 1 (`start_step` from
`tensorquill/attach.json`, 1 by default). A report matches an action only when its rank is the
action's `execution_rank` exactly and its iteration is within the action's active window; nothing is
credited with the rank alone or the step alone, and there is no slack of a step either way. An
occurrence is `found` when each of its actions is matched; `partial` when some are; else `missed`.
A report that matches no injected action is a `false_report` (one matching only a record that was
not injected is listed with that record, and is still false).

The pass rule is the author's, of Nov 22, 2026: the campaign passes when every occurrence is found
and there is no false report. Partials, misses and false reports are written to the result whatever
the verdict. Exit status: 0 on a pass, 1 otherwise, 2 bad arguments, 3 NOT PROVEN: missing ground
truth or an incomplete run, that is no `injection.json`, a record that does not hold the documented
fields, no `completed_iterations`, a record still `pending` when the campaign wrote its artifact, no
capture, captures for other ranks than the campaign's `world_size`, or a rank whose last captured
step is not the campaign's last iteration (the step-to-iteration mapping is checked, not assumed).
`--measure` changes the exit status only (0 whenever the comparison was made): the result's `status`
is the strict verdict either way. The result carries the campaign environment's exact PyTorch version
(`campaign_environment`, written by the entry script on rank 0 inside that environment).

    python sdk/examples/fault_injection_check.py --artifact-dir runs/fault-injection
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PASS, FAIL, USAGE, NOT_PROVEN = 0, 1, 2, 3


class GroundTruthError(ValueError):
    """`injection.json` does not hold what its documented format says."""


def _integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GroundTruthError(f"{what} must be an integer, not {value!r}")
    return value


def _lifetimes(injection: Mapping[str, Any]) -> dict[str, int]:
    """Each incident's active iterations, from the manifest embedded in `injection.json`."""
    manifest = injection.get("manifest")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("incidents"), list):
        raise GroundTruthError("injection.json has no manifest with incidents")
    out: dict[str, int] = {}
    for incident in manifest["incidents"]:
        lifetime = incident.get("lifetime") or {}
        iterations = lifetime.get("iterations")
        out[str(incident.get("incident_id"))] = 1 if iterations is None else _integer(iterations, "lifetime iterations")
    return out


def ground_truth(injection: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The injected actions, and the records that were not injected (neither is a skip)."""
    if injection.get("schema_version") != 1:
        raise GroundTruthError(f"injection.json schema_version {injection.get('schema_version')!r}, expected 1")
    records = injection.get("injections")
    if not isinstance(records, list):
        raise GroundTruthError("injection.json has no list of injections")
    lifetimes = _lifetimes(injection)
    actions, not_injected = [], []
    for record in records:
        status = record.get("status")
        if status == "skipped_probability":
            continue
        succeeded = record.get("injection_succeeded")
        if not isinstance(succeeded, bool):
            raise GroundTruthError(f"injection {record.get('injection_id')!r}: injection_succeeded must be a boolean")
        incident = str(record.get("incident_id"))
        if incident not in lifetimes:
            raise GroundTruthError(f"injection {record.get('injection_id')!r}: incident {incident!r} is not in the manifest")
        item = {
            "injection_id": record.get("injection_id"), "occurrence_id": str(record.get("occurrence_id")),
            "incident_id": incident, "fault_id": record.get("fault_id"),
            "failure_type": record.get("failure_type"), "expected_kind": record.get("expected_kind"),
            "surface": (record.get("target") or {}).get("surface"),
            "rank": _integer(record.get("execution_rank"), "execution_rank"),
            "iteration": _integer(record.get("iteration"), "iteration"), "status": status,
        }
        item["window"] = [item["iteration"], item["iteration"] + lifetimes[incident] - 1]
        (actions if succeeded else not_injected).append(item)
    return actions, not_injected


_RANK_FILE = re.compile(r"\.rank(\d+)\.jsonl$")


def capture_files(folder: Path) -> dict[int, Path]:
    return {int(m.group(1)): p for p in sorted(folder.glob("*.jsonl")) if (m := _RANK_FILE.search(p.name))}


def _records(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a line cut short by a killed process: skipped, as TensorQuill's readers skip it
        if isinstance(record, dict):
            out.append(record)
    return out


def _is_step(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def incident_steps(incident: Mapping[str, Any]) -> dict[int, str]:
    """Every step an incident names, with the key that names it. An incident is listed once per kind
    and its `step` is the first; the later steps it covers are in a list under a key ending in
    `_steps` (`spike_steps` of `grad_spike`, measured on a CPU toy Nov 22, 2026: a gradient scaled at
    steps 25 and 45 gave one incident, step 25, with 45 in `spike_steps`). Such a list can be cut
    short (`spike_count` above its length); a step beyond it is not named, and reads as a miss."""
    steps: dict[int, str] = {}
    if _is_step(incident.get("step")):
        steps[incident["step"]] = "step"
    for key, value in incident.items():
        if key.endswith("_steps") and isinstance(value, list):
            for step in value:
                if _is_step(step):
                    steps.setdefault(step, key)
    return steps


def tensorquill_reports(files: Mapping[int, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every rank-and-step TensorQuill named, and what it named with a step and no rank."""
    from tensorquill import build_evidence
    from tensorquill.evidence import compare_replicas

    reports, unattributed = [], []
    every_record: list[dict[str, Any]] = []
    for rank, path in sorted(files.items()):
        records = _records(path)
        every_record.extend(records)
        evidence = build_evidence(path, detectors="python")
        for incident in evidence.get("incidents", []):
            steps = incident_steps(incident)
            entry = {"source": "rank_evidence", "kind": incident.get("kind"), "rank": rank}
            if not steps:
                unattributed.append({**entry, "step": incident.get("step"), "why": "no step"})
            for step, via in sorted(steps.items()):
                reports.append({**entry, "step": step, "via": via})
    if len(files) > 1 and every_record:
        replica = compare_replicas(every_record)
        for entry in replica.get("attribution", []) or []:
            if entry.get("verdict") == "attributed":
                for rank in entry.get("outliers") or []:
                    reports.append({"source": "replica_attribution", "kind": f"replica_outlier_by_{entry.get('by')}",
                                    "rank": rank, "step": entry.get("step"), "parameter": entry.get("parameter")})
        for incident in replica.get("replica_drift", []) or []:
            unattributed.append({"source": "replica_drift", "kind": "replica_drift", "rank": None,
                                 "step": incident.get("step"), "why": "names no rank"})
    return reports, unattributed


def compare(actions: list[dict[str, Any]], not_injected: list[dict[str, Any]], reports: Iterable[dict[str, Any]],
            *, start_step: int = 1) -> dict[str, Any]:
    """The matching rule of the docstring, on ground truth and reports already read."""
    def iteration(report: Mapping[str, Any]) -> int:
        return int(report["step"]) - start_step + 1

    def matches(report: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
        return report["rank"] == action["rank"] and action["window"][0] <= iteration(report) <= action["window"][1]

    reports = [dict(r, iteration=iteration(r)) for r in reports]
    occurrences: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        occurrences.setdefault(action["occurrence_id"], []).append(action)
    evaluated = []
    for occurrence_id, members in occurrences.items():
        matched = []
        for action in members:
            hits = [r for r in reports if matches(r, action)]
            matched.append({"injection_id": action["injection_id"], "rank": action["rank"], "window": action["window"],
                            "failure_type": action["failure_type"], "surface": action["surface"],
                            "matched_by": [{k: r.get(k) for k in ("source", "kind", "rank", "step", "via", "parameter") if k in r} for r in hits]})
        named = sum(bool(m["matched_by"]) for m in matched)
        evaluated.append({
            "occurrence_id": occurrence_id, "incident_id": members[0]["incident_id"],
            "iteration": members[0]["iteration"], "ranks": sorted({a["rank"] for a in members}),
            "result": "found" if named == len(matched) else "partial" if named else "missed", "actions": matched,
        })
    false_reports = []
    for report in reports:
        if not any(matches(report, action) for action in actions):
            near = [r["injection_id"] for r in not_injected if matches(report, r)]
            false_reports.append({**{k: report.get(k) for k in ("source", "kind", "rank", "step", "via", "iteration", "parameter") if k in report},
                                  **({"matches_record_not_injected": near} if near else {})})
    counts = {name: sum(e["result"] == name for e in evaluated) for name in ("found", "partial", "missed")}
    return {
        "rule": "a report credits an action only with the action's execution_rank and an iteration in its active window",
        "start_step": start_step, "occurrences": evaluated,
        "missed": [e["occurrence_id"] for e in evaluated if e["result"] != "found"],
        "false_reports": false_reports, "not_injected": not_injected,
        "summary": {"occurrences": len(evaluated), **counts, "injected_actions": len(actions),
                    "reports": len(reports), "false_reports": len(false_reports), "not_injected": len(not_injected)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifact-dir", type=Path, required=True, help="the campaign's --artifact-dir")
    parser.add_argument("--captures", type=Path, help="TensorQuill's captures (default: <artifact-dir>/tensorquill)")
    parser.add_argument("--measure", action="store_true", help="exit 0 whenever the comparison was made")
    args = parser.parse_args(argv)
    folder = args.captures or args.artifact_dir / "tensorquill"
    out = args.artifact_dir / "tensorquill_comparison.json"

    def not_proven(why: str) -> int:
        print(f"fault_injection_check: NOT PROVEN: {why}")
        return NOT_PROVEN

    try:
        injection = json.loads((args.artifact_dir / "injection.json").read_text(encoding="utf-8"))
        actions, not_injected = ground_truth(injection)
    except FileNotFoundError:
        return not_proven(f"no injection.json in {args.artifact_dir} (the campaign did not reach its end)")
    except (ValueError, GroundTruthError) as error:
        return not_proven(f"injection.json cannot be read as ground truth: {error}")
    files = capture_files(folder)
    if not files:
        return not_proven(f"no TensorQuill capture (*.rank<R>.jsonl) in {folder}")
    world = injection.get("world_size")
    if isinstance(world, int) and sorted(files) != list(range(world)):
        return not_proven(f"captures for ranks {sorted(files)}, the campaign ran {world}")
    attach = {}
    try:
        attach = json.loads((folder / "attach.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    start_step = int(attach.get("start_step", 1))
    # Step alignment is checked, not assumed: each rank's last captured step must be the campaign's
    # last iteration, so TensorQuill counted one step per iteration (SCOUT, GEMINI and the state reset
    # run around the optimizer step).
    completed = injection.get("completed_iterations")
    if isinstance(completed, bool) or not isinstance(completed, int):
        return not_proven(f"injection.json has no completed_iterations ({completed!r}): the run cannot be shown complete")
    pending = [r["injection_id"] for r in not_injected if r.get("status") == "pending"]
    if pending:
        return not_proven(f"{len(pending)} injection record(s) still pending when the campaign wrote its artifact "
                          f"({pending[:5]}): an incomplete run")
    last = {rank: max((r["step"] for r in _records(path) if _is_step(r.get("step"))), default=None)
            for rank, path in files.items()}
    expected_last = completed + start_step - 1
    off = {rank: step for rank, step in last.items() if step != expected_last}
    if off:
        return not_proven(f"TensorQuill's last step is not the campaign's last iteration on ranks {sorted(off)} "
                          f"(last steps {off}, expected {expected_last} for {completed} iterations from "
                          f"start_step {start_step}): its steps cannot be read as iterations")
    reports, unattributed = tensorquill_reports(files)
    result = compare(actions, not_injected, reports, start_step=start_step)
    result.update({"campaign": injection.get("campaign"), "manifest_identity": injection.get("manifest_identity"),
                   "world_size": world, "completed_iterations": injection.get("completed_iterations"),
                   "captures": {str(r): str(p) for r, p in files.items()}, "attach": attach,
                   "campaign_environment": attach.get("environment") or "not recorded: attach.json has none",
                   "unattributed": unattributed})
    s = result["summary"]
    strict = s["found"] == s["occurrences"] and not s["false_reports"]
    status = PASS if args.measure or strict else FAIL
    result["status"] = "PASS" if strict else "FAIL"  # the verdict, whatever the exit rule
    result["pass_rule"] = "every occurrence found (exact rank, iteration within the window of injection.json), no false report"
    result["exit_rule"] = "measure: exit 0 whenever compared" if args.measure else "strict: exit 0 only on PASS"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fault_injection_check: {s['occurrences']} occurrences ({s['injected_actions']} injected actions) on "
          f"{len(files)} ranks; TensorQuill named rank and step for {s['found']} (partial {s['partial']}, missed "
          f"{s['missed']}); {s['reports']} reports, {s['false_reports']} false; {s['not_injected']} records not "
          f"injected; {len(unattributed)} with a step and no rank")
    for e in result["occurrences"]:
        if e["result"] != "found":
            print(f"  {e['result']}: {e['occurrence_id']} (ranks {e['ranks']}, iteration {e['iteration']})")
    for r in result["false_reports"][:50]:
        print(f"  false report: {r['kind']} rank {r['rank']} step {r['step']} ({r['source']})")
    print(f"fault_injection_check: {result['status']} ({out})")
    return status


if __name__ == "__main__":
    sys.exit(main())
