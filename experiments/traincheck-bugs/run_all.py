"""Run every configuration in `catalog.py`, buggy and healthy, and write the results.

    python run_all.py              # run everything, then write results.json and RESULTS.md
    python run_all.py --only B1,B3 # run some bugs again, keeping the others' results
    python run_all.py --report     # only rewrite RESULTS.md from results.json

Each run is its own process (`<script> --variant buggy|healthy --tag <tag> ...`), with this
worktree's `sdk/src` first on the path, so the TensorQuill measured is the one in this
checkout. A run's log is `out/<run_id>.log`; its evidence and `result.json` are in
`out/<run_id>/`. `results.json` keeps each run's result summary (no records), and is what the
table is written from.

A kind counts as catching the bug in a configuration when it fires on the buggy run and not
on the healthy run of the same configuration. A kind that fires on both is listed as also on
the healthy run, and every incident of a healthy run is a false incident.

A pair is judged only when both runs completed and gave valid observations (`problem`): exit
status 0, a `RESULT` line naming a readable `result.json`, one summary per rank, and the bug's
symptom measured as true or false, with the healthy control's false. Otherwise the table says
why it was not judged. A run that fails, exits nonzero or passes `TIMEOUT` is recorded for its
own case, with its output in the log, and the sweep goes on.

A run of several ranks is read through the all-ranks view (`harness.all_ranks`), where the ranks'
replicas are compared: its `replica_drift` is the run's incident, in no rank's evidence. Such a
run has valid observations only with that view in its result, and a pair whose bug is one of
replica drift is judged only when both runs compared their replicas on at least one step, so
that a run nobody compared does not read as a healthy one.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from catalog import BUGS, SHARED_RUN

HERE = Path(__file__).resolve().parent
SDK = HERE.parent.parent / "sdk" / "src"
OUT = HERE / "out"
RESULTS = HERE / "results.json"
TABLE = HERE / "RESULTS.md"
TIMEOUT = 1800  # seconds a run may take before it and every process it started are killed
VARIANTS = ("buggy", "healthy")
LEVELS = {
    "shipped": "TrainCheck's script",
    "swapped": "TrainCheck's script, model and data swapped",
    "recreated": "Re-created here",
    "simulated": "Simulated here",
}


def _run_child(args: list[str], env: dict[str, str]) -> tuple[int | None, str]:
    """(exit status, or None when it passed TIMEOUT; stdout and stderr). The run gets a session
    of its own, so a timeout kills the processes it started too (the two-process runs' ranks),
    which would otherwise hold its output open and keep the sweep waiting."""
    proc = subprocess.Popen(args, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)

    def kill_group() -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            proc.kill()

    try:
        out, err = proc.communicate(timeout=TIMEOUT)
        return proc.returncode, out + err
    except subprocess.TimeoutExpired:
        kill_group()
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:  # something left the group and still holds the output
            out, err = "", ""
        return None, out + err + f"\nrun_all.py: killed after {TIMEOUT} s\n"
    except BaseException:
        # Ctrl-C reaches only this process now that the run has its own session: stop the run too.
        kill_group()
        raise


def run_one(script: str, variant: str, tag: str, extra: list[str]) -> dict[str, Any]:
    args = [sys.executable, script, "--variant", variant] + (["--tag", tag] if tag else []) + extra
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SDK), str(HERE)]))  # warnings kept: TensorQuill's are evidence
    returncode, output = _run_child(args, env)
    lines = output.splitlines()
    result_line = next((line for line in lines if line.startswith("RESULT ")), None)
    entry: dict[str, Any] = {"script": script, "variant": variant, "tag": tag, "args": extra, "returncode": returncode}
    name = f"{Path(script).name}-{variant}-{tag or 'default'}"
    failure = None
    if returncode is None:
        failure = f"timed out after {TIMEOUT} s"
    if result_line is not None:
        try:
            run_id = json.loads(result_line[len("RESULT "):])["run_id"]
            entry["run_id"] = run_id
            name = run_id
            entry["result"] = json.loads((OUT / run_id / "result.json").read_text())
        except (ValueError, KeyError, TypeError, OSError) as error:
            failure = failure or f"its RESULT line or result.json cannot be read: {type(error).__name__}: {error}"
    elif failure is None:
        errors = [line for line in lines if "Error" in line or "error" in line]
        failure = errors[-1].strip() if errors else (lines[-1] if lines else "no output")
    entry["ok"] = True
    failure = failure or problem(entry)
    if failure:
        entry["ok"] = False
        entry["error"] = failure
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.log").write_text(output)
    return entry


def problem(entry: dict[str, Any] | None) -> str | None:
    """Why `entry` is not a completed run with valid observations, or None when it is one. The
    sweep sets `ok` from this, and the table checks it again, since it is written from stored
    entries."""
    if not entry:
        return "not run"
    if not entry.get("ok"):
        return str(entry.get("error") or "failed")
    if entry.get("returncode") != 0:
        return f"exited with status {entry.get('returncode')} after printing its result"
    result = entry.get("result")
    if not isinstance(result, dict):
        return "no result"
    ranks, world = result.get("ranks"), result.get("world_size", 1)
    if not isinstance(ranks, dict) or not isinstance(world, int) or len(ranks) != world:
        return f"result has {len(ranks) if isinstance(ranks, dict) else 'no'} rank summaries for world size {world}"
    if not all(isinstance(r, dict) and isinstance(r.get("incidents"), list) for r in ranks.values()):
        return "a rank summary has no incidents list"
    if not isinstance(result.get("cross_rank", []), list):
        return "cross_rank is not a list"
    if world > 1 and not isinstance((result.get("all_ranks") or {}).get("relevant_kinds"), dict):
        return "a run of several ranks without the all-ranks view, where its replicas are compared"
    symptom = result.get("symptom")
    if not isinstance(symptom, dict) or not isinstance(symptom.get("bug_present"), bool):
        return "the symptom was not measured (bug_present is not true or false)"
    return None


def replicas_not_compared(entry: dict[str, Any]) -> str | None:
    """For a completed run whose relevant kinds name `replica_drift`: why its replicas count as
    not compared, or None when the all-ranks view compared them on at least one step."""
    result = entry["result"]
    if "replica_drift" not in (result.get("relevant") or []):
        return None
    check = ((result.get("all_ranks") or {}).get("relevant_kinds") or {}).get("replica_drift") or {}
    compared = (check.get("coverage") or {}).get("compared")
    if check.get("value") == "checked" and isinstance(compared, int) and not isinstance(compared, bool) and compared >= 1:
        return None
    return f"replica_drift is {check.get('value', 'not stated')}, with {compared if compared is not None else 'no'} steps compared"


def key(bug: str, tag: str, variant: str) -> str:
    return f"{bug}|{tag}|{variant}"


def run(only: set[str] | None) -> dict[str, Any]:
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() and only else {}
    for bug in BUGS:
        if only and bug["id"] not in only:
            continue
        for tag, extra in bug.get("runs", []):
            for variant in VARIANTS:
                print(f"{bug['id']} {tag or 'default'} {variant} ...", flush=True)
                entry = run_one(bug["script"], variant, tag, extra)
                results[key(bug["id"], tag, variant)] = entry
                RESULTS.write_text(json.dumps(results, indent=1, default=str))
                incidents = _incidents(entry) if entry["ok"] else entry["error"]
                print(f"   -> {incidents}", flush=True)
    return results


# --- the table ------------------------------------------------------------------------------


def _incidents(entry: dict[str, Any]) -> list[tuple[str, int, str]]:
    """(kind, step, ranks) for every incident of a run, ranks merged when a kind fires at the same step."""
    if not entry.get("ok"):
        return []
    result = entry["result"]
    seen: dict[tuple[str, int], list[str]] = {}
    for rank, summary in result["ranks"].items():
        for incident in summary["incidents"]:
            seen.setdefault((incident["kind"], incident["step"]), []).append(rank.replace("rank", ""))
    for incident in result.get("cross_rank", []):
        seen.setdefault((incident["kind"], incident["step"]), []).append("across")
    multi = result.get("world_size", 1) > 1
    return [(kind, step, ",".join(ranks) if multi else "") for (kind, step), ranks in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0]))]


def _fmt(incidents: list[tuple[str, int, str]]) -> str:
    if not incidents:
        return "none"
    def where(ranks: str) -> str:
        if not ranks:
            return ""
        own = ",".join(r for r in ranks.split(",") if r != "across")
        return " (" + " and ".join(x for x in (f"ranks {own}" if own else "", "across ranks" if "across" in ranks.split(",") else "") if x) + ")"

    return ", ".join(f"`{k}` @ {s}{where(r)}" for k, s, r in incidents)


def _pair(results: dict[str, Any], bug: str, tag: str) -> dict[str, Any]:
    """A configuration's two runs, and what the buggy one caught: only when both runs completed
    with valid observations and the bug did not show on the healthy control. Otherwise `why`
    says why the pair is not judged, and nothing is credited."""
    buggy = results.get(key(bug, tag, "buggy"))
    healthy = results.get(key(bug, tag, "healthy"))
    out: dict[str, Any] = {"tag": tag or "default", "buggy": buggy, "healthy": healthy}
    if not buggy or not healthy:
        return out
    out["buggy_problem"], out["healthy_problem"] = problem(buggy), problem(healthy)
    out["bug_showed"] = None if out["buggy_problem"] else buggy["result"]["symptom"]["bug_present"]
    if out["healthy_problem"]:
        out["why"] = f"the healthy control failed ({out['healthy_problem'][:120]})"
    elif healthy["result"]["symptom"]["bug_present"]:
        out["why"] = "the bug showed on the healthy control, so it is not a control"
    if not out["healthy_problem"]:
        out["false"] = _incidents(healthy)
    for variant, entry in (("healthy control", healthy), ("buggy run", buggy)):
        if not out.get("why") and not problem(entry):
            uncompared = replicas_not_compared(entry)
            if uncompared:
                out["why"] = f"the ranks' replicas were not compared on the {variant} ({uncompared})"
    if out["buggy_problem"] or out.get("why"):
        return out
    b = _incidents(buggy)
    healthy_kinds = {k for k, _, _ in out["false"]}
    out["caught"] = [i for i in b if i[0] not in healthy_kinds]
    out["shared"] = [i for i in b if i[0] in healthy_kinds]
    return out


def _settings(entry: dict[str, Any]) -> str:
    s = entry["result"]["settings"]
    keep = [f"{k}={v}" for k, v in s.items() if k not in ("source", "optimizer")]
    return "; ".join(keep)


def _symptom(entry: dict[str, Any]) -> str:
    s = dict(entry["result"]["symptom"])
    s.pop("bug_present", None)
    return "; ".join(f"{k}={v}" for k, v in s.items())


def write_table(results: dict[str, Any]) -> None:
    rows = []
    detail = []
    for bug in BUGS:
        source = SHARED_RUN.get(bug["id"], bug["id"])
        source_bug = next(b for b in BUGS if b["id"] == source)
        configs = [_pair(results, source, tag) for tag, _ in source_bug.get("runs", [])]
        if bug["how"] == "not run":
            ran = bug["cpu"]
            caught = "not run"
            false = "not run"
            showed = "not run"
        else:
            label = LEVELS[bug["how"]] + (f" (the run of {source}, the effect only)" if source != bug["id"] else "")
            ran = f"{label}. CPU: {bug['cpu']}"
            parts_caught, parts_false, parts_showed = [], [], []
            for c in configs:
                tag = c["tag"]
                if not (c["buggy"] and c["healthy"]):
                    parts_caught.append(f"{tag}: not run yet")
                    continue
                if c["buggy_problem"]:
                    parts_caught.append(f"{tag}: buggy run failed ({c['buggy_problem'][:120]})")
                else:
                    if c.get("why"):
                        parts_caught.append(f"{tag}: not judged: {c['why']}")
                    else:
                        shared = f"; also on healthy: {_fmt(c['shared'])}" if c["shared"] else ""
                        verdict = (_fmt(c["caught"]) if c["caught"] else "not caught") if c["bug_showed"] else \
                            "nothing to catch: the bug did not show"
                        parts_caught.append(f"{tag}: {verdict}{shared}")
                    parts_showed.append(f"{tag}: {'yes' if c['bug_showed'] else 'no'}")
                if c["healthy_problem"]:
                    parts_false.append(f"{tag}: healthy run failed ({c['healthy_problem'][:120]})")
                else:
                    parts_false.append(f"{tag}: {_fmt(c['false'])}")
            caught = "<br>".join(parts_caught)
            false = "<br>".join(parts_false)
            showed = "<br>".join(parts_showed) or "no run completed"
        reach = bug.get("reach", "")
        needs = bug.get("needs", "")
        notes = " ".join(x for x in (f"Needs: {needs}." if needs else "",
                                     f"Reach (judged, not tested): {reach}" if reach else "") if x)
        rows.append(f"| {bug['id']} {bug['paper']} | {bug['synopsis']} | {bug['location']} | {bug['in_clone']} | {ran} | "
                    f"{showed} | {caught} | {false} | {bug['review']} | {notes} |")
        if source == bug["id"]:
            for c in configs:
                for variant in VARIANTS:
                    entry = c.get(variant)
                    if not entry:
                        continue
                    failure = problem(entry)
                    if failure is None:
                        r0 = entry["result"]["ranks"].get("rank0", {})
                        run = entry["result"].get("all_ranks") or {}  # a run of several ranks: the view of all of them
                        short = ", ".join(r0.get("checked_without_enough_history", [])) or "none"
                        relevant = "; ".join(
                            f"{k}: {v.get('value')} ({v.get('records')} of {v.get('records_needed')}"
                            + (f"; {v['coverage'].get('compared')} of {v['coverage'].get('steps')} steps compared"
                               if isinstance(v.get("coverage"), dict) else "") + ")"
                            for k, v in (run.get("relevant_kinds") or r0.get("relevant_kinds", {})).items())
                        nc = "; ".join(f"{n.get('check')}: {str(n.get('reason'))[:90]}" for n in r0.get("not_checked", [])) or "none"
                        detail.append(f"| {entry['run_id']} | {_settings(entry)} | {_symptom(entry)} | {_fmt(_incidents(entry))} | "
                                      f"{run.get('run_state') or r0.get('run_state')} | {relevant} | {short} | {nc} |")
                    else:
                        detail.append(f"| {bug['id']} {c['tag']} {variant} | args {entry['args']} | run failed: {failure[:200]} | | | | | |")
    header = (
        "| Bug | Synopsis (Table 3) | Failure location (AE) | Reproduction in the clone | Ran here | Bug showed (plain torch) "
        "| Caught: kind @ step, buggy run only | False incidents (healthy run) | Review's reading (D7) | Notes |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    detail_header = (
        "| Run | Settings | Symptom measured with plain torch | Incidents | Run state | Relevant kinds: value (records of needed) "
        "| Checked without enough history | Not checked |\n|---|---|---|---|---|---|---|---|"
    )
    text = (
        "# TrainCheck's 20 silent errors against TensorQuill: results\n\n"
        "Written by `run_all.py` from `results.json`; do not edit by hand. What the table shows and does not show is in "
        "[README.md](README.md). Steps are TensorQuill's (0-based optimizer steps). \"default\" is each script's own "
        "configuration; the others are named after what they change. A run of several ranks is read through the "
        "all-ranks view: an incident marked \"across ranks\" is the run's own, from comparing the ranks, and that "
        "run's state and relevant kinds are the view's, with how many steps' replicas were compared. Its last two "
        "columns are rank 0's.\n\n"
        "## One row per bug\n\n" + header + "\n" + "\n".join(rows) + "\n\n"
        "## Every run\n\n" + detail_header + "\n" + "\n".join(detail) + "\n"
    )
    TABLE.write_text(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="comma-separated bug IDs to run again")
    p.add_argument("--report", action="store_true", help="only rewrite RESULTS.md")
    args = p.parse_args()
    if args.report:
        results = json.loads(RESULTS.read_text())
    else:
        results = run({x.strip() for x in args.only.split(",") if x.strip()} or None)
    write_table(results)
    print(f"wrote {TABLE.name}", flush=True)


if __name__ == "__main__":
    main()
