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
"""

from __future__ import annotations

import argparse
import json
import os
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
VARIANTS = ("buggy", "healthy")
LEVELS = {
    "shipped": "TrainCheck's script",
    "swapped": "TrainCheck's script, model and data swapped",
    "recreated": "Re-created here",
    "simulated": "Simulated here",
}


def run_one(script: str, variant: str, tag: str, extra: list[str]) -> dict[str, Any]:
    args = [sys.executable, script, "--variant", variant] + (["--tag", tag] if tag else []) + extra
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SDK), str(HERE)]))  # warnings kept: TensorQuill's are evidence
    proc = subprocess.run(args, cwd=HERE, env=env, capture_output=True, text=True, timeout=1800)
    lines = (proc.stdout + proc.stderr).splitlines()
    result_line = next((line for line in lines if line.startswith("RESULT ")), None)
    entry: dict[str, Any] = {"script": script, "variant": variant, "tag": tag, "args": extra, "returncode": proc.returncode}
    if result_line is None:
        errors = [line for line in lines if "Error" in line or "error" in line]
        entry["ok"] = False
        entry["error"] = errors[-1].strip() if errors else (lines[-1] if lines else "no output")
        name = f"{script}-{variant}-{tag or 'default'}"
    else:
        run_id = json.loads(result_line[len("RESULT "):])["run_id"]
        entry["ok"] = True
        entry["run_id"] = run_id
        entry["result"] = json.loads((OUT / run_id / "result.json").read_text())
        name = run_id
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.log").write_text(proc.stdout + proc.stderr)
    return entry


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
        seen.setdefault((incident["kind"], incident["step"]), []).append("all")
    multi = result.get("world_size", 1) > 1
    return [(kind, step, ",".join(ranks) if multi else "") for (kind, step), ranks in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0]))]


def _fmt(incidents: list[tuple[str, int, str]]) -> str:
    if not incidents:
        return "none"
    return ", ".join(f"`{k}` @ {s}" + (f" (ranks {r})" if r else "") for k, s, r in incidents)


def _pair(results: dict[str, Any], bug: str, tag: str) -> dict[str, Any]:
    buggy = results.get(key(bug, tag, "buggy"))
    healthy = results.get(key(bug, tag, "healthy"))
    out: dict[str, Any] = {"tag": tag or "default", "buggy": buggy, "healthy": healthy}
    if not buggy or not healthy:
        return out
    b, h = _incidents(buggy), _incidents(healthy)
    healthy_kinds = {k for k, _, _ in h}
    out["caught"] = [i for i in b if i[0] not in healthy_kinds]
    out["shared"] = [i for i in b if i[0] in healthy_kinds]
    out["false"] = h
    out["bug_showed"] = buggy["result"]["symptom"].get("bug_present") if buggy.get("ok") else None
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
                if not c["buggy"].get("ok"):
                    parts_caught.append(f"{tag}: buggy run failed ({c['buggy']['error'][:120]})")
                else:
                    shared = f"; also on healthy: {_fmt(c['shared'])}" if c["shared"] else ""
                    verdict = (_fmt(c["caught"]) if c["caught"] else "not caught") if c["bug_showed"] else \
                        "nothing to catch: the bug did not show"
                    parts_caught.append(f"{tag}: {verdict}{shared}")
                    parts_showed.append(f"{tag}: {'yes' if c['bug_showed'] else 'no'}")
                if not c["healthy"].get("ok"):
                    parts_false.append(f"{tag}: healthy run failed ({c['healthy']['error'][:120]})")
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
                    if entry.get("ok"):
                        r0 = entry["result"]["ranks"].get("rank0", {})
                        short = ", ".join(r0.get("checked_without_enough_history", [])) or "none"
                        relevant = "; ".join(f"{k}: {v.get('value')} ({v.get('records')} of {v.get('records_needed')})"
                                             for k, v in r0.get("relevant_kinds", {}).items())
                        nc = "; ".join(f"{n.get('check')}: {str(n.get('reason'))[:90]}" for n in r0.get("not_checked", [])) or "none"
                        detail.append(f"| {entry['run_id']} | {_settings(entry)} | {_symptom(entry)} | {_fmt(_incidents(entry))} | "
                                      f"{r0.get('run_state')} | {relevant} | {short} | {nc} |")
                    else:
                        detail.append(f"| {bug['id']} {c['tag']} {variant} | args {entry['args']} | run failed: {entry['error'][:200]} | | | | | |")
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
        "configuration; the others are named after what they change.\n\n"
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
