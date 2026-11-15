"""Read TrainCheck's violations into TensorQuill's evidence.

TrainCheck (OSDI '25) infers training invariants from a clean reference run and checks a
target run against them. It is a separate tool with its own way of instrumenting PyTorch,
and it needs a reference run that TensorQuill does not. TensorQuill does not embed or
re-implement it: this reads the compact `violations_summary.json` that `traincheck-check`
writes (one per checked trace, under its results directory) and shows the violated
invariants as one incident beside TensorQuill's own, so an engineer or agent reads a
single `evidence.json`. Running both tools in one process has not been tried, since each
patches PyTorch's API; run TensorQuill in the job and TrainCheck as it documents.

The format read is the one TrainCheck's `build_violations_summary` writes (read in the
sibling checkout at `7a62cbb`):

    {"first_violation_step": int | null,
     "distinct_invariants_violated": int,
     "violations": [{"display_name": str, "relation_type": str,
                     "first_step": int | null, "last_step": int | null,
                     "occurrences": int}, ...]}

TrainCheck counts steps by its own iteration counter (`meta_vars.step`), which matches
TensorQuill's optimizer steps only when the loop steps the optimizer once per iteration
and both started from the same step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUMMARY_NAME = "violations_summary.json"
SHOWN = 10  # violated invariants listed in the incident
LABEL = 200


def read_traincheck(source: str | Path) -> dict[str, Any]:
    """Merge the TrainCheck summaries at `source`: one `violations_summary.json`, or a
    results directory holding one or more (`traincheck-check` writes one per trace).

    Raises `FileNotFoundError` when there is none and `ValueError` when a file is not
    in the format above.
    """
    path = Path(source)
    if path.is_dir():
        files = sorted([*path.glob(SUMMARY_NAME), *path.glob(f"*/{SUMMARY_NAME}")])
        if not files:
            raise FileNotFoundError(f"no {SUMMARY_NAME} in {path} or its sub-folders")
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"{path} does not exist")
    violations: list[dict[str, Any]] = []
    firsts: list[int] = []
    distinct = 0
    for file in files:
        try:
            document = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{file} is not readable JSON: {error}") from error
        if (
            not isinstance(document, dict)
            or not isinstance(document.get("violations"), list)
            or not isinstance(document.get("distinct_invariants_violated"), int)
        ):
            raise ValueError(f"{file} is not a TrainCheck violations summary")
        distinct += document["distinct_invariants_violated"]
        if isinstance(document.get("first_violation_step"), int):
            firsts.append(document["first_violation_step"])
        for entry in document["violations"]:
            if not isinstance(entry, dict) or "display_name" not in entry:
                raise ValueError(f"{file} has a violation without a display_name")
            violations.append(
                {
                    "display_name": str(entry["display_name"])[:LABEL],
                    "relation_type": entry.get("relation_type"),
                    "first_step": entry.get("first_step"),
                    "last_step": entry.get("last_step"),
                    "occurrences": entry.get("occurrences"),
                }
            )
    violations.sort(key=lambda v: (v["first_step"] is None, v["first_step"] if v["first_step"] is not None else 0))
    return {
        "files": [str(f) for f in files],
        "distinct_invariants_violated": distinct,
        "first_violation_step": min(firsts) if firsts else None,
        "violations": violations,
    }


def traincheck_incident(summary: dict[str, Any], *, step: int) -> dict[str, Any] | None:
    """The incident for a summary, or None when TrainCheck found no violation.

    The incident's `step` is TrainCheck's first violation step; `step` here is the fallback
    (the last step TensorQuill recorded) for a violation with no step information.
    """
    if not summary["distinct_invariants_violated"]:
        return None
    first = summary["first_violation_step"]
    return {
        "kind": "traincheck_violation",
        "step": first if first is not None else step,
        "distinct_invariants_violated": summary["distinct_invariants_violated"],
        "violations": summary["violations"][:SHOWN],
        "source": summary["files"],
        "note": "step is TrainCheck's own iteration count",
    }
