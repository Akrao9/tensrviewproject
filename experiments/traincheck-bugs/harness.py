"""Shared harness for the TrainCheck bug runs: start TensorQuill, build the evidence, report.

Every bug script calls `start` once per process and `finish` at the end. Records go to
`out/<run_id>/<run_id>.rank<r>.jsonl` inside this folder (never `tensorquill-out/` in the
working directory). `finish` closes the session, builds the evidence for every rank file the
run wrote (`tensorquill.build_evidence`, the same code `python -m tensorquill` runs), and, when
there are several ranks, the all-ranks view of the run (`all_ranks`), which is where the ranks'
replicas are compared: one rank's evidence does not hold `replica_drift` (since wave Z, Nov 20,
2026, a rank records an observation and the comparison is made across ranks). It writes
`out/<run_id>/result.json`:
the symptom the script measured with plain torch (independent of TensorQuill), the incidents
with kind, class and step, which kinds were checked, and what was not checked.

Nothing here changes TensorQuill. The default options are the ones the lane was asked to
use: `capture="full", updates=True`, plus whatever a script passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

import tensorquill
from tensorquill.evidence import build_evidence, combine_rank_evidence, load_records, resolve_attempts

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# Keys of an incident that are bulky or repeat what the summary already says.
_DROP = {"scalars", "samples", "class", "kind", "step", "note"}
_LIST_MAX = 4


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--variant", choices=("buggy", "healthy"), default="buggy")
    p.add_argument("--tag", default="", help="suffix for the run ID, to tell configurations apart")
    p.add_argument("--weights", type=int, default=None,
                   help="TensorQuill weight-check interval (default: the capture mode's, every 50 steps)")
    p.add_argument("--replica-every", type=int, default=None,
                   help="TensorQuill cross-rank check interval, for the multi-process scripts (default 50)")
    return p


def run_id_for(bug: str, args: argparse.Namespace) -> str:
    return f"{bug}-{args.variant}{('-' + args.tag) if args.tag else ''}"


def folder_for(run_id: str, *, clean: bool) -> Path:
    folder = OUT / run_id
    if clean and folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def start(model: torch.nn.Module, run_id: str, *, rank: int = 0, weights: int | None = None,
          **options: Any) -> tensorquill.Session:
    """`tensorquill.init(model, capture="full", updates=True, ...)` writing into this folder.

    Single-process scripts get a clean folder here; multi-process ones clean it in the parent
    before spawning, so one rank never deletes another's file.
    """
    folder = folder_for(run_id, clean=False)
    sink = tensorquill.JsonlSink(folder / f"{run_id}.rank{rank}.jsonl")
    kwargs: dict[str, Any] = {"capture": "full", "updates": True}
    if weights is not None:
        kwargs["weights"] = weights
    kwargs.update(options)
    return tensorquill.init(model, run_id=run_id, sink=sink, **kwargs)


def _short(value: Any) -> Any:
    if isinstance(value, list):
        shown = [_short(v) for v in value[:_LIST_MAX]]
        return shown + ([f"... {len(value) - _LIST_MAX} more"] if len(value) > _LIST_MAX else [])
    if isinstance(value, Mapping):
        return {k: _short(v) for k, v in value.items() if k not in _DROP}
    if isinstance(value, float):
        return float(f"{value:.6g}")
    return value


def summarize(evidence: Mapping[str, Any], relevant: tuple[str, ...] = ()) -> dict[str, Any]:
    """The parts of an evidence file the results table needs; `relevant` kinds get their
    `kind_checks` entry copied, so a run too short to judge reads as that, not as clean."""
    incidents = [
        {"kind": i.get("kind"), "class": i.get("class"), "step": i.get("step"), "detail": _short(i)}
        for i in evidence.get("incidents", [])
    ]
    checks = evidence.get("kind_checks") or {}
    by_value: dict[str, list[str]] = {}
    short_history: list[str] = []
    for kind, entry in checks.items():
        by_value.setdefault(entry.get("value", "?"), []).append(kind)
        if entry.get("value") == "checked" and entry.get("enough_history") is False:
            short_history.append(f"{kind} ({entry.get('records')} of {entry.get('records_needed')} records)")
    not_checked = [
        {k: v for k, v in entry.items() if k in ("check", "kind", "reason", "step")}
        for entry in (evidence.get("not_checked") or [])
    ]
    return {
        "run_state": (evidence.get("run_state") or {}).get("state"),
        "incidents": incidents,
        "kinds_checked": sorted(by_value.get("checked", [])),
        "kinds_unavailable": sorted(by_value.get("unavailable", [])),
        "checked_without_enough_history": sorted(short_history),
        "not_checked": not_checked[:8],
        "relevant_kinds": {
            kind: {k: checks[kind].get(k) for k in ("value", "records", "records_needed", "enough_history", "reason")
                   if checks[kind].get(k) is not None}
            for kind in relevant if kind in checks
        },
    }


def _rank_of(path: Path) -> int:
    return int(path.name.rsplit(".rank", 1)[1].split(".")[0])


def all_ranks(files: list[Path], per_rank: Mapping[int, Mapping[str, Any]],
              relevant: tuple[str, ...] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    """The all-ranks view of a run of local files, and the parts of it the results table needs.

    Built as `python -m tensorquill compare` builds it for local files: each rank's evidence with
    its last step, put together by `combine_rank_evidence`, which compares the ranks' replica
    observations from every rank's records as written (not the resolved ones: a restart's rows
    are needed there). The run's own incidents (`run_incidents`: `replica_drift`, and
    `memory_imbalance` where the ranks have memory readings, which a CPU run does not) are in no
    rank's evidence.

    Each `relevant` kind's entry is the view's, with its `coverage` for `replica_drift` (the
    steps with an observation and how many of them were compared across the ranks): a run whose
    replicas were never compared has no `replica_drift` incident either, and must not read as
    one whose replicas agree."""
    written: list[dict[str, Any]] = []
    with_step: dict[int, dict[str, Any]] = {}
    for path in files:
        own = load_records(path, all_attempts=True)
        written.extend(own)
        steps = [r["step"] for r in resolve_attempts(own) if isinstance(r.get("step"), int) and not isinstance(r.get("step"), bool)]
        with_step[_rank_of(path)] = {**per_rank[_rank_of(path)], "last_step": max(steps) if steps else None}
    combined = combine_rank_evidence(with_step, replica_records=written)
    checks = combined.get("kind_checks") or {}
    kinds: dict[str, Any] = {}
    for kind in relevant:
        if kind not in checks:
            continue
        kinds[kind] = {k: checks[kind].get(k) for k in ("value", "records", "records_needed", "enough_history", "reason")
                       if checks[kind].get(k) is not None}
        coverage = checks[kind].get("coverage")
        if isinstance(coverage, Mapping):
            kinds[kind]["coverage"] = {k: coverage.get(k) for k in ("steps", "compared") if coverage.get(k) is not None}
    summary = {
        "run_state": (combined.get("run_state") or {}).get("state"),
        "judged_ranks": combined.get("judged_ranks"),
        "unknown_ranks": combined.get("unknown_ranks"),
        "incidents": [{"kind": i.get("kind"), "class": i.get("class"), "step": i.get("step"), "detail": _short(i)}
                      for i in combined.get("run_incidents") or []],
        "relevant_kinds": kinds,
    }
    return combined, summary


def duplicate_batches(files: list[Path]) -> list[dict[str, Any]]:
    """`duplicate_batches` over every rank's records. The all-ranks view of local files does not
    make this comparison (only the server does: its entry there is `not_configured`), so it is
    made here, as before."""
    records: list[dict[str, Any]] = []
    for path in files:
        records.extend(load_records(path))
    incident = tensorquill.detect_duplicate_batches(records)
    return [{"kind": incident.get("kind"), "step": incident.get("step"), "detail": _short(incident)}] if incident else []


def finish(session: tensorquill.Session | None, run_id: str, *, bug: str, variant: str,
           symptom: Mapping[str, Any], settings: Mapping[str, Any], rank: int = 0,
           world_size: int = 1, relevant: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Close the session; on rank 0, build the evidence of every rank and write the result.

    With several ranks the caller must have closed every rank's session (a barrier) before
    rank 0 reads the files.
    """
    if session is not None:
        session.close()
    if rank != 0:
        return None
    folder = OUT / run_id
    files = sorted(folder.glob(f"{run_id}.rank*.jsonl"))
    per_rank = {}
    evidences: dict[int, dict[str, Any]] = {}
    for path in files:
        evidence = build_evidence(path)
        r = path.name.rsplit(".rank", 1)[1].split(".")[0]
        (folder / f"{run_id}.rank{r}.evidence.json").write_text(json.dumps(evidence, indent=1, default=str))
        per_rank[f"rank{r}"] = summarize(evidence, relevant)
        evidences[int(r)] = evidence
    run: dict[str, Any] | None = None
    across: list[dict[str, Any]] = []
    if len(files) > 1:
        combined, run = all_ranks(files, evidences, relevant)
        (folder / f"{run_id}.all-ranks.evidence.json").write_text(json.dumps(combined, indent=1, default=str))
        across = [{"kind": i["kind"], "step": i["step"], "detail": i["detail"]} for i in run["incidents"]]
        across += [i for i in duplicate_batches(files) if (i["kind"], i["step"]) not in {(a["kind"], a["step"]) for a in across}]
    result = {
        "bug": bug,
        "variant": variant,
        "run_id": run_id,
        "torch": torch.__version__,
        "settings": dict(settings),
        "symptom": dict(symptom),
        "ranks": per_rank,
        "cross_rank": across,
        **({"all_ranks": run} if run is not None else {}),
        "world_size": world_size,
        "relevant": list(relevant),
    }
    (folder / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print("RESULT " + json.dumps({"run_id": run_id, "symptom": result["symptom"],
                                  "incidents": {r: [(i["kind"], i["step"]) for i in s["incidents"]]
                                                for r, s in per_rank.items()},
                                  "cross_rank": [(i["kind"], i["step"]) for i in result["cross_rank"]]},
                                 default=str), flush=True)
    return result


# --- data and models shared by several scripts ---------------------------------------------


def blobs(n: int, shape: tuple[int, ...], classes: int, seed: int, noise: float = 1.0,
          template_seed: int = 0) -> torch.utils.data.TensorDataset:
    """A learnable synthetic classification set: one fixed random template per class
    (`template_seed`) plus Gaussian noise (`seed`). Stands in for the scripts' downloaded
    datasets (CIFAR-100, MNIST); a training and a test set share `template_seed`."""
    templates = torch.randn((classes, *shape), generator=torch.Generator().manual_seed(template_seed))
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=g)
    x = templates[labels] + noise * torch.randn((n, *shape), generator=g)
    return torch.utils.data.TensorDataset(x, labels)


def param_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def changed_params(model: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> list[str]:
    return [n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])]


# --- several CPU processes (gloo) ----------------------------------------------------------


def launch(target: Any, world_size: int, run_id: str, *args: Any) -> None:
    """Run `target(rank, world_size, init_file, *args)` in `world_size` spawned processes,
    rendezvousing through a file in the run's (cleaned) folder rather than a TCP port, since
    other jobs share this machine."""
    import torch.multiprocessing as mp

    folder = folder_for(run_id, clean=True)
    mp.spawn(target, args=(world_size, str(folder / "rendezvous"), *args), nprocs=world_size, join=True)


def join_group(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)


def replica_max_diff(params: list[torch.Tensor]) -> float:
    """Largest absolute difference between rank 0's and any other rank's parameters."""
    dist = torch.distributed
    flat = torch.cat([p.detach().reshape(-1).double() for p in params])
    gathered = [torch.zeros_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flat)
    return max(float((gathered[0] - g).abs().max()) for g in gathered[1:])
