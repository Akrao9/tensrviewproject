"""Check TensorQuill's cross-rank replica comparison on any number of ranks.

Each rank trains the same two-layer model with DDP and records its own observation of every
scheduled step (contract `REPLICA_OBSERVATION_CONTRACT.md`, section 1). The driver plants a
divergence on a chosen rank, parameter and step, and rank 0 reads the verdict while the job trains,
from one of two places:

- `--mode files`: the local rank files, read by the Python reader every `--look-seconds`;
- `--mode server`: the object `replica` at the top level of the run's `watchdog.json` in a running
  server's store (start the server first; `--status-interval` is its `--watchdog-seconds`).

What it asserts, from that result object and never from one rank's record:

- every scheduled step is listed once, with complete coverage, for this run's launch and group;
- a planted step reads drift on exactly the planted parameter and raises one `replica_drift` naming
  that parameter, the step, the launch and the group; a clean step reads agree and raises nothing;
- `add`, `neg` and `swap` move the norm, the sum, and only the projection, across ranks;
- the clock (section 7): from the last required observation's arrival (the file holds the line, or
  the server acknowledged it) to the first look or status holding the step's verdict, at most one
  look or status interval plus `--latency-margin`. Both times and the difference are printed.

With `--hold-rank R`, rank R keeps training while its telemetry from `--hold-from` on is held back
in the driver's own sink (`replica_proof.ProofSink`), never in the SDK:

- `--release after-verdict` (a delayed rank): the step must read `insufficient` naming R while
  training goes on, no step on any rank may wait, and once released every held step is judged;
- `--release never` (a rank that never reports): the steps stay `insufficient` naming R, nothing is
  raised for them, and training runs to its end on every rank.

Which rank drifted is printed as evidence and not asserted: D1 is held. With two ranks the default
plants are two ranks that disagree; with three or more, one outlier among the rest.

Exit status: 0 PASS, 1 FAIL, 2 bad arguments, 3 NOT PROVEN (the reader could not be found).

    python -m torch.distributed.run --nproc-per-node=3 --master-addr 127.0.0.1 --master-port 29531 \\
        examples/replica_check.py --cpu --out runs/replica
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorquill  # noqa: E402
import replica_proof as proof  # noqa: E402
from tensorquill.flight import offset_path  # noqa: E402


SHARED_BYTES = 2048  # room for the run ID and one message, sent as one tensor


def _share_run(plan: proof.Plan, rank: int, device: torch.device) -> tuple[str, Path, str | None]:
    """Rank 0 names the run and makes its folder; every rank learns both before training.

    Sent as one int64 tensor on the run's device (a length, then the bytes of a JSON list) with
    `dist.broadcast`, not with `broadcast_object_list`: torch reads that call's bytes back through
    NumPy, which TensorQuill does not require, and every rank but 0 failed there without it."""
    problem = None
    run_id = plan.run_id or f"replica-check-{secrets.token_hex(6)}"
    work = plan.out / run_id
    if rank == 0:
        if work.exists() and any(work.iterdir()):
            problem = f"{work} is not empty; choose a new --out or --run-id"
        else:
            work.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([run_id, problem]).encode("utf-8")
    if len(payload) > SHARED_BYTES:
        payload = json.dumps(["", f"the run ID and its folder's path are too long to share ({len(payload)} bytes,"
                                  f" {SHARED_BYTES} at most); choose a shorter --run-id or --out"]).encode("utf-8")
    buffer = torch.zeros(1 + SHARED_BYTES, dtype=torch.int64, device=device)
    if rank == 0:
        buffer[0] = len(payload)
        buffer[1 : 1 + len(payload)] = torch.tensor(list(payload), dtype=torch.int64, device=device)
    dist.broadcast(buffer, src=0)
    shared = json.loads(bytes(buffer[1 : 1 + int(buffer[0])].tolist()).decode("utf-8"))
    return shared[0], plan.out / shared[0], shared[1]


def _write_json(path: Path, document: dict) -> None:
    scratch = path.with_name(f".{path.name}.tmp")
    scratch.write_text(json.dumps(document), encoding="utf-8")
    os.replace(scratch, path)


def _wait_for_reports(work: Path, world: int, seconds: float) -> dict[int, dict]:
    deadline = time.time() + seconds
    reports: dict[int, dict] = {}
    while True:
        for rank in range(world):
            if rank not in reports:
                try:
                    reports[rank] = json.loads((work / f"done.rank{rank}.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
        if len(reports) == world or time.time() > deadline:
            return reports
        time.sleep(0.05)


def _last_arrival(reports: dict[int, dict], plan: proof.Plan) -> float | None:
    times = [proof.arrival(report, step, plan.mode) for report in reports.values() for step in plan.scheduled
             if not (plan.hold_rank == report.get("rank") and plan.never_judged(step))]
    return None if any(t is None for t in times) or not times else max(times)


def _fingerprint_time(model: torch.nn.Module, device: torch.device, rank: int) -> None:
    """Today's fingerprint cost on this device, with and without the projection: printed, never
    asserted (a tiny model: only the ratio and a GPU run mean anything). Never stops the check."""
    try:
        timings = {}
        for projection in (False, True):
            tensorquill.parameter_fingerprints(model.parameters(), projection=projection)  # warm up
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                tensorquill.parameter_fingerprints(model.parameters(), projection=projection)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings[projection] = (time.perf_counter() - start) / 20 * 1000
    except Exception as error:  # noqa: BLE001  a timing is evidence, not part of the check
        print(f"rank {rank} fingerprint time not measured: {type(error).__name__}: {error}")
        return
    print(f"rank {rank} fingerprint time on {device.type}: {timings[False]:.3f} ms norm and sum,"
          f" {timings[True]:.3f} ms with the projection (a tiny model: only the ratio and a GPU run mean anything)")


def main(argv: list[str] | None = None) -> int:
    cuda = torch.cuda.device_count() if torch.cuda.is_available() else None
    try:
        plan = proof.validate(argv, os.environ, cuda)
    except proof.UsageError as error:
        print(f"replica_check.py: {error}", file=sys.stderr)
        return proof.EXIT_USAGE
    if plan.cpu:
        device, device_ids, backend = torch.device("cpu"), None, "gloo"
    else:
        torch.cuda.set_device(plan.local_rank)
        device, device_ids, backend = torch.device("cuda", plan.local_rank), [plan.local_rank], "nccl"
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        run_id, work, problem = _share_run(plan, rank, device)
        if problem:
            print(f"replica_check.py: {problem}", file=sys.stderr)
            return proof.EXIT_USAGE
        model = proof.build_model(device)
        base = proof.base_values(model)
        proof.reset_and_plant(model, base, ())
        ddp = DDP(model, device_ids=device_ids)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0)
        x = torch.ones(2, 4, device=device)

        records_file = proof.rank_file(work, run_id, rank)
        progress = proof.Progress()
        held = plan.hold_rank == rank
        sink = proof.ProofSink(
            tensorquill.JsonlSink(records_file), hold_from=plan.hold_from if held else None,
            never=plan.release == "never", release_file=work / "release", progress=progress,
            offsets=offset_path(records_file) if plan.mode == "server" else None,
        )
        observer = None
        if rank == 0:
            reader = proof.resolve_reader(plan.reader) if plan.mode == "files" else None
            observer = proof.Observer(plan, work, run_id, progress, lambda: sink.launch, reader)
            observer.start()

        durations: list[float] = []
        with tensorquill.init(
            model, run_id=run_id, sink=sink, every=1, auto_step=False, replica_group=dist.group.WORLD,
            replica_every=plan.replica_every, endpoint=plan.endpoint if plan.mode == "server" else None,
        ) as session:
            for step in range(plan.steps):
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = ddp(x).sum()
                loss.backward()
                session.log({"train/loss": loss})
                optimizer.step()
                proof.reset_and_plant(model, base, plan.plants_at(step, rank))
                session.step(step)
                if plan.step_seconds:
                    time.sleep(plan.step_seconds)
                durations.append(time.perf_counter() - started)
                progress.step = step
        sink.stop()
        _fingerprint_time(model, device, rank)
        _write_json(work / f"done.rank{rank}.json",
                    {"rank": rank, "steps_done": len(durations), "durations": durations, **sink.report()})
        if rank != 0:
            print(f"rank {rank}: trained {len(durations)} steps; rank 0 reads the verdict")
            return proof.EXIT_PASS

        reports = _wait_for_reports(work, plan.world, seconds=120)
        if plan.mode == "server":
            # The last status must come after the server acknowledged every rank's last observation,
            # and may list steps a later one leaves out: keep looking until both hold or time runs out.
            deadline = time.time() + plan.final_seconds
            last = _last_arrival(reports, plan)
            while time.time() < deadline:
                settled = last is not None and observer.log.last_t is not None and observer.log.last_t >= last
                if settled and not proof.check_result(observer.log.as_result(), plan, sink.launch):
                    break
                time.sleep(0.05)
            observer.stop()
        else:
            observer.stop()
            observer.look()  # the last look, after every rank has closed its file
        records = proof.run_records(work, run_id, plan.world)
        verdict = proof.evaluate(plan, observer.log, observer.seen_while, reports, sink.launch, records,
                                 reader_found=observer.reader is not None)
        print(f"run {run_id}:")
        for line in verdict.lines:
            print(line)
        _write_json(work / "verdict.json", {"code": verdict.code, "failures": verdict.failures})
        return verdict.code
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
