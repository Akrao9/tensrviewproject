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
- the two bits-only plants, `ulp` (one element moved by the smallest step of its dtype) and
  `negzero` (a 0.0 made -0.0), leave fingerprints `agree` and signatures `differ` on exactly the
  planted parameter, and raise no `replica_drift` (contract 5: a signature-only difference is
  evidence); one that raises an incident, or is not seen, fails with that reason;
- the clock (section 7): from the last required observation's arrival (the file holds the line, or
  the server acknowledged it) to the first look or status holding the step's verdict, at most one
  look or status interval plus `--latency-margin`. Both times and the difference are printed.

With `--hold-rank R`, rank R keeps training while its telemetry from `--hold-from` on is held back
in the driver's own sink (`replica_proof.ProofSink`), never in the SDK:

- `--release after-verdict` (a delayed rank): the step must read `insufficient` naming R while
  training goes on, no step on any rank may wait, and once released every held step is judged;
- `--release never` (a rank that never reports): the steps stay `insufficient` naming R, nothing is
  raised for them, and training runs to its end on every rank.

Which rank a planted step names (contract section 4a, D1), from the result's `attribution` key: in
files mode the top-level key of what `tensorquill.evidence:compare_replicas` returns, in server mode
the same key in the `replica` object of `watchdog.json`. With two ranks the default plants are two
ranks that disagree, and every entry of a planted parameter must read `inconclusive`, why
`two_ranks`, naming no rank; with three or more, one outlier among the rest, and the entry by
fingerprints must be `attributed` with `outliers` exactly the planted rank. The two bits-only
plants have no entry by fingerprints; their entry by signatures (as every planted parameter's whose
signatures differ) names the ranks apart from the one signature a strict majority holds, which must
be the planted rank, or reads `no_majority`. When no look carried the key and an entry is due, the
run fails saying "attribution not present"; it never passes for want of the key.

Two arms (`--arm`):

- `control`, the default: SGD with `lr=0`; every step resets every rank's weights to one base, then
  plants. At the end rank 0 prints, and writes into `verdict.json` under `healthy_signatures`, how
  many compared steps had no plant and on how many of those the signatures differed, by parameter.
  That count says whether equal bits read as equal on those devices, not whether training keeps
  them equal.
- `learning`: SGD with `--lr` (default 0.05), no reset and no plant; each rank trains on a batch of
  its own and DDP averages the gradients. Every rank checks that each parameter changed from its
  starting value, and the run fails if one did not or turned not finite. The same count, with the
  learning rate, each rank's weight change and the attribution entries by signatures, goes into
  `verdict.json` under `learning_signatures`, apart from the control's.

Either count is reported, not failed, unless `--healthy-mismatch fail`: how often signatures differ
on healthy steps is what the eight-GPU run measures.

Exit status: 0 PASS, 1 FAIL, 2 bad arguments, 3 NOT PROVEN (the reader could not be found).

How a run ended, when it left no `verdict.json` (lane AD6): each rank says on stderr each stage it
reaches (`replica_check rank R: <stage>`), and a rank that raises leaves `failed.rankR.json` in the
run's folder (or in `--out`, before the folder is known) with the stage, the error and its
traceback, then raises as before. Every rank but 0 stays in `main()` after its `done` file until
rank 0 has written `verdict.json` or its own `failed.rank0.json` (at most the time rank 0 may take
to judge), so that a rank that ends early with a nonzero status, which makes torchrun stop every
rank, cannot stop rank 0 before it judged. The verdict and every file the driver writes are written
by a scratch file and a rename.

    python -m torch.distributed.run --nproc-per-node=3 --master-addr 127.0.0.1 --master-port 29531 \\
        examples/replica_check.py --cpu --out runs/replica
    python -m torch.distributed.run --nproc-per-node=3 --master-addr 127.0.0.1 --master-port 29532 \\
        examples/replica_check.py --cpu --out runs/replica --arm learning --steps 40
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
import traceback
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
REPORTS_SECONDS = 120.0  # how long rank 0 waits for every rank's `done` file
VERDICT_MARGIN = 60.0  # what rank 0 may take, beyond its waits, to judge and write the verdict


class _Trace:
    """Where this rank is in the driver: said on stderr at each stage, and written with the error
    to `failed.rank<R>.json` when the rank raises. Neither ever raises."""

    def __init__(self, rank: object, folder: Path) -> None:
        self.rank, self.folder, self.stage = rank, folder, "starting"

    def reached(self, stage: str) -> None:
        self.stage = stage
        try:
            print(f"replica_check rank {self.rank}: {stage}", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001  a closed stream: the stage is still kept for the failure file
            pass

    def failed(self, error: BaseException) -> None:
        why = f"{type(error).__name__}: {error}"
        try:
            print(f"replica_check rank {self.rank}: failed at stage {self.stage!r}: {why}", file=sys.stderr,
                  flush=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            path = self.folder / f"failed.rank{self.rank}.json"
            _write_json(path, {"rank": self.rank, "stage": self.stage, "error": why, "at": time.time(),
                               "traceback": "".join(traceback.format_exception(error))})
        except Exception:  # noqa: BLE001  the stderr line above is what is left
            pass


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


def _wait_for_verdict(work: Path, seconds: float) -> str:
    """On every rank but 0, after its `done` file: until rank 0 has judged. Files only, no
    collective. "verdict", "rank 0 failed", or "timed out"."""
    deadline = time.time() + seconds
    while True:
        if (work / "verdict.json").exists():
            return "verdict"
        if (work / "failed.rank0.json").exists():
            return "rank 0 failed"
        if time.time() > deadline:
            return "timed out"
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
    trace = _Trace(os.environ.get("RANK", "unknown"), plan.out)
    try:
        return _run(plan, trace)
    except BaseException as error:
        trace.failed(error)
        raise


def _run(plan: proof.Plan, trace: _Trace) -> int:
    if plan.cpu:
        device, device_ids, backend = torch.device("cpu"), None, "gloo"
    else:
        torch.cuda.set_device(plan.local_rank)
        device, device_ids, backend = torch.device("cuda", plan.local_rank), [plan.local_rank], "nccl"
    trace.reached(f"starting the {backend} process group")
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        trace.rank = rank
        trace.reached("sharing the run")
        run_id, work, problem = _share_run(plan, rank, device)
        if problem:
            print(f"replica_check.py: {problem}", file=sys.stderr)
            return proof.EXIT_USAGE
        trace.folder = work
        model = proof.build_model(device)
        base = proof.base_values(model)
        proof.reset_and_plant(model, base, ())  # both arms start from one base on every rank
        ddp = DDP(model, device_ids=device_ids)
        learning = plan.arm == "learning"
        optimizer = torch.optim.SGD(ddp.parameters(), lr=plan.lr)
        x = torch.ones(2, 4, device=device)
        weights: dict = {}  # the learning arm's change of each parameter, by `proof.weight_change`
        before = {name: value.clone() for name, value in base.items()}

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
        trace.reached(f"training {plan.steps} steps in run {run_id}")
        with tensorquill.init(
            model, run_id=run_id, sink=sink, every=1, auto_step=False, replica_group=dist.group.WORLD,
            replica_every=plan.replica_every, endpoint=plan.endpoint if plan.mode == "server" else None,
        ) as session:
            for step in range(plan.steps):
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                if learning:
                    inputs, target = proof.learning_batch(rank, step, device)
                    loss = torch.nn.functional.mse_loss(ddp(inputs), target)
                else:
                    loss = ddp(x).sum()
                loss.backward()
                session.log({"train/loss": loss})
                optimizer.step()
                if learning:
                    before = proof.weight_change(model, base, before, weights)
                else:
                    proof.reset_and_plant(model, base, plan.plants_at(step, rank))
                session.step(step)
                if plan.step_seconds:
                    time.sleep(plan.step_seconds)
                durations.append(time.perf_counter() - started)
                progress.step = step
            trace.reached(f"trained {len(durations)} steps; closing the session")
        trace.reached("session closed")
        sink.stop()
        _fingerprint_time(model, device, rank)
        done = {"rank": rank, "steps_done": len(durations), "durations": durations, **sink.report()}
        if learning:
            done["weights"] = weights
        _write_json(work / f"done.rank{rank}.json", done)
        trace.reached("done file written")
        if rank != 0:
            print(f"rank {rank}: trained {len(durations)} steps; rank 0 reads the verdict", flush=True)
            # Not gone before rank 0 has judged: a rank that ends with a nonzero status (at its own
            # exit, say) makes torchrun stop every rank, rank 0 among them (lane AD6).
            bound = REPORTS_SECONDS + plan.final_seconds + VERDICT_MARGIN
            trace.reached(f"waiting up to {bound:g} s for rank 0's verdict")
            trace.reached(f"rank 0's verdict: {_wait_for_verdict(work, bound)}")
            return proof.EXIT_PASS

        trace.reached(f"waiting for the done files of {plan.world} ranks")
        reports = _wait_for_reports(work, plan.world, seconds=REPORTS_SECONDS)
        trace.reached(f"done files of ranks {sorted(reports)} of {plan.world}; judging")
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
        sys.stdout.flush()  # a pipe: the lines are out before the file says the run was judged
        _write_json(work / "verdict.json", verdict.document())
        trace.reached(f"verdict written (exit {verdict.code})")
        return verdict.code
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
