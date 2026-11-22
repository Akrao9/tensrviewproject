"""A bounded, real collective hang, and what TensorQuill reads from PyTorch's Flight Recorder dumps of it.

Lane AI-3, Nov 22, 2026. The author's terms: "Give the hang test an external timeout and collect actual
dumps."

The driver (this file, run without `--worker`) starts a small job under torchrun. Every rank trains a
tiny model with a TensorQuill session (`every=1`, one `.jsonl` file per rank) and issues exactly one
collective per step on the default group, an `all_reduce` of its flattened gradients; nothing else
issues a collective on that group. At step `--skip-step`, rank `--skip-rank` does not enter that
`all_reduce` and sleeps instead, so the others wait in it for real. The driver waits a bounded time,
then ends the job itself, whatever state it is in: torchrun gives each worker a session of its own
(`start_new_session=True` in torch's elastic `subprocess_handler`), so killing torchrun's group does not
reach them; each rank writes its pid at start, and the driver sends SIGTERM and then SIGKILL to
torchrun's group and to every recorded pid's group, and fails the run if any of them is still alive.
Then it runs `python -m tensorquill <capture of a waiting rank> --flight-recorder <dump dir>` on the
dumps PyTorch wrote.

The independent expectation is the plan: which rank skipped, which collective, at which step. It
passes only when TensorQuill's evidence lists `collective_hang` with

- `collective` the planned one (`all_reduce`) and `collective_seq_id` the planned one (step K's
  collective is the group's (K+1)th: one per step from step 0, sequence numbers from 1);
- `missing_ranks` exactly `[skip_rank]` (the rank at fault) and `expected_ranks` minus that exactly the
  other ranks (the ranks that were waiting);
- every rank of the job among `dumped_ranks`: without a rank's dump the analyzer cannot say whether it
  entered, so without every dump the answer is NOT PROVEN, not judged;
- `step` the last step the waiting rank has on disk, K-2. The dumps do not know the training step; the
  reader dates the incident by the capture, and the capture keeps its newest step back until the next
  step ends (`capture.py`, at `_flush(hold_newest=True)`: "A run that hangs or is killed in the next
  step has one step fewer on disk"), so steps 0 to K-1 ran and 0 to K-2 are written. That holds for a
  rank that is killed (SIGTERM, SIGKILL, or NCCL's abort), which is how this job ends; a session closed
  by an exception (`with session:`) would write K-1, and the session has no exit hook of its own.

How the dumps are written, from the installed torch 2.15 and the sibling `pytorch` source:

- `--dump on-timeout` (the default with NCCL, the GPU run): PyTorch's own trigger. The driver sets
  `TORCH_FR_BUFFER_SIZE`, `TORCH_FR_DUMP_TEMP_FILE=<dumps>/fr_trace_rank_` and their former names
  `TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`, with `TORCH_NCCL_ENABLE_MONITORING=1`,
  `TORCH_NCCL_DUMP_ON_TIMEOUT=1` and `TORCH_FR_DUMP_ON_TIMEOUT=1`, and gives the process group a short
  timeout (`--pg-timeout`). ProcessGroupNCCL's watchdog sees the collective time out and writes
  `<prefix><rank>`. The script writes no dump itself.
- `--dump on-request` (the default with Gloo, the CPU dry run): with Gloo there is no dump on timeout.
  Gloo has no abort hooks (`FlightRecorderHook.cpp`: "Abort hooks are optional -- gloo has none"), and
  the dump-on-failure trigger is an abort hook, so a Gloo timeout writes nothing. Gloo does record its
  collectives (in torch 2.15 natively, `gloo` in `_FR_SELF_RECORDING_BACKENDS`), and PyTorch's own writer
  `torch._C._distributed_c10d._dump_fr_trace_file(rank)` writes the same `<prefix><rank>` pickle. Once
  the hang is established, the driver leaves a `dump-now` file and a thread in each rank calls that
  writer. The dumps are PyTorch's, but the moment they are written is the script's: this does not
  show the automatic dump path.

What the CPU dry run can and cannot show, on torch 2.15 (Nov 22, 2026): the dumps are written, but
PyTorch's own analyzer cannot read them. Torch 2.15's native Gloo recording publishes the group's
membership under an empty name (`pg_config` `{'': ...}`) while the entries name the group `'0'`, and
the analyzer's `build_collectives` raises `KeyError: ('0', 0)`. Upstream removed Gloo's native recording
and records it through `FlightRecorderHook` with the group's real name in e8f48e8 (#203544, Nov 6, 2026),
after 2.15. The driver runs PyTorch's analyzer (`torch.distributed.flight_recorder.fr_trace`) on the
dumps itself, as a separate check: when PyTorch's own analyzer cannot read them the run is NOT PROVEN,
whatever TensorQuill did. TensorQuill failing where PyTorch's analyzer reads them is a FAIL.

Exit status: 0 PASS, 1 FAIL, 2 bad arguments or a missing requirement, 3 NOT PROVEN (no dump was
written, or PyTorch's analyzer cannot read the dumps). The result is `<out>/result.json`.

    # GPU (2 or more GPUs, NCCL):
    python sdk/examples/flight_recorder_hang_check.py --out runs/fr-hang --nproc 2
    # CPU dry run (Gloo, about 30 s):
    python sdk/examples/flight_recorder_hang_check.py --cpu --out runs/fr-hang-cpu
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

PASS, FAIL, USAGE, NOT_PROVEN = 0, 1, 2, 3
PREFIX = "fr_trace_rank_"
COLLECTIVE = "all_reduce"


# --- the worker: one rank of the job, started by torchrun -----------------------------------------


def _touch(path: Path, document: dict | None = None) -> None:
    scratch = path.with_name(f".{path.name}.tmp")
    scratch.write_text(json.dumps(document or {}), encoding="utf-8")
    os.replace(scratch, path)


def _dump_on_request(work: Path, rank: int, backend: str) -> None:
    """Wait for the driver's `dump-now` file, then have PyTorch write this rank's dump."""
    import torch

    while not (work / "dump-now").exists():
        time.sleep(0.05)
    try:
        if backend == "gloo":
            # PyTorch's own writer, to <TORCH_FR_DUMP_TEMP_FILE><rank>; the GIL is released inside.
            torch._C._distributed_c10d._dump_fr_trace_file(rank)
        else:
            # NCCL: the same pickle PyTorch's watchdog writes, written to the same name.
            data = torch._C._distributed_c10d._dump_nccl_trace()
            target = Path(os.environ["TORCH_FR_DUMP_TEMP_FILE"] + str(rank))
            target.write_bytes(data)
        _touch(work / f"dumped.rank{rank}.json", {"rank": rank, "at": time.time()})
    except Exception as error:  # noqa: BLE001  said to the driver, which decides
        _touch(work / f"dump-failed.rank{rank}.json", {"rank": rank, "error": f"{type(error).__name__}: {error}"})


def worker(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    import tensorquill

    work = Path(args.work)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    _touch(work / "pids" / f"rank{rank}.json", {"rank": rank, "pid": os.getpid()})
    if args.backend == "nccl":
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.pg_timeout))
    if args.dump == "on-request":
        threading.Thread(target=_dump_on_request, args=(work, rank, args.backend), daemon=True).start()

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    records = work / "captures" / f"{args.run_id}.rank{rank}.jsonl"
    session = tensorquill.init(model, run_id=args.run_id, sink=tensorquill.JsonlSink(records), every=1)
    generator = torch.Generator().manual_seed(rank)
    for step in range(args.steps):
        inputs = torch.randn(4, 8, generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).pow(2).mean()
        loss.backward()
        session.log({"train/loss": loss})
        flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        if step == args.skip_step:
            if rank == args.skip_rank:
                # The planted fault: this rank does not enter the collective the others enter.
                _touch(work / f"skipped.rank{rank}.json", {"rank": rank, "step": step, "at": time.time()})
                while True:
                    time.sleep(1.0)
            _touch(work / f"entered.rank{rank}.json", {"rank": rank, "step": step, "at": time.time()})
        dist.all_reduce(flat)  # the one collective of the step on the default group
        if device.type == "cuda":
            torch.cuda.synchronize(device)  # NCCL returns at once; wait here, as a gradient read would
        offset = 0
        for p in model.parameters():
            n = p.grad.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad) / world)
            offset += n
        optimizer.step()  # ends TensorQuill's step (its optimizer hook)
    # Not reached when the plan's step is within --steps: the job is ended by the driver.
    _touch(work / f"finished.rank{rank}.json", {"rank": rank})
    session.close()
    dist.destroy_process_group()


# --- the driver -------------------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); `ps` tells it apart.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _signal_group(pid: int, sig: int) -> None:
    for send in (lambda: os.killpg(pid, sig), lambda: os.kill(pid, sig)):
        try:
            send()
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _end_job(launcher: subprocess.Popen, pids: list[int], grace: float) -> dict[str, Any]:
    """End torchrun and every worker: SIGTERM to each group, then SIGKILL to what is left."""
    everyone = [launcher.pid, *pids]
    for pid in everyone:
        _signal_group(pid, signal.SIGTERM)
    deadline = time.time() + grace
    while time.time() < deadline and any(_alive(p) for p in everyone):
        launcher.poll()
        time.sleep(0.1)
    killed = [p for p in everyone if _alive(p)]
    for pid in killed:
        _signal_group(pid, signal.SIGKILL)
    try:
        launcher.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.time() + 10
    while time.time() < deadline and any(_alive(p) for p in everyone):
        time.sleep(0.1)
    return {"sigkilled": killed, "left_running": [p for p in everyone if _alive(p)],
            "launcher_returncode": launcher.returncode}


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pytorch_analyzer(dumps: Path, env: dict[str, str]) -> dict[str, Any]:
    """PyTorch's own analyzer on the dumps, as its command line runs it."""
    command = [sys.executable, "-m", "torch.distributed.flight_recorder.fr_trace", str(dumps),
               "--allow-incomplete-ranks", "-p", PREFIX]
    done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or [""]
    return {"command": command, "returncode": done.returncode, "last_line": tail[0][:400]}


def _check(plan: dict[str, Any], evidence: dict[str, Any]) -> tuple[list[str], list[str], dict | None]:
    """The plan against TensorQuill's evidence: the failures, why it cannot be judged, and the incident.

    With a rank's dump missing (the analyzer runs with `--allow-incomplete-ranks`), its answer cannot say
    whether that rank entered the collective: on constructed dumps without the skipped rank's file it
    lists no missing rank at all (`test_a_rank_with_no_dump_is_not_taken_for_the_rank_at_fault`). So the
    answer is judged only when every rank of the job dumped."""
    hangs = [i for i in evidence.get("incidents", []) if i.get("kind") == "collective_hang"]
    if not hangs:
        kinds = sorted({str(i.get("kind")) for i in evidence.get("incidents", [])})
        return [f"no collective_hang in the evidence (incidents: {kinds or 'none'})"], [], None
    hit = hangs[0]
    absent = sorted(set(range(plan["nproc"])) - set(hit.get("dumped_ranks") or []))
    if absent:
        return [], [f"ranks {absent} wrote no dump, so whether they entered the collective cannot be read from the "
                    f"dumps (missing_ranks {hit.get('missing_ranks')})"], hit
    failures = []
    if hit.get("collective") != plan["collective"]:
        failures.append(f"collective {hit.get('collective')!r}, planned {plan['collective']!r}")
    if hit.get("collective_seq_id") != plan["collective_seq_id"]:
        failures.append(f"collective_seq_id {hit.get('collective_seq_id')!r}, planned {plan['collective_seq_id']}")
    if hit.get("missing_ranks") != [plan["skip_rank"]]:
        failures.append(f"missing_ranks {hit.get('missing_ranks')!r}, planned [{plan['skip_rank']}] (the rank at fault)")
    waiting = sorted(set(hit.get("expected_ranks") or []) - set(hit.get("missing_ranks") or []))
    if waiting != plan["waiting_ranks"]:
        failures.append(f"waiting ranks (expected_ranks minus missing_ranks) {waiting}, planned {plan['waiting_ranks']}")
    if hit.get("step") != plan["last_recorded_step"]:
        failures.append(f"step {hit.get('step')!r}, planned {plan['last_recorded_step']} (the last step on disk)")
    return failures, [], hit


def drive(args: argparse.Namespace) -> int:
    import torch

    backend = "gloo" if args.cpu else "nccl"
    dump = args.dump or ("on-request" if backend == "gloo" else "on-timeout")
    if backend == "nccl":
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.nproc:
            print(f"flight_recorder_hang_check: needs {args.nproc} CUDA devices for NCCL "
                  f"(found {torch.cuda.device_count() if torch.cuda.is_available() else 0}); use --cpu for the "
                  "Gloo dry run")
            return USAGE
    if not 0 <= args.skip_rank < args.nproc or args.nproc < 2:
        print("flight_recorder_hang_check: --nproc must be at least 2 and --skip-rank one of its ranks")
        return USAGE
    if not 2 <= args.skip_step < args.steps:
        print("flight_recorder_hang_check: --skip-step must be at least 2 (a step must be on disk first) and below --steps")
        return USAGE
    pg_timeout = args.pg_timeout or (60.0 if dump == "on-timeout" else 3600.0)
    # NCCL's watchdog needs the timeout and then time to dump; the bound must be past both.
    bound = args.bound or ((pg_timeout + args.dump_wait + 60.0) if dump == "on-timeout" else 120.0)

    run_id = f"fr-hang-{secrets.token_hex(4)}"
    work = args.out / run_id
    for sub in ("captures", "dumps", "pids"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    dumps = work / "dumps"
    waiting = sorted(set(range(args.nproc)) - {args.skip_rank})
    plan = {
        "skip_rank": args.skip_rank, "skip_step": args.skip_step, "collective": COLLECTIVE,
        "collective_seq_id": args.skip_step + 1, "waiting_ranks": waiting,
        "last_recorded_step": args.skip_step - 2, "nproc": args.nproc,
    }
    env = dict(os.environ)
    env.update({
        "TORCH_FR_BUFFER_SIZE": "2000", "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
        "TORCH_FR_DUMP_TEMP_FILE": str(dumps / PREFIX), "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(dumps / PREFIX),
        "TORCH_NCCL_ENABLE_MONITORING": "1", "TORCH_NCCL_DUMP_ON_TIMEOUT": "1", "TORCH_FR_DUMP_ON_TIMEOUT": "1",
        "PYTHONUNBUFFERED": "1",
    })
    result: dict[str, Any] = {
        "run_id": run_id, "backend": backend, "dump": dump, "torch": torch.__version__,
        "pg_timeout_seconds": pg_timeout, "bound_seconds": bound, "plan": plan,
        "dump_written_by": ("PyTorch's ProcessGroupNCCL watchdog on the collective's timeout" if dump == "on-timeout"
                            else "PyTorch's Flight Recorder writer, called by the script once the hang was established "
                                 "(not PyTorch's dump on timeout)"),
    }
    print(f"flight_recorder_hang_check: run {run_id} in {work}; {backend}, {args.nproc} ranks, rank {args.skip_rank} "
          f"skips the {COLLECTIVE} of step {args.skip_step}; dumps {dump}; bound {bound:g} s")

    command = [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={args.nproc}",
               "--master-addr", "127.0.0.1", "--master-port", str(args.master_port), str(Path(__file__).resolve()),
               "--worker", "--work", str(work), "--run-id", run_id, "--backend", backend, "--dump", dump,
               "--skip-rank", str(args.skip_rank), "--skip-step", str(args.skip_step), "--steps", str(args.steps),
               "--pg-timeout", str(pg_timeout)]
    log = (work / "job.log").open("w", encoding="utf-8")
    started = time.time()
    launcher = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def markers(name: str) -> list[int]:
        return sorted(r for r in range(args.nproc) if (work / f"{name}.rank{r}.json").exists())

    hang_at = None
    requested_at = None
    ended_by = "bound"
    while time.time() - started < bound:
        if launcher.poll() is not None:
            ended_by = f"job exited on its own (status {launcher.returncode})"
            break
        if hang_at is None and markers("skipped") == [args.skip_rank] and markers("entered") == waiting:
            hang_at = time.time()
        if markers("finished"):
            break
        if hang_at is not None and dump == "on-request":
            if requested_at is None and time.time() - hang_at >= args.hold:
                _touch(work / "dump-now")
                requested_at = time.time()
            if requested_at is not None and (
                len(markers("dumped")) + len(markers("dump-failed")) == args.nproc
                or time.time() - requested_at > args.dump_wait
            ):
                ended_by = "dumps requested and written (or their wait ran out)"
                break
        if dump == "on-timeout" and hang_at is not None:
            written = [p for p in dumps.iterdir() if p.name.startswith(PREFIX)]
            if len(written) == args.nproc and time.time() - max(p.stat().st_mtime for p in written) > 5:
                ended_by = "every rank's dump written"
                break
        time.sleep(0.2)
    elapsed = time.time() - started
    finished_early = markers("finished")
    # Whatever state the job is in, it ends here.
    pids = [d["pid"] for r in range(args.nproc) if (d := _read(work / "pids" / f"rank{r}.json"))]
    ending = _end_job(launcher, pids, grace=args.grace)
    log.close()
    dump_files = sorted(p.name for p in dumps.iterdir() if p.name.startswith(PREFIX))
    result.update({
        "hang_established": hang_at is not None,
        "hang_seconds_before_end": None if hang_at is None else round(time.time() - hang_at, 1),
        "ended_by": ended_by, "elapsed_seconds": round(elapsed, 1), "worker_pids": pids, **ending,
        "dump_files": dump_files,
        "dump_failures": [_read(work / f"dump-failed.rank{r}.json") for r in markers("dump-failed")],
    })
    lines = [f"hang established: {hang_at is not None} (rank {args.skip_rank} skipped, ranks {waiting} entered)",
             f"job ended by the driver after {elapsed:.1f} s ({ended_by}); SIGKILL needed for {len(ending['sigkilled'])}"
             f" of {len(pids) + 1} processes; still running: {ending['left_running'] or 'none'}",
             f"dumps written: {dump_files or 'none'} ({result['dump_written_by']})"]
    failures: list[str] = []
    not_proven: list[str] = []
    if finished_early:
        failures.append(f"ranks {finished_early} finished: the collective did not hang")
    if hang_at is None:
        failures.append("the hang was not established (no skip and entry markers from every rank)")
    if ending["left_running"]:
        failures.append(f"processes still running after the driver ended the job: {ending['left_running']}")
    if len(pids) != args.nproc:
        failures.append(f"{len(pids)} of {args.nproc} ranks wrote a pid file")
    capture = work / "captures" / f"{run_id}.rank{waiting[0]}.jsonl"
    if not failures and not dump_files:
        not_proven.append("no Flight Recorder dump was written")
    if not failures and dump_files:
        result["pytorch_analyzer"] = analyzer = _pytorch_analyzer(dumps, env)
        lines.append(f"PyTorch's analyzer on the dumps: status {analyzer['returncode']}"
                     + (f" ({analyzer['last_line']})" if analyzer["returncode"] else ""))
        evidence_path = work / "evidence.json"
        tq = [sys.executable, "-m", "tensorquill", str(capture), "--flight-recorder", str(dumps),
              "--flight-recorder-prefix", PREFIX, "--detectors", "python", "-o", str(evidence_path)]
        done = subprocess.run(tq, capture_output=True, text=True, env=env, timeout=300)
        (work / "tensorquill.log").write_text(done.stdout + done.stderr, encoding="utf-8")
        result["tensorquill"] = {"command": tq, "returncode": done.returncode,
                                 "last_line": ((done.stderr or done.stdout).strip().splitlines()[-1:] or [""])[0][:400]}
        evidence = _read(evidence_path) if done.returncode == 0 else None
        lines.append(f"python -m tensorquill: status {done.returncode}"
                     + ("" if done.returncode == 0 else f" ({result['tensorquill']['last_line']})"))
        if analyzer["returncode"] != 0:
            not_proven.append("PyTorch's own Flight Recorder analyzer cannot read these dumps "
                              f"({analyzer['last_line']}); nothing TensorQuill reports from them can be judged")
            if done.returncode != 0:
                result["tensorquill_defect"] = ("python -m tensorquill ended with an error instead of listing "
                                                "flight_recorder under not_checked")
                lines.append(f"also: {result['tensorquill_defect']}")
        elif evidence is None:
            failures.append(f"python -m tensorquill gave no evidence (status {done.returncode}) on dumps PyTorch's "
                            "analyzer reads")
        else:
            result["not_checked"] = [n for n in evidence.get("not_checked", []) if n.get("check") == "flight_recorder"]
            if result["not_checked"]:
                failures.append(f"TensorQuill did not check the dumps: {result['not_checked'][0].get('reason')}")
            else:
                found, unjudged, hit = _check(plan, evidence)
                result["incident"] = hit
                failures.extend(found)
                not_proven.extend(unjudged)
                if hit:
                    lines.append(f"TensorQuill: collective_hang on {hit.get('collective')} seq {hit.get('collective_seq_id')}, "
                                 f"missing (at fault) {hit.get('missing_ranks')}, expected {hit.get('expected_ranks')}, "
                                 f"step {hit.get('step')}")
    status = FAIL if failures else NOT_PROVEN if not_proven else PASS
    result.update({"status": {PASS: "PASS", FAIL: "FAIL", NOT_PROVEN: "NOT PROVEN"}[status],
                   "failures": failures, "not_proven": not_proven})
    _touch(work / "result.json", result)
    _touch(args.out / "result.json", result)
    for line in lines:
        print(f"  {line}")
    for why in failures:
        print(f"  FAIL: {why}")
    for why in not_proven:
        print(f"  NOT PROVEN: {why}")
    print(f"flight_recorder_hang_check: {result['status']} ({work / 'result.json'})")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("runs/flight-recorder-hang"))
    parser.add_argument("--cpu", action="store_true", help="Gloo on CPU (the dry run) instead of NCCL on GPUs")
    parser.add_argument("--nproc", type=int, default=3)
    parser.add_argument("--skip-rank", type=int, default=1)
    parser.add_argument("--skip-step", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dump", choices=("on-timeout", "on-request"),
                        help="who triggers the dump (default: on-timeout with NCCL, on-request with Gloo)")
    parser.add_argument("--pg-timeout", type=float,
                        help="process group timeout, s (default: 60 with on-timeout, 3600 with on-request)")
    parser.add_argument("--bound", type=float, help="the external timeout, s (default: pg timeout + dump wait + 60, "
                        "or 120 with on-request)")
    parser.add_argument("--hold", type=float, default=3.0, help="s the hang must last before dumps are requested")
    parser.add_argument("--dump-wait", type=float, default=30.0, help="s to wait for the dumps")
    parser.add_argument("--grace", type=float, default=5.0, help="s between SIGTERM and SIGKILL")
    parser.add_argument("--master-port", type=int, default=29641)
    # The worker's arguments, given by the driver.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--work", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return PASS
    return drive(args)


if __name__ == "__main__":
    sys.exit(main())
