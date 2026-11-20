"""The parts of `replica_check.py` that are not the training loop, kept apart so they can be tested
on synthetic records and status files without a process group.

What is here, in the order the driver uses it:

- `validate`: the arguments and the launcher's environment, for any world size of two or more.
- `Plant`, `build_model`, `reset_and_plant`: the divergence the driver plants on one rank, one
  parameter and one step (`add`, `neg` or `swap`, the three it has always planted).
- `ProofSink`: wraps each rank's `JsonlSink`. It logs when each line reaches the file, and, on the
  held rank, holds that rank's lines back (released later, or never). The suppression lives here,
  in the driver's own files, never in the SDK. In server mode it also logs, on a thread of its own,
  each change of the stream sender's `.offset` file: when the server acknowledged each byte.
- `VerdictLog`: every result object the driver saw (the server's `replica` in `watchdog.json`, or
  the file reader's), kept by launch, group and step, with when each state was first seen. A
  server may bound how many steps it lists, so nothing is judged from the last look alone.
- `check_result`, `check_history`, `check_no_wait`, `check_latency`, `check_kinds`: the assertions.
  Each returns a list of failures, empty when it holds; none asserts which rank drifted (D1 is held).

The contract is `REPLICA_OBSERVATION_CONTRACT.md` at the repository root.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

# --- The reader of local rank files (lane Z2) -----------------------------------------------
#
# Assumed, since lane Z2's function was not named when this was written: a function in
# `tensorquill.evidence` that takes one run's records (every rank's lines, in file order) and
# returns the object of contract section 8 (`{"v", "tolerance", "steps", "replica_drift"}`, with
# `legacy` when there is one), spelled as the fixtures spell it. `--reader module:function` names
# another, as does the environment variable below (read by the driver and by its tests alike, so a
# renamed reader is one setting on the trial merge, not a silent skip). A result wrapped as
# `{"replica": {...}}` is unwrapped. When the function cannot be found the run ends NOT PROVEN
# (exit 3), never PASS.
READER_ENV = "TENSORQUILL_REPLICA_READER"
DEFAULT_READER = os.environ.get(READER_ENV) or "tensorquill.evidence:replica_result"
ACK_POLL = 0.005  # how often each rank looks at its sender's offset file (server mode)

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_PROVEN = 0, 1, 2, 3
TOLERANCE = 1e-6

# --- The model and the planted divergence ----------------------------------------------------

PARAM_NAMES = ("0.weight", "0.bias", "1.weight")
KINDS = ("add", "neg", "swap")


@dataclass(frozen=True)
class Plant:
    rank: int
    step: int
    param: str
    kind: str

    def __str__(self) -> str:
        return f"{self.kind} on {self.param} of rank {self.rank} at step {self.step}"


def parse_plant(text: str) -> Plant:
    """`RANK:STEP:PARAM:KIND`, as `--plant 2:3:1.weight:neg`."""
    parts = text.split(":")
    if len(parts) != 4:
        raise ValueError(f"--plant {text!r}: expected RANK:STEP:PARAM:KIND")
    rank, step, param, kind = parts
    try:
        return Plant(int(rank), int(step), param, kind)
    except ValueError:
        raise ValueError(f"--plant {text!r}: RANK and STEP are whole numbers") from None


def default_plants(world: int) -> list[Plant]:
    """The three the driver has always planted, each on one rank: with two ranks, two ranks that
    disagree; with three or more, one outlier among the others (the controls D1 will need)."""
    last = world - 1
    return [Plant(last, 1, "0.weight", "add"), Plant(1, 3, "1.weight", "neg"), Plant(last, 5, "1.weight", "swap")]


def build_model(device: Any) -> Any:
    """Two layers, three parameters, so a check that names the planted parameter can be wrong."""
    import torch

    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 1, bias=False)).to(device)


def base_values(model: Any) -> dict[str, Any]:
    """Fixed, distinct, non-zero values, the same on every rank: a sign flip changes the sum and
    not the norm, and a swap of two elements changes neither, only the projection."""
    import torch

    out = {}
    for offset, (name, param) in enumerate(model.named_parameters()):
        n = param.numel()
        values = (torch.arange(1, n + 1, dtype=torch.float32) + offset * 0.5) / n
        out[name] = values.reshape(param.shape).to(param.device)
    return out


def reset_and_plant(model: Any, base: Mapping[str, Any], plants: Iterable[Plant]) -> None:
    """Every rank's weights back to the base, then this rank's plants for this step."""
    import torch

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in base.items():
            params[name].copy_(value)
        for plant in plants:
            flat = params[plant.param].view(-1)
            if plant.kind == "add":
                flat[0].add_(0.25)
            elif plant.kind == "neg":
                flat[0].neg_()
            elif plant.kind == "swap":
                flat[[1, 2]] = flat[[2, 1]].clone()


# --- Arguments --------------------------------------------------------------------------------


class UsageError(ValueError):
    pass


@dataclass
class Plan:
    world: int
    rank: int
    local_rank: int
    mode: str  # "files" or "server"
    out: Path
    steps: int
    replica_every: int
    step_seconds: float
    plants: list[Plant]
    hold_rank: int | None = None
    hold_from: int | None = None
    release: str | None = None  # "after-verdict" or "never"
    hold_steps: int = 3
    interval: float = 0.5  # one look of the file reader, or one server status interval
    margin: float = 1.0
    wait_slack: float = 0.5
    endpoint: str | None = None
    store: Path | None = None
    reader: str = DEFAULT_READER
    cpu: bool = True
    run_id: str | None = None
    final_seconds: float = 60.0

    @property
    def group(self) -> list[int]:
        return list(range(self.world))

    @property
    def scheduled(self) -> list[int]:
        return [s for s in range(self.steps) if (s + 1) % self.replica_every == 0]

    def plants_at(self, step: int, rank: int | None = None) -> list[Plant]:
        return [p for p in self.plants if p.step == step and (rank is None or p.rank == rank)]

    def held(self, step: int) -> bool:
        """Whether this step's observation of the held rank is held back."""
        return self.hold_rank is not None and step >= self.hold_from

    def never_judged(self, step: int) -> bool:
        return self.held(step) and self.release == "never"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replica_check.py",
        description="Check TensorQuill's cross-rank replica comparison on any number of ranks (contract section 7).",
    )
    p.add_argument("--cpu", action="store_true", help="use Gloo on the CPU instead of CUDA and NCCL")
    p.add_argument("--out", type=Path, default=Path("runs/replica"),
                   help="a folder for this check; each run writes <out>/<run_id>/<run_id>.rank<N>.jsonl"
                        " (default: runs/replica)")
    p.add_argument("--mode", choices=("files", "server"), default="files",
                   help="where the verdict comes from: the Python reader of the local rank files, or a server's"
                        " watchdog.json (needs --endpoint, --store and --status-interval)")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--replica-every", type=int, default=1)
    p.add_argument("--step-seconds", type=float, default=0.2,
                   help="sleep in each training step, so the run spans several looks or status intervals")
    p.add_argument("--plant", action="append", default=None, metavar="RANK:STEP:PARAM:KIND",
                   help=f"plant a divergence (PARAM one of {', '.join(PARAM_NAMES)}; KIND one of {', '.join(KINDS)});"
                        " repeat for more; default: three plants on steps 1, 3 and 5")
    p.add_argument("--no-plants", action="store_true", help="plant nothing: every step must read clean")
    p.add_argument("--hold-rank", type=int, help="hold back this rank's telemetry (it keeps training)")
    p.add_argument("--hold-from", type=int, default=2, help="the first step whose records are held")
    p.add_argument("--release", choices=("after-verdict", "never"), default="after-verdict",
                   help="after-verdict: release once the step reads insufficient and training has gone"
                        " --hold-steps further (a delayed rank); never: a rank that never reports")
    p.add_argument("--hold-steps", type=int, default=3)
    p.add_argument("--look-seconds", type=float, default=0.5, help="files mode: how often the reader looks")
    p.add_argument("--status-interval", type=float,
                   help="server mode: the server's --watchdog-seconds (the driver cannot read it; give the same value)")
    p.add_argument("--latency-margin", type=float, default=1.0,
                   help="seconds allowed above one look or status interval before the clock fails")
    p.add_argument("--wait-slack", type=float, default=0.5,
                   help="seconds a training step may take above --step-seconds while a rank is held")
    p.add_argument("--endpoint", help="server mode: the server's grpc:// address")
    p.add_argument("--store", type=Path, help="server mode: the server's --store folder (on this host)")
    p.add_argument("--reader", default=DEFAULT_READER, help="files mode: the reader, as module:function")
    p.add_argument("--run-id", help="default: a new one made by rank 0")
    p.add_argument("--final-seconds", type=float, default=60.0,
                   help="server mode: how long to wait after training for the last status")
    return p


def validate(argv: list[str] | None, env: Mapping[str, str], cuda_devices: int | None = None) -> Plan:
    """The arguments and the launcher's environment, checked before any process group is made.
    Raises `UsageError` with what to change."""
    args = build_parser().parse_args(argv)
    try:
        world = int(env["WORLD_SIZE"])
        rank = int(env["RANK"])
        local_rank = int(env.get("LOCAL_RANK", rank))
    except (KeyError, ValueError):
        raise UsageError("launch with torchrun: python -m torch.distributed.run --nproc-per-node=N ..."
                         " (WORLD_SIZE and RANK are not set)") from None
    if world < 2:
        raise UsageError(f"WORLD_SIZE is {world}: a group of one rank compares nothing; launch two or more")
    local_world = env.get("LOCAL_WORLD_SIZE")
    if local_world is not None and int(local_world) != world:
        raise UsageError(f"LOCAL_WORLD_SIZE {local_world} is not WORLD_SIZE {world}: the driver reads every"
                         " rank's files and clocks on one host, so run it on one node")
    if not args.cpu:
        if cuda_devices is None:
            raise UsageError("no CUDA: pass --cpu for Gloo")
        if cuda_devices < world:
            raise UsageError(f"{world} ranks need {world} visible CUDA GPUs; {cuda_devices} are visible (or pass --cpu)")
    if args.steps < 2:
        raise UsageError("--steps must be at least 2")
    if args.replica_every < 1:
        raise UsageError("--replica-every must be at least 1")
    if args.step_seconds < 0 or args.look_seconds <= 0 or args.latency_margin < 0 or args.wait_slack < 0:
        raise UsageError("--step-seconds, --latency-margin and --wait-slack are 0 or more; --look-seconds above 0")
    if args.no_plants and args.plant:
        raise UsageError("--no-plants and --plant are opposites")
    try:
        plants = [] if args.no_plants else [parse_plant(t) for t in args.plant] if args.plant else default_plants(world)
    except ValueError as error:
        raise UsageError(str(error)) from None
    scheduled = {s for s in range(args.steps) if (s + 1) % args.replica_every == 0}
    for plant in plants:
        if not 0 <= plant.rank < world:
            raise UsageError(f"plant {plant}: rank {plant.rank} is not one of the {world} ranks")
        if plant.step not in scheduled:
            raise UsageError(f"plant {plant}: step {plant.step} is not a scheduled step below --steps {args.steps}"
                             f" (with --replica-every {args.replica_every})")
        if plant.param not in PARAM_NAMES:
            raise UsageError(f"plant {plant}: the model's parameters are {', '.join(PARAM_NAMES)}")
        if plant.kind not in KINDS:
            raise UsageError(f"plant {plant}: the kinds are {', '.join(KINDS)}")
    plan = Plan(
        world=world, rank=rank, local_rank=local_rank, mode=args.mode, out=args.out, steps=args.steps,
        replica_every=args.replica_every, step_seconds=args.step_seconds, plants=plants,
        hold_steps=args.hold_steps, margin=args.latency_margin, wait_slack=args.wait_slack,
        endpoint=args.endpoint, store=args.store, reader=args.reader, cpu=args.cpu, run_id=args.run_id,
        final_seconds=args.final_seconds,
    )
    if args.hold_rank is not None:
        if not 0 <= args.hold_rank < world:
            raise UsageError(f"--hold-rank {args.hold_rank} is not one of the {world} ranks")
        if args.hold_from not in scheduled:
            raise UsageError(f"--hold-from {args.hold_from} is not a scheduled step below --steps {args.steps}")
        if args.hold_steps < 1:
            raise UsageError("--hold-steps must be at least 1")
        if args.release == "after-verdict" and args.hold_from + args.hold_steps >= args.steps - 1:
            raise UsageError(f"--steps {args.steps} leaves no training after the release: give at least"
                             f" {args.hold_from + args.hold_steps + 2}")
        if args.step_seconds <= 0:
            raise UsageError("a held rank needs --step-seconds above 0, or training ends before any look")
        plan.hold_rank, plan.hold_from, plan.release = args.hold_rank, args.hold_from, args.release
    if args.mode == "server":
        missing = [flag for flag, value in (("--endpoint", args.endpoint), ("--store", args.store),
                                            ("--status-interval", args.status_interval)) if value is None]
        if missing:
            raise UsageError(f"--mode server needs {', '.join(missing)}")
        if args.status_interval <= 0:
            raise UsageError("--status-interval must be above 0 (the server's --watchdog-seconds)")
        plan.interval = args.status_interval
    else:
        plan.interval = args.look_seconds
    return plan


# --- The reader adapter -----------------------------------------------------------------------


def resolve_reader(spec: str) -> Callable[[list[dict]], dict] | None:
    """`module:function`, or None when either cannot be found."""
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    found = getattr(module, attribute, None)
    return found if callable(found) else None


def read_result(reader: Callable[[list[dict]], dict], records: list[dict]) -> dict:
    """The reader's result as contract section 8's object."""
    result = reader(records)
    if isinstance(result, Mapping) and isinstance(result.get("replica"), Mapping) and "steps" not in result:
        result = result["replica"]
    if not isinstance(result, Mapping) or not isinstance(result.get("steps"), list):
        raise TypeError(f"the reader gave {type(result).__name__}, not an object with a `steps` list")
    return dict(result)


def read_rank_lines(path: Path) -> list[dict]:
    """The whole JSON lines of one rank file; a line still being written is left for the next look."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    out = []
    for line in data.split(b"\n")[:-1]:  # the part after the last newline is not whole yet
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def rank_file(work: Path, run_id: str, rank: int) -> Path:
    return work / f"{run_id}.rank{rank}.jsonl"


def run_records(work: Path, run_id: str, world: int) -> list[dict]:
    """Every rank's records of this run, rank by rank, each in its file's order."""
    return [r for rank in range(world) for r in read_rank_lines(rank_file(work, run_id, rank))
            if r.get("run_id") == run_id]


# --- The sink ---------------------------------------------------------------------------------


@dataclass
class Progress:
    """The training step a rank has reached, shared with its sink and its observer."""
    step: int = -1


class ProofSink:
    """Wraps a rank's `JsonlSink`. Writes one record at a time and logs, for each, its step, the
    byte where its line ends and when the file held it. On the held rank, records of steps from
    `hold_from` on are kept in memory instead, until `release_file` exists (looked for at each
    write and every `poll` seconds on a thread of its own, so a release never waits for training)
    or, with `never`, for good: they are dropped at close. A delayed rank not released by its
    close is written then, and `released_at_close` says so: a run too short for the proof.

    With `offsets`, a thread logs each change of the stream sender's offset file: the bytes of
    this rank's file the server has acknowledged, and when."""

    def __init__(self, inner: Any, *, hold_from: int | None = None, never: bool = False,
                 release_file: Path | None = None, progress: Progress | None = None,
                 offsets: Path | None = None, poll: float = ACK_POLL,
                 clock: Callable[[], float] = time.time) -> None:
        self.inner = inner
        self.path = inner.path
        self.hold_from = hold_from
        self.never = never
        self.release_file = release_file
        self.progress = progress or Progress()
        self.clock = clock
        self.held: list[dict] = []
        self.written: list[dict] = []  # {"step", "end", "t", "replica"}
        self.offsets: list[tuple[float, int]] = []
        self.launch: str | None = None
        self.released_at: float | None = None
        self.released_at_step: int | None = None
        self.released_at_close = False
        self.dropped = 0
        self.held_steps: set[int] = set()
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._threads: list[threading.Thread] = []
        if hold_from is not None and not never and release_file is not None:
            self._threads.append(threading.Thread(target=self._watch_release, args=(poll * 4,), daemon=True))
        if offsets is not None:
            self._offset_file = offsets
            self._threads.append(threading.Thread(target=self._watch_offsets, args=(poll,), daemon=True))
        for thread in self._threads:
            thread.start()

    @property
    def holding(self) -> bool:
        return self.hold_from is not None and self.released_at is None

    def _write_one(self, record: dict) -> None:
        self.inner.write([record])
        replica = record.get("replica")
        if isinstance(replica, Mapping) and self.launch is None:
            self.launch = replica.get("launch")
        self.written.append({"step": record.get("step"), "end": os.path.getsize(self.path), "t": self.clock(),
                             "replica": isinstance(replica, Mapping)})

    def _release(self, at_close: bool = False) -> None:
        # under self._lock
        if self.released_at is not None or self.hold_from is None:
            return
        self.released_at, self.released_at_step = self.clock(), self.progress.step
        self.released_at_close = at_close
        held, self.held = self.held, []
        for record in held:
            self._write_one(record)

    def _maybe_release(self) -> None:
        if self.holding and not self.never and self.release_file is not None and self.release_file.exists():
            self._release()

    def write(self, records: list[dict]) -> None:
        with self._lock:
            self._maybe_release()
            for record in records:
                step = record.get("step")
                if self.holding and isinstance(step, int) and step >= self.hold_from:
                    self.held.append(record)
                    self.held_steps.add(step)
                else:
                    self._write_one(record)

    def _watch_release(self, poll: float) -> None:
        while not self._closing.wait(poll):
            with self._lock:
                self._maybe_release()
            if not self.holding:
                return

    def _read_offset(self) -> int | None:
        try:
            text = self._offset_file.read_text(encoding="ascii").strip()
        except OSError:
            return None
        try:
            document = json.loads(text)
        except ValueError:
            return None
        offset = document.get("offset") if isinstance(document, dict) else document
        return offset if isinstance(offset, int) and not isinstance(offset, bool) else None

    def _note_offset(self) -> None:
        offset = self._read_offset()
        if offset is not None and (not self.offsets or self.offsets[-1][1] != offset):
            self.offsets.append((self.clock(), offset))

    def _watch_offsets(self, poll: float) -> None:
        while not self._closing.wait(poll):
            self._note_offset()

    def close(self) -> None:
        with self._lock:
            if self.holding:
                if self.never:
                    self.dropped = len(self.held)
                    self.held = []
                else:
                    self._release(at_close=True)
        self.inner.close()

    def stop(self) -> None:
        """After the session closed (its sender has sent what it will): stop the threads, with one
        last look at the offset file."""
        self._closing.set()
        for thread in self._threads:
            thread.join(timeout=5)
        if hasattr(self, "_offset_file"):
            self._note_offset()

    def report(self) -> dict:
        return {
            "written": self.written, "offsets": self.offsets, "launch": self.launch,
            "held_steps": sorted(self.held_steps), "released_at": self.released_at,
            "released_at_step": self.released_at_step, "released_at_close": self.released_at_close,
            "dropped": self.dropped,
        }


# --- What the driver saw ----------------------------------------------------------------------


def _key(entry: Mapping[str, Any]) -> tuple:
    group = entry.get("group")
    return (entry.get("launch"), tuple(group) if isinstance(group, list) else None, entry.get("step"),
            entry.get("why") if entry.get("coverage") == "not_established" else None)


def _incident_key(incident: Mapping[str, Any]) -> str:
    return json.dumps(incident, sort_keys=True)


@dataclass
class VerdictLog:
    """Every result object seen, merged: the newest entry per launch, group and step, every
    incident ever listed, and when each step was first seen in each state."""
    entries: dict[tuple, dict] = field(default_factory=dict)
    incidents: dict[str, dict] = field(default_factory=dict)
    first_seen: dict[tuple, float] = field(default_factory=dict)  # (key, state) -> t
    looks: int = 0
    last_t: float | None = None
    errors: list[str] = field(default_factory=list)

    @staticmethod
    def state(entry: Mapping[str, Any]) -> tuple:
        coverage = entry.get("coverage")
        if coverage == "insufficient":
            return (coverage, tuple(entry.get("missing") or ()))
        if coverage == "complete":
            return (coverage, entry.get("fingerprints"))
        return (coverage, entry.get("why"))

    def add(self, t: float, result: Mapping[str, Any] | None) -> None:
        self.looks += 1
        self.last_t = t
        if not isinstance(result, Mapping):
            return
        for entry in result.get("steps") or []:
            if not isinstance(entry, Mapping):
                continue
            key = _key(entry)
            self.entries[key] = dict(entry)
            self.first_seen.setdefault((key, self.state(entry)), t)
        for incident in result.get("replica_drift") or []:
            if isinstance(incident, Mapping):
                self.incidents.setdefault(_incident_key(incident), dict(incident))

    def as_result(self) -> dict:
        return {"steps": list(self.entries.values()), "replica_drift": list(self.incidents.values())}

    def first(self, launch: str, group: list[int], step: int, state: tuple) -> float | None:
        return self.first_seen.get(((launch, tuple(group), step, None), state))

    def first_judged(self, launch: str, group: list[int], step: int) -> float | None:
        times = [t for ((key, state), t) in self.first_seen.items()
                 if key == (launch, tuple(group), step, None) and state[0] == "complete"]
        return min(times) if times else None


# --- The assertions ---------------------------------------------------------------------------

_LAUNCH = re.compile(r"^[0-9a-f]{32}$")


def check_result(result: Mapping[str, Any], plan: Plan, launch: str | None) -> list[str]:
    """Contract sections 4 and 5, per scheduled step, from the result object (never from one rank's
    record): the coverage; for a judged step, drift on exactly the planted parameters or agreement;
    `replica_drift` raised on exactly the planted, judged steps, each naming the parameters, the
    step, the launch and the group; and nothing for a step that stays insufficient."""
    failures: list[str] = []
    if not isinstance(launch, str) or not _LAUNCH.match(launch):
        return [f"rank 0's own record names no launch (got {launch!r}): nothing can be matched"]
    group = plan.group
    steps = [e for e in result.get("steps") or [] if isinstance(e, Mapping)]
    for entry in steps:
        if entry.get("launch") != launch:
            failures.append(f"an entry of another start or none ({entry.get('launch')!r}) at step"
                            f" {entry.get('step')}: {dict(entry)}")
        elif entry.get("coverage") == "not_established":
            failures.append(f"step {entry.get('step')} is not_established ({entry.get('why')}): {dict(entry)}")
        elif entry.get("step") not in plan.scheduled:
            failures.append(f"an entry for step {entry.get('step')}, which is not scheduled")
    mine = [e for e in steps if e.get("launch") == launch and e.get("coverage") != "not_established"]
    for step in plan.scheduled:
        found = [e for e in mine if e.get("step") == step]
        if not found:
            failures.append(f"step {step}: no entry for launch {launch} (a step must be listed to be checked)")
            continue
        if len(found) > 1:
            failures.append(f"step {step}: {len(found)} entries where one is due: {found}")
            continue
        entry = found[0]
        if entry.get("group") != group:
            failures.append(f"step {step}: group {entry.get('group')}, not {group}")
        if plan.never_judged(step):
            if entry.get("coverage") != "insufficient" or entry.get("missing") != [plan.hold_rank]:
                failures.append(f"step {step}: rank {plan.hold_rank} never reported, so insufficient with"
                                f" missing [{plan.hold_rank}] is due; got {dict(entry)}")
            continue
        if entry.get("coverage") != "complete":
            failures.append(f"step {step}: coverage {entry.get('coverage')} (missing {entry.get('missing')});"
                            " complete is due")
            continue
        planted = sorted({p.param for p in plan.plants_at(step)})
        if planted:
            if entry.get("fingerprints") != "drift" or sorted(entry.get("drift_parameters") or []) != planted:
                failures.append(f"step {step}: drift on {planted} is due; fingerprints {entry.get('fingerprints')}"
                                f" on {entry.get('drift_parameters')}")
            if entry.get("signatures") == "agree":
                failures.append(f"step {step}: signatures agree where {planted} was changed")
        else:
            if entry.get("fingerprints") != "agree":
                failures.append(f"step {step}: a clean step must read agree; fingerprints {entry.get('fingerprints')}"
                                f" ({entry.get('drift_parameters') or entry.get('fingerprints_why')})")
            if entry.get("signatures") == "differ":
                failures.append(f"step {step}: signatures differ on a clean step ({entry.get('signature_parameters')})")
    incidents = [i for i in result.get("replica_drift") or [] if isinstance(i, Mapping)]
    due = {s: sorted({p.param for p in plan.plants_at(s)}) for s in plan.scheduled
           if plan.plants_at(s) and not plan.never_judged(s)}
    raised: dict[int, list[Mapping[str, Any]]] = {}
    for incident in incidents:
        if incident.get("source") != "replica":
            failures.append(f"an incident from {incident.get('source')!r}: the driver gathers nothing, so only"
                            f" observations may raise: {dict(incident)}")
            continue
        raised.setdefault(incident.get("step"), []).append(incident)
    for step, found in sorted(raised.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        if step not in due:
            failures.append(f"replica_drift raised at step {step}, where none is due: {found}")
            continue
        if len(found) != 1:
            failures.append(f"step {step}: {len(found)} replica_drift incidents where one is due")
        incident = found[0]
        if sorted(incident.get("parameters") or []) != due[step]:
            failures.append(f"step {step}: replica_drift names {incident.get('parameters')}, not {due[step]}")
        if incident.get("launch") != launch or incident.get("group") != group:
            failures.append(f"step {step}: replica_drift names launch {incident.get('launch')} and group"
                            f" {incident.get('group')}, not {launch} and {group}")
    for step in sorted(set(due) - set(raised)):
        failures.append(f"step {step}: replica_drift on {due[step]} is due and was not raised")
    return failures


def check_history(log: VerdictLog, plan: Plan, launch: str | None, held: Mapping[str, Any] | None,
                  seen_while: Mapping[int, int]) -> list[str]:
    """Section 7's two proofs, from when each state was first seen. `held` is the held rank's sink
    report; `seen_while[step]` the furthest training step rank 0 had reached at a look that still
    read that step insufficient."""
    if plan.hold_rank is None:
        return []
    failures: list[str] = []
    if held is None:
        return [f"rank {plan.hold_rank} left no report"]
    first_held = plan.hold_from
    state = ("insufficient", (plan.hold_rank,))
    t_insufficient = log.first(launch, plan.group, first_held, state) if launch else None
    if t_insufficient is None:
        failures.append(f"step {first_held} was never seen insufficient naming rank {plan.hold_rank}")
    else:
        reached = seen_while.get(first_held)
        if reached is None or reached <= first_held:
            failures.append(f"step {first_held} read insufficient only while training was at step {reached}:"
                            " it shows no training going on past it")
    if plan.release == "never":
        if held.get("released_at") is not None:
            failures.append(f"rank {plan.hold_rank}'s telemetry was released in a run where it never reports")
        if not held.get("dropped"):
            failures.append(f"rank {plan.hold_rank} held nothing back: the proof did not suppress anything")
        return failures
    released = held.get("released_at")
    if released is None:
        failures.append(f"rank {plan.hold_rank}'s telemetry was never released")
        return failures
    if held.get("released_at_close"):
        failures.append(f"rank {plan.hold_rank}'s telemetry was released only at its close: the run was too short"
                        " for the step to be seen insufficient while training went on (raise --steps or"
                        " --step-seconds)")
    if t_insufficient is not None and t_insufficient > released:
        failures.append("the step was first seen insufficient after the release")
    for step in plan.scheduled:
        if not plan.held(step):
            continue
        judged = log.first_judged(launch, plan.group, step) if launch else None
        if judged is None:
            failures.append(f"step {step} was never judged after the release")
        elif judged < released:
            failures.append(f"step {step} was judged at {judged:.3f}, before the release at {released:.3f}")
    return failures


def check_no_wait(reports: Mapping[int, Mapping[str, Any]], plan: Plan) -> list[str]:
    """Training ran to its end on every rank, and while a rank was held no step on any rank took
    more than `--wait-slack` above its own sleep."""
    failures: list[str] = []
    for rank in plan.group:
        report = reports.get(rank)
        if report is None:
            failures.append(f"rank {rank} left no report: it did not finish")
            continue
        if report.get("steps_done") != plan.steps:
            failures.append(f"rank {rank} trained {report.get('steps_done')} of {plan.steps} steps")
        if plan.hold_rank is None:
            continue
        durations = report.get("durations") or []
        for step, took in enumerate(durations):
            if step >= plan.hold_from and took - plan.step_seconds > plan.wait_slack:
                failures.append(f"rank {rank} step {step} took {took:.3f} s, {took - plan.step_seconds:.3f} s above"
                                f" its sleep (allowed {plan.wait_slack} s) while rank {plan.hold_rank} was held")
    return failures


def arrival(report: Mapping[str, Any], step: int, mode: str) -> float | None:
    """When this rank's observation of `step` was accepted: in files mode when the file held its
    line; in server mode when the sender's offset first covered the line's last byte."""
    lines = [w for w in report.get("written") or [] if w.get("step") == step and w.get("replica")]
    if not lines:
        return None
    line = lines[0]
    if mode == "files":
        return line["t"]
    for t, offset in report.get("offsets") or []:
        if offset >= line["end"]:
            return t
    return None


def check_latency(log: VerdictLog, plan: Plan, launch: str | None,
                  reports: Mapping[int, Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """Section 7's clock, for every judged step: from the last required observation's arrival to
    the first look or status that holds the step's verdict. Fails above one interval plus the
    margin. Returns (failures, the lines to print)."""
    failures: list[str] = []
    lines: list[str] = []
    bound = plan.interval + plan.margin
    what = "look of the reader" if plan.mode == "files" else "status interval"
    if plan.mode == "server":
        lines.append(f"each rank looked at its offset file every {ACK_POLL * 1000:g} ms: an acknowledgement seen"
                     " late makes a step's time read short by up to that much")
    for step in plan.scheduled:
        if plan.never_judged(step):
            lines.append(f"step {step}: never judged (rank {plan.hold_rank} never reports): no clock")
            continue
        arrived = [arrival(reports.get(rank) or {}, step, plan.mode) for rank in plan.group]
        if any(t is None for t in arrived):
            late = [rank for rank, t in zip(plan.group, arrived) if t is None]
            accepted = "written to its file" if plan.mode == "files" else "acknowledged by the server"
            failures.append(f"step {step}: the observation of rank(s) {late} was never seen {accepted}")
            continue
        start = max(arrived)
        stop = log.first_judged(launch, plan.group, step) if launch else None
        if stop is None:
            failures.append(f"step {step}: no verdict was seen")
            continue
        took = stop - start
        lines.append(f"step {step}: last observation accepted at {_clock(start)}, verdict seen at {_clock(stop)},"
                     f" {took:+.3f} s (bound {bound:.3f} s: one {what} {plan.interval} s + margin {plan.margin} s)")
        if took > bound:
            failures.append(f"step {step}: the verdict came {took:.3f} s after the last observation, above {bound:.3f} s")
    return failures, lines


def _clock(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


def check_kinds(records: list[dict], plan: Plan, launch: str | None) -> tuple[list[str], list[str]]:
    """From every rank's record of a planted step (a cross-rank reading, as the result object is):
    `add` moves the norm; `neg` keeps the norm and moves the sum; `swap` keeps both and moves only
    the projection. Also prints each rank's fingerprints of the planted parameter, the evidence D1
    will count, without naming any rank. Returns (failures, lines)."""
    failures: list[str] = []
    lines: list[str] = []
    for step in sorted({p.step for p in plan.plants}):
        if plan.never_judged(step):
            continue
        by_rank = {r["rank"]: r["replica"] for r in records
                   if r.get("step") == step and isinstance(r.get("replica"), Mapping)
                   and r["replica"].get("launch") == launch}
        if set(by_rank) != set(plan.group):
            failures.append(f"step {step}: records with an observation from ranks {sorted(by_rank)}, not {plan.group}")
            continue
        for param in sorted({p.param for p in plan.plants_at(step)}):
            kinds = {p.kind for p in plan.plants_at(step) if p.param == param}
            try:
                fps = {rank: [float(v) for v in obs["params"][param]["fp"]] for rank, obs in by_rank.items()}
            except (KeyError, TypeError, ValueError):
                failures.append(f"step {step}: {param} is not in every rank's params")
                continue
            values = sorted({tuple(v) for v in fps.values()})
            lines.append(f"step {step} {param}: {len(values)} distinct fingerprint(s) among {len(fps)} ranks"
                         f" (D1 held: no rank is named) " + "; ".join(f"rank {r} {fps[r]}" for r in sorted(fps)))
            scale = max(1e-12, sum(v[0] for v in fps.values()) / len(fps))

            def spread(column: int) -> float:
                column_values = [v[column] for v in fps.values() if len(v) > column]
                if len(column_values) < len(fps):
                    return math.nan
                return (max(column_values) - min(column_values)) / scale

            norm, total, projection = spread(0), spread(1), spread(2)
            if len(kinds) > 1:
                continue
            [kind] = kinds
            if kind == "add" and not norm > TOLERANCE:
                failures.append(f"step {step} {param}: an added 0.25 must move the norm (spread {norm})")
            if kind == "neg" and not (norm <= TOLERANCE and total > TOLERANCE):
                failures.append(f"step {step} {param}: a sign flip keeps the norm and moves the sum"
                                f" (spreads {norm}, {total})")
            if kind == "swap" and not (norm <= TOLERANCE and total <= TOLERANCE and projection > TOLERANCE):
                failures.append(f"step {step} {param}: a swap moves only the projection (spreads {norm}, {total},"
                                f" {projection})")
    return failures, lines


def check_records(records: list[dict], plan: Plan) -> list[str]:
    """Each rank's own file holds every step it was due to write, each with the loss: every step on
    a rank that reported, the steps before the hold on one that never did."""
    failures: list[str] = []
    for rank in plan.group:
        mine = [r for r in records if r.get("rank") == rank]
        due = [s for s in range(plan.steps) if not (rank == plan.hold_rank and plan.never_judged(s))]
        steps = [r.get("step") for r in mine]
        if steps != due:
            failures.append(f"rank {rank}'s file holds steps {steps}, not {due}")
        if not all("train/loss" in (r.get("scalars") or {}) for r in mine):
            failures.append(f"rank {rank}: the loss was not recorded on every step")
    return failures


@dataclass
class Verdict:
    code: int
    lines: list[str]
    failures: list[str]


def evaluate(plan: Plan, log: VerdictLog, seen_while: Mapping[int, int], reports: Mapping[int, Mapping[str, Any]],
             launch: str | None, records: list[dict], reader_found: bool) -> Verdict:
    """Every check, composed once, for the driver and its tests alike: PASS only when each holds;
    NOT PROVEN when the file reader was not found (whatever else held); FAIL otherwise."""
    lines = [f"{plan.world} ranks, mode {plan.mode}, launch {launch}, group {plan.group},"
             f" plants: {'; '.join(map(str, plan.plants)) or 'none'}"]
    if plan.hold_rank is not None:
        held = reports.get(plan.hold_rank, {})
        lines.append(f"rank {plan.hold_rank} held from step {plan.hold_from} ({plan.release}): held steps"
                     f" {held.get('held_steps')}, released at training step {held.get('released_at_step')}"
                     f"{' (at its close)' if held.get('released_at_close') else ''}, dropped {held.get('dropped')}"
                     f" records; step {plan.hold_from} still read insufficient while rank 0 trained step"
                     f" {seen_while.get(plan.hold_from)}")
    failures = check_no_wait(reports, plan) + check_records(records, plan)
    if plan.mode == "files" and not reader_found:
        lines += [f"FAIL {f}" for f in failures]
        lines.append(f"NOT PROVEN: the reader {plan.reader} was not found (lane Z2's function; --reader or"
                     f" {READER_ENV} names another). Trained {plan.steps} steps on {len(reports)} of {plan.world}"
                     " ranks; no verdict was read.")
        return Verdict(EXIT_FAIL if failures else EXIT_NOT_PROVEN, lines, failures)
    result = log.as_result()
    for entry in sorted(result["steps"], key=lambda e: (e.get("step") is None, e.get("step") or 0)):
        lines.append(f"  step {entry.get('step')}: {entry.get('coverage')}"
                     + (f" missing {entry.get('missing')}" if entry.get("missing") else "")
                     + (f" fingerprints {entry.get('fingerprints')}" if entry.get("fingerprints") else "")
                     + (f" on {entry.get('drift_parameters')}" if entry.get("drift_parameters") else "")
                     + (f" signatures {entry.get('signatures')}" if entry.get("signatures") else ""))
    lines += [f"  replica_drift: {incident}" for incident in result["replica_drift"]]
    failures += check_result(result, plan, launch)
    failures += check_history(log, plan, launch, reports.get(plan.hold_rank) if plan.hold_rank is not None
                              else None, seen_while)
    late, clock_lines = check_latency(log, plan, launch, reports)
    kinds, d1_lines = check_kinds(records, plan, launch)
    failures += late + kinds
    lines += [f"  {line}" for line in clock_lines + d1_lines]
    lines += [f"  reader error at a look: {error}" for error in log.errors[:5]]
    lines += [f"FAIL {f}" for f in failures]
    if failures:
        return Verdict(EXIT_FAIL, lines, failures)
    lines.append(f"PASS {plan.mode} on {plan.world} ranks ({log.looks} looks)")
    return Verdict(EXIT_PASS, lines, failures)


# --- The observer -----------------------------------------------------------------------------


class Observer:
    """On rank 0, on a thread of its own: looks at the verdict while the job trains, records what
    it saw and when, and releases a delayed rank once its first held step reads insufficient and
    training has gone `hold_steps` further. It never touches the training loop."""

    def __init__(self, plan: Plan, work: Path, run_id: str, progress: Progress,
                 launch: Callable[[], str | None], reader: Callable[[list[dict]], dict] | None,
                 clock: Callable[[], float] = time.time) -> None:
        self.plan, self.work, self.run_id = plan, work, run_id
        self.progress, self.launch, self.reader, self.clock = progress, launch, reader, clock
        self.log = VerdictLog()
        self.seen_while: dict[int, int] = {}
        self.release_file = work / "release"
        self.released_at: float | None = None
        self._status_mark: tuple | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def status_file(self) -> Path:
        return Path(self.plan.store) / "runs" / self.run_id / "watchdog.json"

    def look(self) -> bool:
        """One look. True when it read something new."""
        if self.plan.mode == "files":
            if self.reader is None:
                return False
            records = run_records(self.work, self.run_id, self.plan.world)
            try:
                result = read_result(self.reader, records)
            except Exception as error:  # noqa: BLE001  said at the end, never raised into training
                self.log.errors.append(f"{type(error).__name__}: {error}")
                return False
            self.log.add(self.clock(), result)
        else:
            try:
                stat = self.status_file.stat()
            except OSError:
                return False
            mark = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            if mark == self._status_mark:
                return False
            try:
                document = json.loads(self.status_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False  # met mid-replace; read again at the next poll
            self._status_mark = mark
            self.log.add(stat.st_mtime_ns / 1e9, document.get("replica") if isinstance(document, dict) else None)
        self._after_look()
        return True

    def _after_look(self) -> None:
        launch = self.launch()
        if launch is None:
            return
        reached = self.progress.step
        for step in self.plan.scheduled:
            entry = self.log.entries.get((launch, tuple(self.plan.group), step, None))
            if entry and entry.get("coverage") == "insufficient":
                # the furthest training step reached at a look that still read this step insufficient
                self.seen_while[step] = max(self.seen_while.get(step, -1), reached)
        plan = self.plan
        if plan.hold_rank is None or plan.release != "after-verdict" or self.released_at is not None:
            return
        entry = self.log.entries.get((launch, tuple(plan.group), plan.hold_from, None)) or {}
        if (VerdictLog.state(entry) == ("insufficient", (plan.hold_rank,))
                and reached >= plan.hold_from + plan.hold_steps):
            self.release_file.touch()
            self.released_at = self.clock()

    def _run(self) -> None:
        poll = self.plan.interval if self.plan.mode == "files" else 0.01
        while not self._stop.wait(poll):
            self.look()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
