"""Measure TensorQuill's added step time well enough to tell 1% (and 0.5%), or say it cannot.

    python examples/overhead_harness.py --out runs/overhead-routers --model granite --rounds 30
    python examples/overhead_harness.py --out runs/overhead-grads --model granite --arm-set grads --rounds 30
    python examples/overhead_harness.py --out runs/overhead-k3 --model granite --limit 0.005 \\
        --arm 'k3={"init": {"capture": "routers", "some_switch": true}, "baseline": "routers"}'
    python examples/overhead_harness.py --out runs/overhead-toy --model toy --device cpu --rounds 2   # dry run
    python examples/overhead_harness.py plan --sd 0.011 --limit 0.01 --true 0.005                  # rounds needed

**Arms.** An arm is a name, `init` (keyword arguments for `tensorquill.init`, or null for no
TensorQuill at all), `env` (environment variables for its process), `limit` (a fraction such as
0.01, or null to report it without judging it; absent: `--limit`), `baseline` (the arm it is paired
with, `off` unless given) and `log_loss` (whether it logs the loss each step, as the gate does;
default true). A detector or switch is judged over the same capture without it, not over capture
off (the author, Nov 26, 2026, for `K3`'s 0.5%): give it `"baseline": "routers"`, or whichever arm
is that capture. The harness passes `sink` (a JSON Lines file under `arms/capture/`) and `run_id`
itself, so an arm may not set them; it passes the model's router spec as `routers` unless the arm
sets `"routers"` itself (`null`: none). Every run has two capture-off arms, `off` (the baseline)
and `off-again` (the same arm again), so the noise between identical arms is measured in the same
run. Ready-made sets (`--arm-set`): `routers` (the default), `capture` (router-only and full
capture), `grads` (router-only, and `grads=1`, `10`, `50` with it), `none`. More arms come from
`--arm NAME=JSON` (repeatable) or `--arms-file` (a JSON list of objects with a `name`), so a switch
not yet built (a pinned flush, a new detector) needs no change here.

**A round** runs every arm once, each in a new Python process (arms differ in environment, and one
arm's writer thread must never run while another is timed), never two at once. Rounds come in
blocks of k (k arms): each block is a cyclic Latin square of a fresh permutation drawn from
`--seed`, its rows shuffled, so each arm takes each position once per block and which arm follows
which changes between blocks. In each process: the model and batches from fixed seeds (the same in
every arm), `tensorquill.init` if the arm has one, `torch.compile` (`--compile`), `--warmup` steps,
`gc.collect()`, then `--steps` timed steps in windows of `--window`. The device is waited for
(`torch.cuda.synchronize()`, every stream, so a flush on a side stream is counted) before each
window's first timestamp and after its last. An arm's figure for the round is the timed wall time
over the timed steps, flush steps included.

**The statistic.** Per round r: d_r = arm time / its baseline's time in the same round, minus one.
The estimate is the median of d_r; its range is the distribution-free interval for the median from
order statistics (`--confidence`, 0.95 two-sided), which exists from 6 rounds on. Against its limit
L an arm is **met** when the interval's top is at most L, **not met** when its bottom is above L,
and **cannot tell** otherwise. Every verdict is cannot tell when `off-again`'s interval leaves out 0
(identical arms called different: about 5% of honest runs, by the interval's coverage), when the GPU
was shared or could not be checked, or off a CUDA GPU (a dry run). The smallest difference the run
could tell is the half-width of `off-again`'s interval; from its paired SD the result also gives the
rounds needed to tell each limit for an arm at half of it, with 90% power.

**The GPU is not shared**, checked through NVML (`nvidia-ml-py`) or `nvidia-smi`: before the run,
after each arm process (a second after it ends), and after the run, no process may hold the GPU
and its utilization must be under 5%; just before and just after its timed steps each arm sees at
most one process, itself. The driver never opens a CUDA context. A process list that never shows
the run's own arm (a container's view) leaves utilization as the only check, and the result says so;
no reader at all marks the run not checked. Another user seen, no check, or a GPU not shown to be
unshared (utilization only) makes every verdict cannot tell: such a run still measures and reports
every figure, but scores no limit (the author, Nov 26, 2026). Start from a fresh runtime: a notebook kernel holding the GPU counts as another
process.

**Files** in `--out`: `plan.json` (settings, for `--resume`), `arms/<arm>-r<round>.json` and `.log`,
`gpu-samples.jsonl`, `result.json` (every round's raw times, the device, driver, PyTorch, the arms'
settings, the seed and orders, the GPU checks), `verdict.json` and `table.txt`. `--resume` finishes
the planned rounds of a run that stopped; more rounds than planned need `--extend`, and the result
records that rounds were added after a look. If `off` fails in the first round the run stops there:
the model cannot be measured in that mode on this machine (the four-arm gate's CUDA-graph failure).

Exit status, as the run sheet's: 0 every judged arm met (or a dry run, or nothing judged and the
identical arms agree), 1 an arm not met or an arm process failed, 2 bad arguments or a missing
requirement, 3 not proven (cannot tell, or the baseline could not run).
"""

from __future__ import annotations

import argparse
import functools
import gc
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_PROVEN = 0, 1, 2, 3
DEFAULT_LIMIT = 0.01
FLUSH_EVERY = 50  # tensorquill.init's default `every`
IDLE_UTILIZATION_LIMIT = 5.0  # percent: the GPU busier than this while no arm runs is someone else's work
SETTLE_SECONDS = 1.0
BASELINE, TWIN = "off", "off-again"
TOY_SHAPE = (2, 32)  # batch size and sequence length of the toy model's batches
GRANITE = "ibm-granite/granite-3.0-1b-a400m-base"
COMPILE_MODES = ("none", "default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune")
CUDA_GRAPH_MODES = ("reduce-overhead", "max-autotune")
ARM_KEYS = {"name", "init", "env", "limit", "baseline", "log_loss"}
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

BASELINE_ARMS = (
    {"name": BASELINE, "init": None, "env": {}, "limit": None, "baseline": None, "log_loss": True},
    {"name": TWIN, "init": None, "env": {}, "limit": None, "baseline": BASELINE, "log_loss": True},
)
_ROUTERS = {"name": "routers", "init": {"capture": "routers"}}
ARM_SETS: dict[str, list[dict]] = {
    "none": [],
    "routers": [_ROUTERS],
    "capture": [_ROUTERS, {"name": "full", "init": {"capture": "full"}, "limit": None}],
    "grads": [_ROUTERS] + [{"name": f"grads-{n}", "init": {"capture": "routers", "grads": n}, "limit": None}
                           for n in (1, 10, 50)],
}


class ArmError(ValueError):
    """An arm the harness cannot run: refused before any process starts."""


# ---- arms ----------------------------------------------------------------------------------


def normalize_arm(spec: Any, *, default_limit: float | None) -> dict:
    if not isinstance(spec, dict):
        raise ArmError(f"an arm is a JSON object, not {spec!r}")
    unknown = set(spec) - ARM_KEYS
    if unknown:
        raise ArmError(f"arm {spec.get('name')!r}: unknown keys {sorted(unknown)} (an arm has {sorted(ARM_KEYS)})")
    name = spec.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise ArmError(f"arm name {name!r}: letters, digits, '.', '_' and '-' only")
    if name in (BASELINE, TWIN):
        raise ArmError(f"arm name {name!r} is reserved: every run has the two capture-off arms {BASELINE} and {TWIN}")
    init = spec.get("init", {})
    if init is not None and not isinstance(init, dict):
        raise ArmError(f"arm {name!r}: init must be an object of tensorquill.init settings, or null for capture off")
    for owned in ("sink", "run_id"):
        if init and owned in init:
            raise ArmError(f"arm {name!r}: the harness sets {owned} itself (each arm's files go beside its result)")
    env = spec.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ArmError(f"arm {name!r}: env must map names to strings")
    limit = spec["limit"] if "limit" in spec else default_limit
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit < 0):
        raise ArmError(f"arm {name!r}: limit must be a fraction such as 0.01, or null; not {limit!r}")
    baseline = spec.get("baseline", BASELINE)
    if not isinstance(baseline, str):
        raise ArmError(f"arm {name!r}: baseline must name another arm")
    log_loss = spec.get("log_loss", True)
    if not isinstance(log_loss, bool):
        raise ArmError(f"arm {name!r}: log_loss must be true or false")
    return {"name": name, "init": None if init is None else dict(init), "env": dict(env),
            "limit": None if limit is None else float(limit), "baseline": baseline, "log_loss": log_loss}


def parse_arm_option(text: str) -> dict:
    name, sep, body = text.partition("=")
    if not sep:
        raise ArmError(f"--arm takes NAME=JSON, not {text!r}")
    try:
        spec = json.loads(body)
    except ValueError as error:
        raise ArmError(f"--arm {name}: not JSON ({error})") from None
    if not isinstance(spec, dict):
        raise ArmError(f"--arm {name}: the JSON must be an object")
    return {**spec, "name": name}


def resolve_arms(arm_set: str, options: list[dict], arms_file: Path | None, *, default_limit: float | None) -> list[dict]:
    if arm_set not in ARM_SETS:
        raise ArmError(f"--arm-set {arm_set!r}: one of {', '.join(ARM_SETS)}")
    specs = [dict(spec) for spec in ARM_SETS[arm_set]] + list(options)
    if arms_file is not None:
        loaded = json.loads(Path(arms_file).read_text())
        if not isinstance(loaded, list):
            raise ArmError(f"{arms_file}: a JSON list of arms")
        specs += loaded
    arms = [dict(spec) for spec in BASELINE_ARMS] + [normalize_arm(s, default_limit=default_limit) for s in specs]
    names = [a["name"] for a in arms]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ArmError(f"arm names given twice: {duplicates}")
    for arm in arms[2:]:
        if arm["baseline"] not in names or arm["baseline"] == arm["name"]:
            raise ArmError(f"arm {arm['name']!r}: baseline {arm['baseline']!r} is not another arm of this run")
    return arms


def check_alignment(arms: list[dict], *, warmup: int, steps: int, window: int) -> None:
    """Every window must hold the same number of flushes in every arm."""
    if window < 1 or warmup < 1 or steps < 1 or warmup % window or steps % window:
        raise ArmError(f"--warmup ({warmup}) and --steps ({steps}) must be positive multiples of --window ({window})")
    for arm in arms:
        if arm["init"] is None:
            continue
        every = arm["init"].get("every", FLUSH_EVERY)
        if isinstance(every, bool) or not isinstance(every, int) or every < 1 or window % every:
            raise ArmError(f"arm {arm['name']!r}: every={every!r} must divide --window ({window}), so each "
                           "window holds the same number of flushes")


def check_routers(arms: list[dict], *, model_has_routers: bool) -> None:
    if model_has_routers:
        return
    for arm in arms:
        init = arm["init"]
        if init is not None and init.get("capture") == "routers" and not init.get("routers"):
            raise ArmError(f"arm {arm['name']!r}: capture='routers' but this model has no router spec "
                           "(give the arm its own \"routers\", or use capture='full' with \"routers\": null)")


def init_kwargs(init: dict, routers: dict, *, sink: Any, run_id: str) -> dict:
    kwargs = dict(init)
    if "routers" in kwargs:
        if kwargs["routers"] is None:
            del kwargs["routers"]
    elif routers:
        kwargs["routers"] = routers
    kwargs["sink"] = sink
    kwargs["run_id"] = run_id
    return kwargs


# ---- the statistics --------------------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def _order_index(n: int, confidence: float) -> tuple[int, float] | None:
    """The largest k whose interval [x_(k), x_(n-k+1)] covers 1 - 2 P(B <= k - 1) >= confidence,
    B ~ Binomial(n, 1/2), with that coverage; None when even k = 1 covers less."""
    best = None
    below, total = 0, 2 ** n  # below: the count of outcomes B <= k - 1, kept exact in integers
    for k in range(1, n // 2 + 1):
        below += math.comb(n, k - 1)
        coverage = 1 - 2 * below / total
        if coverage < confidence:
            break
        best = (k, coverage)
    return best


def median_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float, float] | None:
    """The distribution-free interval for the median: [x_(k), x_(n-k+1)] for the largest k whose coverage,
    1 - 2 P(B <= k - 1), is at least `confidence`. None when even [min, max] covers less."""
    xs = sorted(values)
    found = _order_index(len(xs), confidence)
    if found is None:
        return None
    k, coverage = found
    return xs[k - 1], xs[len(xs) - k], coverage


def min_rounds(confidence: float = 0.95) -> int:
    n = 1
    while median_interval([0.0] * n, confidence) is None:
        n += 1
    return n


def judge(interval: tuple[float, float] | None, limit: float | None) -> str:
    if limit is None:
        return "reported"
    if interval is None:
        return "cannot tell"
    lo, hi = interval[0], interval[1]
    if hi <= limit:
        return "met"
    if lo > limit:
        return "not met"
    return "cannot tell"


def rounds_needed(paired_sd: float, margin: float, confidence: float = 0.95, power: float = 0.9) -> int:
    """Rounds for the interval's edge to clear a limit `margin` away from the true difference, with `power`:
    the median's standard error is sqrt(pi/2) sd / sqrt(n) under normal noise."""
    normal = statistics.NormalDist()
    z = normal.inv_cdf(1 - (1 - confidence) / 2) + normal.inv_cdf(power)
    n = math.ceil((z * math.sqrt(math.pi / 2) * paired_sd / margin) ** 2) if margin > 0 else 10 ** 9
    return max(n, min_rounds(confidence))


def simulate_power(rounds: int, paired_sd: float, true: float, limit: float, *, confidence: float = 0.95,
                   trials: int = 2000, seed: int = 0, twin: bool = False) -> float:
    """How often the harness's own interval calls an arm with overhead `true` met against `limit`, with
    normal paired noise of SD `paired_sd`. With `twin`, a run whose identical pair is called different
    does not count as met (that rule voids about 5% of honest runs)."""
    rng = random.Random(seed)
    met = 0
    for _ in range(trials):
        interval = median_interval([rng.gauss(true, paired_sd) for _ in range(rounds)], confidence)
        if twin:
            pair = median_interval([rng.gauss(0.0, paired_sd) for _ in range(rounds)], confidence)
            if pair is not None and not pair[0] <= 0 <= pair[1]:
                continue
        met += judge(interval, limit) == "met"
    return met / trials


def rounds_by_simulation(paired_sd: float, true: float, limit: float, *, confidence: float = 0.95,
                         power: float = 0.9, trials: int = 2000, seed: int = 0, cap: int = 5000) -> int | None:
    """The fewest rounds whose simulated rate of "met" (the identical-arm rule counted) reaches `power`,
    searched upwards from the formula's count; None past `cap` rounds."""
    n = rounds_needed(paired_sd, limit - true, confidence, power)
    while n <= cap:
        if simulate_power(n, paired_sd, true, limit, confidence=confidence, trials=trials, seed=seed, twin=True) >= power:
            return n
        n = max(n + 1, math.ceil(n * 1.05))
    return None


def analyze(times: dict[int, dict[str, float]], arms: list[dict], *, confidence: float = 0.95,
            problems: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Paired differences, medians, intervals and verdicts from each round's block times (ms per step)."""
    out: dict[str, dict] = {}
    for arm in arms:
        name = arm["name"]
        own = [times[r][name] for r in sorted(times) if name in times[r]]
        entry: dict[str, Any] = {"baseline": arm["baseline"], "limit": arm["limit"],
                                 "median_ms": statistics.median(own) if own else None,
                                 "ms_by_round": {r: times[r][name] for r in sorted(times) if name in times[r]}}
        if name == BASELINE:
            out[name] = {**entry, "rounds": len(own), "verdict": "baseline", "why": "the arm every other is paired with"}
            continue
        base = arm["baseline"]
        rounds = [r for r in sorted(times) if name in times[r] and base in times[r]]
        d = [times[r][name] / times[r][base] - 1 for r in rounds]
        found = median_interval(d, confidence)
        entry.update(rounds=len(d), differences=d, median=statistics.median(d) if d else None,
                     interval=None if found is None else [found[0], found[1]],
                     coverage=None if found is None else found[2])
        out[name] = entry
    twin = out[TWIN]
    agree = None if twin["interval"] is None else twin["interval"][0] <= 0 <= twin["interval"][1]
    sd = statistics.stdev(twin["differences"]) if len(twin["differences"]) >= 2 else None
    limits = sorted({0.01, 0.005} | {a["limit"] for a in arms if a["limit"] is not None}, reverse=True)
    noise = {
        "identical_arms_agree": agree,
        "smallest_difference_told": None if twin["interval"] is None else (twin["interval"][1] - twin["interval"][0]) / 2,
        "paired_sd": sd,
        "rounds_needed": {f"{lim:g}": rounds_needed(sd, lim / 2, confidence) for lim in limits} if sd else None,
        "rounds_needed_means": "rounds for an arm at half the limit to be called met with 90% power, given this "
                               "run's paired SD between the identical arms",
    }
    blockers = list(problems)
    if agree is False:
        blockers.append("the two identical capture-off arms were called different: the noise is not under control")
    few = f"at least {min_rounds(confidence)} rounds are needed for a {confidence:.0%} interval"
    for arm in arms:
        name, entry = arm["name"], out[arm["name"]]
        if name == BASELINE:
            continue
        if name == TWIN:
            if problems or entry["interval"] is None:
                entry["verdict"] = "cannot tell"
                entry["why"] = "; ".join(problems) if problems else f"{entry['rounds']} rounds: {few}"
            else:
                entry["verdict"] = "agrees" if agree else "differs"
                entry["why"] = "the identical arm's interval " + ("holds" if agree else "leaves out") + " zero"
            continue
        verdict = judge(entry["interval"], arm["limit"])
        if verdict == "reported":
            entry["why"] = "reported, not judged (no limit)"
        elif blockers:
            verdict, entry["why"] = "cannot tell", "; ".join(blockers)
        elif entry["interval"] is None:
            entry["why"] = f"{entry['rounds']} rounds: {few}"
        elif verdict == "cannot tell":
            entry["why"] = "the interval holds the limit"
        else:
            entry["why"] = "the interval is " + ("under" if verdict == "met" else "over") + " the limit"
        entry["verdict"] = verdict
    return {"arms": out, "noise": noise, "confidence": confidence, "problems": list(problems), "blockers": blockers}


def overall(analysis: dict, *, errors: dict[str, list], dry_run: bool) -> tuple[str, int]:
    if errors:
        return "fail", EXIT_FAIL
    if dry_run:
        return "dry_run", EXIT_PASS
    judged = [e["verdict"] for n, e in analysis["arms"].items() if n not in (BASELINE, TWIN) and e["verdict"] != "reported"]
    if "not met" in judged:
        return "fail", EXIT_FAIL
    if "cannot tell" in judged or analysis["arms"][TWIN]["verdict"] != "agrees":
        return "not_proven", EXIT_NOT_PROVEN
    return ("pass" if judged else "measured"), EXIT_PASS


# ---- the round layout ------------------------------------------------------------------------


def round_orders(names: list[str], rounds: int, seed: int) -> list[list[str]]:
    """Blocks of len(names) rounds; each block a cyclic Latin square of a fresh seeded permutation, rows shuffled.
    A longer run's first rounds are a shorter run's, so a resumed or extended run keeps its orders."""
    rng = random.Random(seed)
    orders: list[list[str]] = []
    k = len(names)
    while len(orders) < rounds:
        base = list(names)
        rng.shuffle(base)
        rows = [base[i:] + base[:i] for i in range(k)]
        rng.shuffle(rows)
        orders.extend(rows)
    return orders[:rounds]


# ---- the GPU is not shared -------------------------------------------------------------------


def _number(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def parse_smi_apps(text: str) -> list[dict]:
    apps = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) >= 2 and fields[0].isdigit():
            apps.append({"pid": int(fields[0]), "used_mib": _number(fields[1])})
    return apps


SMI_GPU_FIELDS = ("utilization.gpu", "memory.used", "temperature.gpu", "clocks.sm", "power.draw", "compute_mode")


def parse_smi_gpu(text: str) -> dict:
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    fields = [f.strip() for f in line.split(",")]
    fields += [""] * (len(SMI_GPU_FIELDS) - len(fields))
    keys = ("utilization_gpu", "memory_used_mib", "temperature_c", "clock_sm_mhz", "power_w")
    out: dict[str, Any] = {k: _number(v) for k, v in zip(keys, fields)}
    out["compute_mode"] = fields[5] or None
    return out


class GpuReader:
    """What another user of the GPU would show: its processes and its utilization, by NVML or nvidia-smi."""

    def __init__(self, uuid: str | None, index: int = 0) -> None:
        self.uuid, self.index, self.method, self._nvml, self._handle = uuid, index, None, None, None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = (pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid
                            else pynvml.nvmlDeviceGetHandleByIndex(index))
            self._nvml, self.method = pynvml, "nvml"
        except Exception:  # noqa: BLE001  # no NVML: try nvidia-smi
            if shutil.which("nvidia-smi"):
                self.method = "nvidia-smi"

    def _smi(self, query: str) -> str:
        target = self.uuid or str(self.index)
        done = subprocess.run(["nvidia-smi", "-i", target, query, "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30)
        if done.returncode:  # a failed query is a failed reading, never "no processes"
            raise RuntimeError(f"nvidia-smi exit {done.returncode}: {(done.stderr or done.stdout).strip()[:300]}")
        return done.stdout

    def sample(self, where: str) -> dict:
        reading: dict[str, Any] = {"where": where, "at": time.time(), "method": self.method, "processes": []}
        try:
            if self.method == "nvml":
                nv, h = self._nvml, self._handle
                seen = {}
                for getter in (nv.nvmlDeviceGetComputeRunningProcesses, nv.nvmlDeviceGetGraphicsRunningProcesses):
                    for p in getter(h):
                        used = getattr(p, "usedGpuMemory", None)
                        seen[p.pid] = {"pid": int(p.pid), "used_mib": None if used is None else used / 2 ** 20}
                reading["processes"] = list(seen.values())
                reading["utilization_gpu"] = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
                reading["memory_used_mib"] = nv.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20
                for key, call in (("temperature_c", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                                  ("clock_sm_mhz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                                  ("power_w", lambda: nv.nvmlDeviceGetPowerUsage(h) / 1000)):
                    try:
                        reading[key] = float(call())
                    except Exception:  # noqa: BLE001
                        reading[key] = None
            elif self.method == "nvidia-smi":
                reading["processes"] = parse_smi_apps(self._smi("--query-compute-apps=pid,used_memory"))
                reading.update(parse_smi_gpu(self._smi("--query-gpu=" + ",".join(SMI_GPU_FIELDS))))
        except Exception as error:  # noqa: BLE001  # a failed read is recorded, never taken as "nothing there"
            reading["error"] = f"{type(error).__name__}: {error}"
        return reading

    def idle(self, where: str) -> dict:
        """After a settle, three readings half a second apart: the most processes any showed, the least utilization
        (NVML's utilization is averaged over a past period, which may still hold the arm that just ended)."""
        time.sleep(SETTLE_SECONDS)
        readings = []
        for i in range(3):
            readings.append(self.sample(where))
            if i < 2:
                time.sleep(0.5)
        utilizations = [r["utilization_gpu"] for r in readings if r.get("utilization_gpu") is not None]
        most = max(readings, key=lambda r: len(r["processes"]))
        return {**readings[-1], "processes": most["processes"],
                "utilization_gpu": min(utilizations) if utilizations else None,
                "errors": [r["error"] for r in readings if "error" in r]}


def exclusivity(samples: list[dict], *, method: str | None, device_type: str) -> dict:
    """Whether anything but this run used the GPU. Between arms no process may be there (the driver holds no
    CUDA context); inside an arm only the arm itself."""
    if device_type != "cuda":
        return {"status": "not applicable", "usable": True, "method": method, "reasons": []}
    if method is None:
        return {"status": "not checked", "usable": False, "method": None,
                "reasons": ["no NVML and no nvidia-smi: the GPU's other processes could not be read"]}
    reasons = []
    for s in samples:
        n = len(s.get("processes", []))
        if s.get("error"):
            reasons.append(f"{s['where']}: the reading failed ({s['error']})")
        if s["where"].startswith("arm"):
            if n >= 2:
                reasons.append(f"{s['where']}: {n} processes on the GPU, this run's arm and {n - 1} other")
            continue
        if n >= 1:
            reasons.append(f"{s['where']}: {n} process(es) on the GPU while no arm of this run was running")
        util = s.get("utilization_gpu")
        if util is not None and util >= IDLE_UTILIZATION_LIMIT:
            reasons.append(f"{s['where']}: GPU {util:g}% busy while no arm of this run was running")
    if reasons:
        return {"status": "shared", "usable": False, "method": method, "reasons": reasons}
    if not any(len(s.get("processes", [])) >= 1 for s in samples if s["where"].startswith("arm")):
        # The author, Nov 26, 2026: such a run measures and reports, but scores no limit; the plan's condition
        # for these measurements is a GPU that is not shared.
        return {"status": "utilization only", "usable": False, "method": method,
                "reasons": ["the GPU was not shown to be unshared: the process list never showed this run's own arm "
                            "(a container's view?), so only the GPU's utilization while idle was checked"]}
    return {"status": "exclusive", "usable": True, "method": method, "reasons": []}


# ---- one arm, in its own process -------------------------------------------------------------


def _sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(spec: dict, device):
    """The model, its router spec and the batches: from fixed seeds, the same in every arm."""
    import torch

    import tensorquill

    torch.manual_seed(0)
    total = spec["warmup"] + spec["steps"]
    generator = torch.Generator().manual_seed(spec["data_seed"])
    if spec["model"] == "toy":
        from profile_overhead import _ToyMoE

        model = _ToyMoE().to(device).train()
        routers = {f"routers.{i}": tensorquill.Router(8, top_k=model.top_k) for i in range(len(model.routers))}
        return model, routers, torch.randint(0, 256, (total, *TOY_SHAPE), generator=generator).to(device)
    if spec["model"] == "granite":
        from types import SimpleNamespace

        from moe_dogfood import build_model as granite_model, granite_routers, load_batches

        model = granite_model()
        batches = load_batches(SimpleNamespace(batch_size=spec["batch_size"], seq_len=spec["seq_len"]), total)
        return model, granite_routers(model), batches
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(spec["model"], dtype=dtype).to(device).train()
    routers: dict = {}
    try:
        from moe_dogfood import granite_routers

        routers = granite_routers(model)
    except Exception:  # noqa: BLE001  # not a Granite MoE: no router spec
        routers = {}
    vocab = int(model.config.vocab_size)
    shape = (total, spec["batch_size"], spec["seq_len"])
    return model, routers, torch.randint(0, vocab, shape, generator=generator).to(device)


def run_arm(spec: dict) -> dict[str, Any]:
    import torch

    import tensorquill
    from granite_compiler_gate import counter_snapshot

    arm = spec["arm"]
    device = torch.device(spec["device"])
    out = Path(spec["result"])
    result: dict[str, Any] = {
        "arm": arm["name"], "round": spec["round"], "pid": os.getpid(), "device": str(device),
        "torch": torch.__version__, "tensorquill_file": tensorquill.__file__, "init": arm["init"],
        "env_applied": {k: os.environ.get(k) for k in arm["env"]}, "started": time.time(),
    }
    if spec.get("fail_baseline_for_test") and arm["name"] == BASELINE:
        raise RuntimeError("planted for a test: capture off cannot run here")
    model, routers, batches = build_model(spec, device)
    if arm["init"] is not None:
        check_routers([arm], model_has_routers=bool(routers))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=device.type == "cuda")
    session = None
    sink_path = out.parent / "capture" / f"{arm['name']}-r{spec['round']}.jsonl"  # TensorQuill's own files
    sink_path.parent.mkdir(exist_ok=True)
    if arm["init"] is not None:
        session = tensorquill.init(model, **init_kwargs(arm["init"], routers, sink=tensorquill.JsonlSink(sink_path),
                                                        run_id=f"{arm['name']}-r{spec['round']}"))
    mode = spec["compile"]
    backend = spec["backend"] or ("inductor" if device.type == "cuda" else "aot_eager")
    cuda_graphs = device.type == "cuda" and backend == "inductor" and mode in CUDA_GRAPH_MODES
    result.update(compile=mode, backend=None if mode == "none" else backend, cuda_graphs=cuda_graphs,
                  routers=len(routers))
    reader = GpuReader(spec.get("gpu_uuid")) if device.type == "cuda" else None
    gpu: list[dict] = []

    def step(runner, batch):
        optimizer.zero_grad(set_to_none=True)
        if cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        loss = runner(input_ids=batch, labels=batch, use_cache=False).loss
        loss.backward()
        if session is not None and arm["log_loss"]:
            session.log({"train/loss": loss})  # before optimizer.step(), where the step ends
        optimizer.step()
        return loss

    window, steps, warmup = spec["window"], spec["steps"], spec["warmup"]
    try:
        with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
            if mode == "none":
                runner = model
            else:
                runner = torch.compile(model, backend=backend, mode=None if mode == "default" or backend != "inductor"
                                       else mode, fullgraph=False)
            for batch in batches[:warmup]:
                loss = step(runner, batch)
            _sync(torch, device)
            gc.collect()
            warm = counter_snapshot()
            if reader:
                gpu.append(reader.sample("arm before timing"))
            window_ms: list[float] = []
            step_ms: list[float] = []
            events: list = []
            timed = batches[warmup:]
            for w in range(steps // window):
                _sync(torch, device)
                start = time.perf_counter()
                for batch in timed[w * window:(w + 1) * window]:
                    if device.type == "cuda":
                        begin = torch.cuda.Event(enable_timing=True)
                        begin.record()
                    else:
                        begin = time.perf_counter()
                    loss = step(runner, batch)
                    if device.type == "cuda":
                        end = torch.cuda.Event(enable_timing=True)
                        end.record()
                        events.append((begin, end))
                    else:
                        step_ms.append((time.perf_counter() - begin) * 1000)
                _sync(torch, device)
                window_ms.append((time.perf_counter() - start) * 1000)
            step_ms += [b.elapsed_time(e) for b, e in events]
            final = counter_snapshot()
            if reader:
                gpu.append(reader.sample("arm after timing"))
            last = float(loss.detach())
    finally:
        if session is not None:
            session.close()
    records = len(tensorquill.load_records(sink_path)) if session is not None else None
    slowest = max(range(len(step_ms)), key=step_ms.__getitem__) if step_ms else None
    result.update(
        state="measured", block_ms=sum(window_ms) / steps, window_ms=window_ms, step_ms=step_ms,
        step_median_ms=statistics.median(step_ms) if step_ms else None,
        slowest_step={"step": warmup + slowest, "ms": step_ms[slowest],
                      "over_median_ms": step_ms[slowest] - statistics.median(step_ms)} if slowest is not None else None,
        final_loss=last, loss_finite=math.isfinite(last), records=records,
        graph_breaks=final["graph_breaks"], recompiles_after_warmup=final["unique_graphs"] - warm["unique_graphs"],
        cudagraph_skips=final["cudagraph_skips"] if cuda_graphs else None, gpu_samples=gpu, ended=time.time(),
    )
    if not math.isfinite(last):
        result.update(state="error", error=f"the final loss is not finite ({last})")
    return result


def worker_main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec["result"])
    try:
        result = run_arm(spec)
    except Exception as error:  # noqa: BLE001  # every arm's result is written, a failure too
        result = {"arm": spec["arm"]["name"], "round": spec["round"], "pid": os.getpid(), "state": "error",
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    out.write_text(json.dumps(result, indent=1, default=str))
    shown = {k: v for k, v in result.items() if k not in ("step_ms", "traceback", "window_ms")}
    print(json.dumps(shown, indent=1, default=str))
    return 0 if result["state"] == "measured" else 1


def _nvml_uuid(uuid: Any) -> str | None:
    """NVML's and nvidia-smi's name for the device: `GPU-<uuid>` (or `MIG-<uuid>`), from PyTorch's."""
    if not uuid:
        return None
    text = str(uuid)
    return text if text.startswith(("GPU-", "MIG-")) else f"GPU-{text}"


def probe_main(device_arg: str) -> int:
    """The device's facts, from a short process of its own, so the driver never opens a CUDA context."""
    import platform

    import torch

    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    facts: dict[str, Any] = {"device_type": device_arg, "torch": torch.__version__, "cuda": torch.version.cuda,
                             "python": platform.python_version(), "platform": platform.platform()}
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print(json.dumps({"error": "--device cuda, and no CUDA GPU is visible"}))
            return EXIT_USAGE
        props = torch.cuda.get_device_properties(0)
        facts.update(device_name=props.name, capability=list(torch.cuda.get_device_capability(0)),
                     memory_mib=props.total_memory / 2 ** 20, devices_visible=torch.cuda.device_count(),
                     uuid=_nvml_uuid(getattr(props, "uuid", None)),
                     cudnn=torch.backends.cudnn.version())
        try:
            import pynvml

            pynvml.nvmlInit()
            facts["driver"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception:  # noqa: BLE001
            if shutil.which("nvidia-smi"):
                facts["driver"] = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True).stdout.strip().splitlines()[:1]
    elif device_arg == "mps":
        if not torch.backends.mps.is_available():
            print(json.dumps({"error": "--device mps, and MPS is not available"}))
            return EXIT_USAGE
        facts["device_name"] = "MPS"
    else:
        facts["device_name"] = f"CPU ({platform.machine() or platform.processor() or 'unknown'})"
    for package in ("transformers", "tensorquill"):
        try:
            module = __import__(package)
            facts[package] = getattr(module, "__version__", "?")
        except Exception:  # noqa: BLE001
            facts[package] = None
    print(json.dumps(facts))
    return 0


# ---- the driver ------------------------------------------------------------------------------


def _arm_env(arm: dict) -> dict[str, str]:
    env = dict(os.environ)
    found = importlib.util.find_spec("tensorquill")
    paths = [str(HERE)]
    if found and found.origin:
        paths.insert(0, str(Path(found.origin).resolve().parents[1]))  # the arms measure this tensorquill
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(arm["env"])
    return env


def _settings(args, arms: list[dict]) -> dict[str, Any]:
    return {"model": args.model, "device": args.device, "compile": args.compile, "backend": args.backend,
            "warmup": args.warmup, "steps": args.steps, "window": args.window, "batch_size": args.batch_size,
            "seq_len": args.seq_len, "seed": args.seed, "data_seed": args.data_seed, "confidence": args.confidence,
            "limit": args.limit, "arms": arms}


def _pct(x: float | None, signed: bool = True) -> str:
    if x is None:
        return "-"
    return f"{100 * x:+.2f}%" if signed else f"{100 * x:.2f}%"


def table(result: dict) -> str:
    facts, analysis = result["device"], result["analysis"]
    conf = analysis["confidence"]
    lines = [f"TensorQuill overhead harness | {facts.get('device_name')} ({facts.get('device_type')}) | torch "
             f"{facts.get('torch')} | model {result['settings']['model']} | compile {result['settings']['compile']} | "
             f"{result['rounds_done']} rounds | warm-up {result['settings']['warmup']}, timed {result['settings']['steps']} "
             f"steps in windows of {result['settings']['window']} | seed {result['seed']}", ""]
    lines.append(f"{'arm':<14}{'baseline':<12}{'rounds':>7}{'median ms':>11}{'vs baseline':>13}"
                 f"{f'{conf:.0%} interval':>24}{'limit':>9}  verdict")
    for arm in result["arms"]:
        e = analysis["arms"][arm["name"]]
        median_ms = "-" if e["median_ms"] is None else f"{e['median_ms']:.3f}"
        if arm["name"] == BASELINE:
            lines.append(f"{arm['name']:<14}{'-':<12}{e['rounds']:>7}{median_ms:>11}{'':>13}{'':>24}{'':>9}  baseline")
            continue
        interval = "none" if e["interval"] is None else f"[{_pct(e['interval'][0])}, {_pct(e['interval'][1])}]"
        lines.append(f"{arm['name']:<14}{e['baseline']:<12}{e['rounds']:>7}{median_ms:>11}{_pct(e['median']):>13}"
                     f"{interval:>24}{_pct(e['limit'], False):>9}  {e['verdict']}")
    lines.append("")
    for arm in result["arms"][1:]:
        lines.append(f"  {arm['name']}: {analysis['arms'][arm['name']]['why']}")
    noise = analysis["noise"]
    lines.append("")
    if noise["smallest_difference_told"] is None:
        lines.append("Noise: too few rounds for an interval on the identical capture-off arms; the smallest "
                     "difference this run could tell is not known.")
    else:
        lines.append(f"Noise: the identical capture-off arms {'agree' if noise['identical_arms_agree'] else 'DIFFER'}; "
                     f"the smallest difference this run could tell is about +/-{_pct(noise['smallest_difference_told'], False)}; "
                     f"paired SD {_pct(noise['paired_sd'], False)}.")
    if noise["rounds_needed"]:
        lines.append("Rounds this noise needs to call an arm at half the limit met (90% power): "
                     + ", ".join(f"{float(k):.1%} limit: {v}" for k, v in noise["rounds_needed"].items()))
    gpu = result["gpu_check"]
    lines.append(f"GPU check: {gpu['status']}" + (f" (by {gpu['method']})" if gpu["method"] else "") + "".join(f"\n  {r}" for r in gpu["reasons"]))
    for note in result.get("extended_after_a_look", []):
        lines.append(f"Rounds {note['from'] + 1} to {note['to']} were added after a look at the first {note['from']}: "
                     "the planned count was not kept, so read the interval with that in mind.")
    for problem in analysis["problems"]:
        lines.append(f"Problem: {problem}")
    for name, errs in result["errors"].items():
        lines.append(f"Error: {name}: {errs[0]}" + (f" (and {len(errs) - 1} more)" if len(errs) > 1 else ""))
    lines.append("")
    lines.append(f"Status: {result['status']} (exit {result['code']})")
    return "\n".join(lines)


def _write(out: Path, name: str, document: Any) -> None:
    (out / name).write_text(json.dumps(document, indent=1, default=str))


def driver_main(args) -> int:
    out = Path(args.out)
    default_limit = args.limit
    try:
        options = [parse_arm_option(text) for text in args.arm]
        arms = resolve_arms(args.arm_set, options, args.arms_file, default_limit=default_limit)
        check_alignment(arms, warmup=args.warmup, steps=args.steps, window=args.window)
        if args.model in ("toy", "granite"):
            check_routers(arms, model_has_routers=True)
    except (ArmError, OSError, ValueError) as error:
        print(f"overhead_harness: {error}", file=sys.stderr)
        return EXIT_USAGE
    settings = _settings(args, arms)
    extended: list[dict] = []
    if args.resume:
        if not (out / "plan.json").exists():
            print(f"overhead_harness: --resume, and {out} holds no plan.json", file=sys.stderr)
            return EXIT_USAGE
        plan = json.loads((out / "plan.json").read_text())
        if plan["settings"] != json.loads(json.dumps(settings)):
            changed = sorted(k for k in settings if plan["settings"].get(k) != json.loads(json.dumps(settings[k])))
            print(f"overhead_harness: --resume with other settings than the plan's ({', '.join(changed)})", file=sys.stderr)
            return EXIT_USAGE
        extended = plan.get("extended_after_a_look", [])
        if args.rounds < plan["rounds"]:
            print(f"overhead_harness: the plan has {plan['rounds']} rounds; --rounds may not be fewer", file=sys.stderr)
            return EXIT_USAGE
        if args.rounds > plan["rounds"]:
            if not args.extend:
                print(f"overhead_harness: the plan has {plan['rounds']} rounds; more than planned needs --extend, "
                      "which records that rounds were added after a look", file=sys.stderr)
                return EXIT_USAGE
            extended = extended + [{"from": plan["rounds"], "to": args.rounds}]
    elif out.exists():
        print(f"overhead_harness: {out} exists (use --resume to finish its planned rounds)", file=sys.stderr)
        return EXIT_USAGE

    probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--probe", args.device],
                           env=_arm_env({"env": {}}), capture_output=True, text=True, timeout=600)
    try:
        facts = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        facts = {"error": f"the device probe failed (exit {probe.returncode}): {probe.stderr.strip()[-2000:]}"}
    if "error" in facts:
        print(f"overhead_harness: {facts['error']}", file=sys.stderr)
        return EXIT_USAGE
    device_type = facts["device_type"]
    dry_run = device_type != "cuda"

    (out / "arms").mkdir(parents=True, exist_ok=True)
    _write(out, "plan.json", {"settings": settings, "rounds": args.rounds, "extended_after_a_look": extended})
    names = [a["name"] for a in arms]
    by_name = {a["name"]: a for a in arms}
    orders = round_orders(names, args.rounds, args.seed)
    reader = GpuReader(facts.get("uuid")) if device_type == "cuda" else None
    samples_path = out / "gpu-samples.jsonl"

    def record_sample(sample: dict) -> None:
        with samples_path.open("a") as handle:
            handle.write(json.dumps(sample, default=str) + "\n")

    print(f"torch {facts['torch']} on {facts.get('device_name')} ({device_type}) | model {args.model} | compile "
          f"{args.compile} | {args.rounds} rounds x {len(arms)} arms ({', '.join(names)}) | seed {args.seed}", flush=True)
    if reader:
        record_sample(reader.idle("before the run"))
    results: dict[tuple[str, int], dict] = {}
    stopped: str | None = None
    done_before = 0
    for r, order in enumerate(orders):
        for name in order:
            path = out / "arms" / f"{name}-r{r}.json"
            if args.resume and path.exists():
                previous = json.loads(path.read_text())
                if previous.get("state") == "measured":
                    results[(name, r)] = previous
                    done_before += 1
                    continue
            arm = by_name[name]
            spec_path = out / "arms" / f"{name}-r{r}.spec.json"
            spec = {**{k: settings[k] for k in ("model", "compile", "backend", "warmup", "steps", "window",
                                                 "batch_size", "seq_len", "data_seed")},
                    "device": device_type, "arm": arm, "round": r, "result": str(path), "gpu_uuid": facts.get("uuid"),
                    "fail_baseline_for_test": args.fail_baseline_for_test}
            spec_path.write_text(json.dumps(spec, indent=1))
            started = time.time()
            with (out / "arms" / f"{name}-r{r}.log").open("w") as log:
                try:
                    code: Any = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)],
                                               env=_arm_env(arm), stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.arm_timeout).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            try:
                result = json.loads(path.read_text())
            except (OSError, ValueError):
                result = {"arm": name, "round": r, "state": "error",
                          "error": f"the arm process wrote no result (exit {code}); see {path.with_suffix('.log').name}"}
                path.write_text(json.dumps(result, indent=1))
            result["exit"] = code
            results[(name, r)] = result
            for sample in result.get("gpu_samples", []):
                record_sample({**sample, "arm": name, "round": r})
            if reader:
                record_sample(reader.idle(f"after {name} round {r + 1}"))
            shown = f"{result['block_ms']:.3f} ms/step" if result.get("state") == "measured" else result.get("error")
            print(f"{name} round {r + 1}: {result.get('state')} {shown} ({time.time() - started:.0f} s, exit {code})",
                  flush=True)
            if name == BASELINE and r == 0 and result.get("state") != "measured":
                stopped = (f"capture off could not run in this mode in round 1, so nothing can be measured against it "
                           f"here: {result.get('error')}")
                break
        if stopped:
            break
    if done_before:
        print(f"resumed: {done_before} arm runs were already done", flush=True)
    if reader and not stopped:
        record_sample(reader.idle("after the run"))

    samples = [json.loads(line) for line in samples_path.read_text().splitlines()] if samples_path.exists() else []
    gpu_check = exclusivity(samples, method=reader.method if reader else None, device_type=device_type)
    problems: list[str] = []
    if dry_run:
        problems.append(f"dry run on {device_type}: timings off a CUDA GPU say nothing about one")
    if not gpu_check["usable"]:
        problems += [f"the GPU check: {reason}" for reason in gpu_check["reasons"]] or ["the GPU check failed"]
    if stopped:
        problems.append(stopped)
    times: dict[int, dict[str, float]] = {}
    errors: dict[str, list[str]] = {}
    for (name, r), res in sorted(results.items(), key=lambda item: (item[0][1], item[0][0])):
        if res.get("state") == "measured":
            times.setdefault(r, {})[name] = res["block_ms"]
        elif not (stopped and name == BASELINE):
            errors.setdefault(name, []).append(f"round {r + 1}: {res.get('error')}")
    analysis = analyze(times, arms, confidence=args.confidence, problems=problems)
    if stopped:
        status, code = "not_proven", EXIT_NOT_PROVEN
    else:
        status, code = overall(analysis, errors=errors, dry_run=dry_run)
    result_doc = {
        "status": status, "code": code, "device": facts, "settings": settings, "seed": args.seed, "arms": arms,
        "orders": orders, "rounds_planned": args.rounds, "rounds_done": len(times),
        "extended_after_a_look": extended, "gpu_check": gpu_check, "analysis": analysis, "errors": errors,
        "env_seen": {k: v for k, v in os.environ.items() if k.startswith(("TORCH", "PYTORCH", "CUDA", "TENSORQUILL", "NCCL"))},
        "rounds": {f"{name}-r{r}": {k: v for k, v in res.items() if k != "traceback"} for (name, r), res in results.items()},
    }
    _write(out, "result.json", result_doc)
    reasons = ([stopped] if stopped else []) + analysis["blockers"] + [e for errs in errors.values() for e in errs]
    _write(out, "verdict.json", {
        "code": code, "status": status, "reasons": reasons, "limit": args.limit, "confidence": args.confidence,
        "gpu_check": gpu_check["status"], "noise": analysis["noise"], "extended_after_a_look": extended,
        "arms": {name: {k: e.get(k) for k in ("baseline", "rounds", "median", "interval", "coverage", "limit", "verdict", "why")}
                 for name, e in analysis["arms"].items() if name != BASELINE},
    })
    text = table(result_doc)
    (out / "table.txt").write_text(text + "\n")
    print("\n" + text + f"\n\nResult: {out / 'result.json'}")
    return code


def plan_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="overhead_harness.py plan",
                                     description="Rounds this harness needs, by formula and by simulating its interval.")
    parser.add_argument("--sd", type=float, required=True, help="paired SD of one round's difference, e.g. 0.011")
    parser.add_argument("--limit", type=float, default=DEFAULT_LIMIT)
    parser.add_argument("--true", type=float, help="the arm's true overhead (default: half the limit)")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    true = args.limit / 2 if args.true is None else args.true
    n = rounds_needed(args.sd, args.limit - true, args.confidence)
    power = simulate_power(n, args.sd, true, args.limit, confidence=args.confidence, twin=True)
    searched = rounds_by_simulation(args.sd, true, args.limit, confidence=args.confidence)
    print(f"paired SD {args.sd:.2%}, limit {args.limit:.2%}, true overhead {true:.2%}: the formula gives {n} rounds "
          f"(simulated: met in {power:.0%} of runs, counting the identical-arm rule); 90% of runs met from "
          f"{searched if searched else 'more than 5000'} rounds by simulation")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="new directory for the run (or one to --resume)")
    parser.add_argument("--model", default="toy", help="toy (built in, CPU allowed), granite, or a Hugging Face id")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--compile", choices=COMPILE_MODES, default="default",
                        help="none (eager) or a torch.compile mode; default: Inductor without CUDA graphs")
    parser.add_argument("--backend", help="torch.compile backend (default: inductor on CUDA, aot_eager elsewhere)")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200, help="timed steps per arm and round")
    parser.add_argument("--window", type=int, default=FLUSH_EVERY, help="steps between device waits and timestamps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--arm-set", default="routers", help=f"one of {', '.join(ARM_SETS)}")
    parser.add_argument("--arm", action="append", default=[],
                        help="NAME=JSON, an arm of its own (repeatable); a detector or switch such as K3 is judged "
                             'over the same capture without it: "baseline": "routers"')
    parser.add_argument("--arms-file", type=Path, help="a JSON list of arms")
    parser.add_argument("--limit", type=float, default=DEFAULT_LIMIT, help="the default limit, e.g. 0.01 or 0.005")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0, help="for the arms' order in each round")
    parser.add_argument("--data-seed", type=int, default=0, help="for the toy and random-token batches")
    parser.add_argument("--arm-timeout", type=float, default=3600)
    parser.add_argument("--resume", action="store_true", help="finish the planned rounds of an existing --out")
    parser.add_argument("--extend", action="store_true", help="with --resume: allow more rounds than planned (recorded)")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--probe", help=argparse.SUPPRESS)
    parser.add_argument("--fail-baseline-for-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["plan"]:
        return plan_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main(args.worker)
    if args.probe:
        return probe_main(args.probe)
    if args.out is None:
        parser.error("--out is required")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if not 0 < args.confidence < 1:
        parser.error("--confidence is a fraction, such as 0.95")
    if args.extend and not args.resume:
        parser.error("--extend goes with --resume")
    return driver_main(args)


if __name__ == "__main__":
    sys.exit(main())
