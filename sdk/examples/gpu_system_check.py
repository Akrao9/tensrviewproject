"""GPU memory, NVML and data-wait readings, checked against readings taken alongside in the same run.

    python examples/gpu_system_check.py --out runs/system                 # CUDA, nvidia-ml-py, nvidia-smi
    python examples/gpu_system_check.py --out runs/system-cpu             # CPU: data wait only, exit 3

Trains the small MoE of `gpu_check.py` from a `torch.utils.data.DataLoader` with
`tensorquill.init(..., system=True, data_wait=True)`, flushing every `--every` steps, and checks
what TensorQuill recorded against what the script reads itself:

- **Data wait.** The loader sleeps a known time before each batch: `--wait-ms` on every step, four
  times that on steps 5, 15, 25, ... First, before `tensorquill.init`, the same loop (the same model,
  optimizer and step, through `train_steps`) runs `--off-steps` steps (`OFF_STEPS`, 100) with capture
  off, on a loader of its own with the same sleeps, and the script's own timer around each `next()`
  gives **the loader's own delay**: for each sleep length, the median over those steps of the time in
  `next()` less the sleep, and never less than 0 (`loader_own_delay`). Then, with capture on,
  TensorQuill's `data_wait` on every step must be at least the sleep; at most the sleep plus the
  loader's own delay for that sleep length plus `--wait-slack-ms` (15 ms) on 98% of steps, the steps
  over it listed with what was allowed; at most the sleep plus `OUTLIER_MS` (100 ms) on every step;
  within 5 ms of the script's own timer around the same `next()`; `step_time` at least `data_wait`.
  The measured delay is kept in `system.json` (`data_wait.loader_own_delay`). This part is a real
  check on any machine.

  Why so. The owner's decision of Nov 27, 2026: the loader's own delay is measured with capture off in
  the same run and 15 ms is allowed on top of it, in place of the sleep plus 15 ms. On the A100 run of
  Nov 23 (`a2e652d`) all 300 steps were over the sleep plus 15 ms, by a median 24.5 ms; capture off
  showed the same 24 ms, and `data_wait` equalled the script's own timer: the loader was late, not the
  capture. What is judged is what capture adds to the loader's time, so the loader's time is taken on
  the same machine, process, model and loop. The capture-off steps run before `init` because the
  hooks `init` registers are never removed, so no part of a run after it is capture off. 100 steps
  give 90 of the short sleep and 10 of the long one, enough for a median of each; they cost the time
  of 100 steps. The median, because the first step of a process is slow (on the A100 run's loader
  check its whole time was 240 ms over with capture off) and these steps come first, so they also take the warm-up: one
  or a few slow steps do not move a median of 90, or of 10. Per sleep length, because the long sleeps
  are 10% of steps, more than the 2% the 98% line leaves out: a delay that grew with the sleep would
  fail every long step under one number for both; with a delay that does not, the two medians agree.
  Never below 0, so a loader quicker than its sleep (not seen) is allowed what the old rule allowed.
  The 100 ms bound on every step is still counted from the sleep alone, as before: the decision named
  only the 15 ms. Limits: the capture-off steps come first, so a delay that changes over a run (the
  machine's load, a collection the process owes) is measured at its start only; the `nvidia-smi`
  thread does not run during them.

  Before the capture-off steps were added (lane AI-1's runs, Nov 22, 2026), on the Mac this script was
  written on, with TensorQuill on, step 100 (at `--every 50`) spent 34 to 73 ms more in `next()` than the sleep, by both timers, in every run; with TensorQuill off, in the same
  loop, it did not: a delay of the training thread (`loader_delay_check.py`), not a fault of the timer.
  With the capture-off steps first, one run on that Mac (Nov 27, 2026, under heavy load) had its one
  such step at step 219, 94 ms over its sleep. One such step is inside the 98% line.
- **CUDA memory** (the record's `system.cuda`, at each flush). Right after `optimizer.step()` on
  that step the script reads `torch.cuda.memory_allocated`, `memory_reserved` and
  `max_memory_allocated`. These are the same allocator's counters, so they check that TensorQuill
  read them and when, within 2% or 32 MiB. The independent ones: `total` against the device's
  properties (exactly) and `nvidia-smi`'s `memory.total` (within 3%: the driver keeps some back), and
  `nvidia-smi`'s `memory.used`, which must be at least TensorQuill's `reserved` less 2%.
- `nvidia-smi` must give usable samples: if it errors on every call, or reports `memory.used` as
  `[N/A]`, the comparisons that need it are not proven (exit 3), with its error, not failed.
- **NVML** (`system.nvml`, the `nvml` extra: `pip install nvidia-ml-py`). A thread runs `nvidia-smi`
  for this GPU, chosen by UUID, every `--smi-interval` seconds during training. Each NVML reading
  is paired with the `nvidia-smi` sample nearest its `read_at` (within 1 s): temperature within 3 C,
  SM clock within 15%, the enforced power limit within 1 W, the volatile ECC counts equal where
  `nvidia-smi` reports them. NVML readings are one flush late and the last is dropped, so `--steps`
  must give at least four flushes.

  **Power** is compared over a steady window instead (`check_power`): TensorQuill's median
  `power_w` over its NVML readings against `nvidia-smi`'s median `power.draw` over its samples taken
  from the first reading's `read_at` to the last's, both ends included; within 25% or 25 W
  (`POWER_REL`, `POWER_ABS_W`, the per-reading rule's figures kept). It needs at least
  `POWER_MIN_READINGS` (3) readings with power and `POWER_MIN_SAMPLES` (5) samples with power in
  the window, or power is not proven. The window, both medians and every value in them are kept in
  `system.json` (`nvml.power_window`).

  Why so. The owner's decision of Nov 27, 2026: medians over a steady window are compared with
  `nvidia-smi`'s, and the 25% stays. On the A100 run of Nov 23 two readings were apart from their
  nearest sample (314 against 164 W, 214 against 117 W) while the load changed: two instruments read at different
  instants. The window is steady by construction, not by a measurement: it starts at the first NVML
  reading, which is asked for at the first flush, so at least `--off-steps` plus `--every` steps into
  the loop, past the first step's one-time work and the GPU's rise from idle; it ends at the last
  reading TensorQuill kept, asked for at the second-last flush, so the idle GPU after the loop is
  outside it. Inside it every step does the same work and the sleeps repeat every 10 steps, so the
  load over a second or more is the same throughout. A median of 3 or more readings, and of 5 or more
  samples, is not moved by one value taken during a transient (a flush, a change of power state).
  Limit, not measured: TensorQuill reads NVML right after a flush, at the same point of a step each
  time, while `nvidia-smi`'s samples fall anywhere in a step. Where a board reports instantaneous
  power rather than an average over a second, a load that changes within a step can keep the two
  medians apart. NVML's header says of `nvmlDeviceGetPowerUsage`, which TensorQuill calls: "On Ampere
  (except GA100) or newer GPUs, the API returns power averaged over 1 sec interval. On GA100 and older
  architectures, instantaneous power is returned" (read in the copy NVIDIA publishes with go-nvml,
  Nov 27, 2026). The A100 is GA100. How `nvidia-smi`'s `power.draw` is averaged on it was not read.

**Neither rule has run on a GPU.** Their windows and statistics were fixed from the reasoning above
before any measurement, and the recorded A100 numbers were not used to choose them.

The result is `system.json`, with every pairing. Exit status: 1 if any check fails; else 3 (not
proven) when a part could not be checked here: no CUDA, `nvidia-ml-py` absent, or no `nvidia-smi`;
else 0. On a CPU the script says NVML and CUDA memory are absent, checks that TensorQuill said so
too, checks the data wait, and exits 3.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_check import MoE, routers  # noqa: E402

import tensorquill  # noqa: E402

MIB = 1 << 20
SMI_FIELDS = ("temperature.gpu", "clocks.sm", "power.draw", "enforced.power.limit", "memory.total", "memory.used",
              "ecc.errors.uncorrected.volatile.total", "ecc.errors.corrected.volatile.total")
NOT_PROVEN = 3
OUTLIER_MS = 100.0  # no step's data_wait may be later than its sleep by more than this
OWN_TIMER_S = 0.005  # TensorQuill's data_wait against the script's own timer around the same next()
WITHIN_SLACK = 0.98  # the share of steps whose data_wait must be within the slack
OFF_STEPS = 100  # capture-off steps before init that measure the loader's own delay (--off-steps)
MIN_OFF_STEPS = 20  # so that each sleep length (one step in ten is long) has at least two
POWER_REL, POWER_ABS_W = 0.25, 25.0  # the medians' tolerance: 25% or 25 W, whichever is larger
POWER_MIN_READINGS = 3  # NVML readings with power in the window
POWER_MIN_SAMPLES = 5  # nvidia-smi samples with power.draw in the window
POWER_NAME = "power_w median against nvidia-smi power.draw median over the steady window"


class SleepyBatches(torch.utils.data.Dataset):
    """Batch `i` is made after sleeping `delay(i)` seconds: a known data wait for each step."""

    def __init__(self, steps: int, batch: int, dim: int, wait_s: float) -> None:
        self.steps, self.batch, self.dim, self.wait_s = steps, batch, dim, wait_s

    def delay(self, i: int) -> float:
        return self.wait_s * (4 if i % 10 == 5 else 1)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, i: int) -> torch.Tensor:
        time.sleep(self.delay(i))
        return torch.randn(self.batch, self.dim, generator=torch.Generator().manual_seed(i))


class Smi:
    """`nvidia-smi` for one GPU, sampled on a thread of its own, each sample stamped with wall time."""

    def __init__(self, uuid: str, interval: float) -> None:
        self.cmd = ["nvidia-smi", "-i", uuid, f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits"]
        self.interval, self.samples, self.errors = interval, [], []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="smi", daemon=True)

    def sample(self) -> dict[str, Any] | None:
        before = time.time()
        try:
            out = subprocess.run(self.cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except (subprocess.SubprocessError, OSError) as error:
            self.errors.append(f"{type(error).__name__}: {error}")
            return None
        values = [v.strip() for v in out.splitlines()[0].split(",")]
        row: dict[str, Any] = {"t": (before + time.time()) / 2}
        for field, value in zip(SMI_FIELDS, values):
            try:
                row[field] = float(value)
            except ValueError:
                row[field] = None  # "[N/A]" or "[Not Supported]"
        self.samples.append(row)
        return row

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "Smi":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(15)

    def nearest(self, t: float) -> dict[str, Any] | None:
        return min(self.samples, key=lambda s: abs(s["t"] - t), default=None)


def within(a: float, b: float, *, rel: float = 0.0, absolute: float = 0.0) -> bool:
    return abs(a - b) <= max(rel * abs(b), absolute)


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.not_proven: list[str] = []

    def add(self, part: str, name: str, ok: bool, detail: str) -> None:
        self.items.append({"part": part, "name": name, "ok": bool(ok), "detail": detail})

    def unproven(self, part: str, why: str) -> None:
        self.not_proven.append(f"{part}: {why}")


def train_steps(model, optimizer, loader, device, steps: int, after=None) -> dict[int, float]:
    """`steps` steps of the loop, the same with capture off and on; returns the script's own timer
    around each `next()` on the loader, by step. `after(step)` runs right after `optimizer.step()`."""
    own: dict[int, float] = {}
    batches = iter(loader)
    for step in range(steps):
        began = time.perf_counter()
        x = next(batches)  # TensorQuill times this call too, when it watches the loaders
        own[step] = time.perf_counter() - began
        x = x.to(device, non_blocking=True)
        loss = (model(x) - x.sum(dim=1, keepdim=True)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if after is not None:
            after(step)
    return own


def loader_own_delay(own: dict[int, float], data: SleepyBatches) -> dict[float, float]:
    """The loader's own delay, from the script's timer around `next()` with capture off: for each sleep
    length (seconds), the median of the time in `next()` less the sleep, never less than 0."""
    by_sleep: dict[float, list[float]] = {}
    for step, took in own.items():
        by_sleep.setdefault(data.delay(step), []).append(took - data.delay(step))
    return {sleep: max(statistics.median(excess), 0.0) for sleep, excess in sorted(by_sleep.items())}


def own_delay_summary(own: dict[int, float], data: SleepyBatches) -> dict:
    """What `system.json` keeps of the capture-off steps."""
    delay = loader_own_delay(own, data)
    rows = []
    for sleep in delay:
        excess = [took - sleep for step, took in own.items() if data.delay(step) == sleep]
        rows.append({"sleep_ms": round(sleep * 1e3, 6), "steps": len(excess), "delay_ms": delay[sleep] * 1e3,
                     "median_ms": statistics.median(excess) * 1e3, "max_ms": max(excess) * 1e3})
    return {"steps": len(own), "statistic": "per sleep length, the median of the time in next() less the sleep, "
            "at least 0, over steps run with capture off before init", "by_sleep": rows,
            "next_ms": {step: took * 1e3 for step, took in own.items()}}


def check_data_wait(checks: Checks, records: list[dict], data: SleepyBatches, slack_s: float,
                    own: dict[int, float] | None = None, own_delay: dict[float, float] | None = None) -> dict:
    """`own`: the script's own timer around each `next()` on the loader, by step. `own_delay`: the
    loader's own delay by sleep length (`loader_own_delay`), allowed on top of the sleep before the
    slack; a sleep length it does not hold, or none given, is allowed none."""
    allowed = lambda want: want + (own_delay or {}).get(want, 0.0) + slack_s  # noqa: E731
    by_step = {r["step"]: r for r in records}
    low, high, missing, short = [], [], [], []
    for step in range(data.steps):
        r = by_step.get(step)
        if r is None or not isinstance(r.get("data_wait"), (int, float)):
            missing.append(step)
            continue
        want, got = data.delay(step), r["data_wait"]
        if got < want:
            low.append((step, got, want))
        elif got > allowed(want):
            high.append((step, got, want))
        if isinstance(r.get("step_time"), (int, float)) and r["step_time"] < got:
            short.append(step)
    checks.add("data_wait", "every step has a data_wait", not missing, f"{len(missing)} without, first {missing[:3]}")
    checks.add("data_wait", "data_wait is at least the injected sleep", not low,
               f"{len(low)} below" + (f", first step {low[0][0]}: {low[0][1] * 1e3:.2f} ms < {low[0][2] * 1e3:.2f} ms" if low else ""))
    late = [(step, got, want) for step, got, want in high if got > want + OUTLIER_MS / 1000]
    listed = ", ".join(f"step {st}: {g * 1e3:.1f} ms for {w * 1e3:.0f} ms, {allowed(w) * 1e3:.1f} allowed" for st, g, w in high[:5])
    measured = "; the loader's own delay with capture off: " + (", ".join(
        f"{d * 1e3:.2f} ms at {s * 1e3:g} ms sleeps" for s, d in own_delay.items()) if own_delay else "not measured, 0 allowed")
    checks.add("data_wait", f"data_wait is at most the sleep plus the loader's own delay plus {slack_s * 1e3:g} ms "
               f"on {100 * WITHIN_SLACK:g}% of steps", len(high) <= (1 - WITHIN_SLACK) * data.steps,
               f"{len(high)} of {data.steps} above" + (f": {listed}" if high else "") + measured)
    checks.add("data_wait", f"data_wait is at most the sleep plus {OUTLIER_MS:g} ms on every step", not late,
               f"{len(late)} above" + (f", first step {late[0][0]}: {late[0][1] * 1e3:.1f} ms" if late else ""))
    checks.add("data_wait", "step_time is at least data_wait", not short, f"{len(short)} steps shorter, first {short[:3]}")
    apart = []
    if own is not None:
        apart = [(s, by_step[s]["data_wait"], own[s]) for s in own
                 if s in by_step and isinstance(by_step[s].get("data_wait"), (int, float))
                 and abs(by_step[s]["data_wait"] - own[s]) > OWN_TIMER_S]
        checks.add("data_wait", f"data_wait is within {OWN_TIMER_S * 1e3:g} ms of the script's own timer around next()",
                   not apart and len(own) == data.steps,
                   f"{len(apart)} apart" + (f", first step {apart[0][0]}: {apart[0][1] * 1e3:.2f} ms recorded, "
                                            f"{apart[0][2] * 1e3:.2f} ms timed" if apart else ""))
    long_steps = [s for s in range(data.steps) if s % 10 == 5 and s in by_step]
    excess = sorted((r["data_wait"] - data.delay(s)) * 1e3 for s, r in by_step.items() if isinstance(r.get("data_wait"), (int, float)))
    return {"steps": data.steps, "long_wait_steps": len(long_steps),
            "excess_ms_median": excess[len(excess) // 2] if excess else None, "excess_ms_max": excess[-1] if excess else None,
            "over_slack": [{"step": st, "data_wait_ms": g * 1e3, "injected_ms": w * 1e3, "allowed_ms": allowed(w) * 1e3,
                            "own_timer_ms": own[st] * 1e3 if own and st in own else None} for st, g, w in high],
            "sample": {s: {"injected_ms": data.delay(s) * 1e3, "data_wait_ms": (by_step[s].get("data_wait") or 0) * 1e3}
                       for s in list(by_step)[:3] + long_steps[:2]}}


def check_memory(checks: Checks, systems: dict[int, dict], alongside: dict[int, dict], device, smi: Smi | None,
                 total: int | None = None) -> list:
    """`total`: the device's memory from its properties (read here when not given)."""
    pairs = []
    if device.type != "cuda":
        notes = [s["cuda"] for s in systems.values() if "cuda" in s]
        said = bool(notes) and "unavailable" in notes[0]
        checks.add("cuda_memory", "TensorQuill says CUDA memory is unavailable off CUDA", said, str(notes[:1]))
        checks.unproven("cuda_memory", "no CUDA device here")
        return pairs
    total = torch.cuda.get_device_properties(device).total_memory if total is None else total
    readings = {step: s["cuda"] for step, s in systems.items() if isinstance(s.get("cuda"), dict) and "allocated" in s["cuda"]}
    unusable: list[int] = []
    checks.add("cuda_memory", "a CUDA memory reading at every flush", set(readings) == set(alongside),
               f"recorded at steps {sorted(readings)}, read alongside at {sorted(alongside)}")
    tol = lambda v: max(0.02 * v, 32 * MIB)  # noqa: E731
    for step in sorted(set(readings) & set(alongside)):
        tq, own = readings[step], alongside[step]
        row = {"step": step, "tensorquill": tq, "torch_cuda": own}
        for key, mine in (("allocated", own["allocated"]), ("reserved", own["reserved"])):
            checks.add("cuda_memory", f"step {step}: {key} against torch.cuda", abs(tq[key] - mine) <= tol(mine),
                       f"{tq[key] / MIB:.1f} MiB against {mine / MIB:.1f} MiB")
        checks.add("cuda_memory", f"step {step}: peak_allocated against torch.cuda.max_memory_allocated",
                   tq["peak_allocated"] <= own["peak"] and own["peak"] - tq["peak_allocated"] <= tol(own["peak"]),
                   f"{tq['peak_allocated'] / MIB:.1f} MiB against {own['peak'] / MIB:.1f} MiB")
        checks.add("cuda_memory", f"step {step}: total against the device's properties", tq["total"] == total,
                   f"{tq['total']} against {total}")
        sample = smi.nearest(own["t"]) if smi else None
        if smi is not None and (sample is None or sample.get("memory.used") is None):
            unusable.append(step)
        if sample and sample.get("memory.used") is not None:
            used = sample["memory.used"] * MIB
            checks.add("cuda_memory", f"step {step}: nvidia-smi memory.used covers TensorQuill's reserved",
                       used >= 0.98 * tq["reserved"], f"used {used / MIB:.0f} MiB, reserved {tq['reserved'] / MIB:.0f} MiB")
            checks.add("cuda_memory", f"step {step}: total against nvidia-smi memory.total",
                       within(tq["total"], sample["memory.total"] * MIB, rel=0.03),
                       f"{tq['total'] / MIB:.0f} MiB against {sample['memory.total']:.0f} MiB")
            row["nvidia_smi"] = sample
        pairs.append(row)
    if smi is None:
        checks.unproven("cuda_memory", "nvidia-smi is not on PATH: memory.used and memory.total not compared")
    elif unusable:
        checks.unproven("cuda_memory", f"no usable nvidia-smi memory sample at steps {unusable[:5]}"
                        + (f" ({smi.errors[0]})" if smi.errors else " (memory.used [N/A])"))
    return pairs


def check_power(checks: Checks, readings: list[dict], smi: Smi) -> dict | None:
    """TensorQuill's median `power_w` against `nvidia-smi`'s median `power.draw` over the steady
    window, from the first NVML reading's `read_at` to the last's (the module's docstring says why).
    None when either side has no power value at all, as a per-reading pair was left out before."""
    power = sorted((r for r in readings if isinstance(r.get("power_w"), (int, float))), key=lambda r: r["read_at"])
    drawn = [s for s in smi.samples if isinstance(s.get("power.draw"), (int, float))]
    if not power or not drawn:
        return None  # not read by TensorQuill (its `unavailable` list says why) or [N/A] in nvidia-smi
    start, end = power[0]["read_at"], power[-1]["read_at"]
    inside = [s for s in drawn if start <= s["t"] <= end]
    window: dict[str, Any] = {"start": start, "end": end, "readings": len(power),
                              "tensorquill_w": [r["power_w"] for r in power],
                              "nvidia_smi_w": [s["power.draw"] for s in inside],
                              "tensorquill_median_w": None, "nvidia_smi_median_w": None}
    if len(power) < POWER_MIN_READINGS or len(inside) < POWER_MIN_SAMPLES:
        checks.unproven("nvml", f"power: {len(power)} NVML readings and {len(inside)} nvidia-smi samples in the "
                        f"steady window ({end - start:.1f} s); at least {POWER_MIN_READINGS} and "
                        f"{POWER_MIN_SAMPLES} are needed")
        return window
    mine = statistics.median(window["tensorquill_w"])
    theirs = statistics.median(window["nvidia_smi_w"])
    window.update(tensorquill_median_w=mine, nvidia_smi_median_w=theirs)
    checks.add("nvml", POWER_NAME, within(mine, theirs, rel=POWER_REL, absolute=POWER_ABS_W),
               f"{mine:.1f} W against {theirs:.1f} W: medians of {len(power)} readings and {len(inside)} samples "
               f"over {end - start:.1f} s (within {100 * POWER_REL:g}% or {POWER_ABS_W:g} W)")
    return window


def check_nvml(checks: Checks, systems: dict[int, dict], device, smi: Smi | None) -> dict:
    """Returns `pairs`, each reading with its nearest `nvidia-smi` sample, and `power_window`."""
    nothing: dict[str, Any] = {"pairs": [], "power_window": None}
    notes = [s["nvml"] for s in systems.values() if "nvml" in s]
    unavailable = [n["unavailable"] for n in notes if isinstance(n.get("unavailable"), str)]
    if device.type != "cuda":
        checks.add("nvml", "TensorQuill says NVML is unavailable off CUDA", bool(unavailable), str(unavailable[:1]))
        checks.unproven("nvml", "NVML is absent: no CUDA device here")
        return nothing
    if unavailable:
        # Absent here (no nvidia-ml-py, no driver): not a failure of TensorQuill's, and nothing proven.
        checks.unproven("nvml", f"TensorQuill: {unavailable[0]}")
        return nothing
    readings = [n for n in notes if "read_at" in n]
    checks.add("nvml", "NVML readings on at least three flushes", len(readings) >= 3, f"{len(readings)} readings")
    if smi is None:
        checks.unproven("nvml", "nvidia-smi is not on PATH: nothing to compare NVML with")
        return nothing
    if not smi.samples:
        # The harness's failure (a query field this driver does not know, say), not TensorQuill's.
        checks.unproven("nvml", f"nvidia-smi gave no usable sample: {smi.errors[0] if smi.errors else 'no output'}")
        return nothing
    pairs = []
    for reading in readings:
        sample = smi.nearest(reading["read_at"])
        if sample is None or abs(sample["t"] - reading["read_at"]) > 1.0:
            checks.add("nvml", "an nvidia-smi sample within 1 s of the reading", False,
                       f"nearest {abs(sample['t'] - reading['read_at']):.2f} s away" if sample else "no sample")
            continue
        at = f"reading at {time.strftime('%H:%M:%S', time.localtime(reading['read_at']))}"
        compare = (
            ("temperature_c", "temperature.gpu", dict(absolute=3.0)),
            ("sm_clock_mhz", "clocks.sm", dict(rel=0.15)),
            ("power_limit_w", "enforced.power.limit", dict(absolute=1.0)),
            ("ecc_uncorrected", "ecc.errors.uncorrected.volatile.total", dict(absolute=0.0)),
            ("ecc_corrected", "ecc.errors.corrected.volatile.total", dict(absolute=0.0)),
        )
        for mine, theirs, tol in compare:
            if reading.get(mine) is None or sample.get(theirs) is None:
                continue  # not read by TensorQuill (its `unavailable` list says why) or [N/A] in nvidia-smi
            checks.add("nvml", f"{at}: {mine} against nvidia-smi {theirs}", within(reading[mine], sample[theirs], **tol),
                       f"{reading[mine]} against {sample[theirs]} ({tol})")
        pairs.append({"tensorquill": reading, "nvidia_smi": sample, "dt_s": sample["t"] - reading["read_at"]})
    return {"pairs": pairs, "power_window": check_power(checks, readings, smi)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the capture and system.json")
    cuda = torch.cuda.is_available()
    parser.add_argument("--dim", type=int, default=1024 if cuda else 64)
    parser.add_argument("--blocks", type=int, default=4 if cuda else 2)
    parser.add_argument("--batch", type=int, default=4096 if cuda else 256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--every", type=int, default=50, help="steps between flushes")
    parser.add_argument("--wait-ms", type=float, default=20.0, help="the loader's sleep before each batch")
    parser.add_argument("--wait-slack-ms", type=float, default=15.0,
                        help="how far over the sleep plus the loader's own delay (measured with capture off) data_wait may be")
    parser.add_argument("--off-steps", type=int, default=OFF_STEPS,
                        help="capture-off steps before init that measure the loader's own delay")
    parser.add_argument("--smi-interval", type=float, default=0.25, help="seconds between nvidia-smi samples")
    args = parser.parse_args(argv)
    if args.steps < 4 * args.every:
        parser.error("--steps must give at least four flushes (NVML readings are one flush late)")
    if args.off_steps < MIN_OFF_STEPS:
        parser.error(f"--off-steps must be at least {MIN_OFF_STEPS}, so that each sleep length has two steps")
    device = torch.device("cuda" if cuda else "cpu")
    args.out.mkdir(parents=True)
    name = torch.cuda.get_device_name() if cuda else "CPU"
    print(f"torch {torch.__version__} on {name}", flush=True)

    torch.manual_seed(0)
    model = MoE(args.dim, args.blocks).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    data = SleepyBatches(args.steps, args.batch, args.dim, args.wait_ms / 1000)
    loader = torch.utils.data.DataLoader(data, batch_size=None, sampler=range(args.steps), num_workers=0)
    smi = None
    if cuda and shutil.which("nvidia-smi"):
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        smi = Smi(uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}", args.smi_interval)
    jsonl = args.out / "system.jsonl"
    alongside: dict[int, dict] = {}

    def read_memory(step: int) -> None:
        if cuda and (step + 1) % args.every == 0:
            alongside[step] = {"t": time.time(), "allocated": torch.cuda.memory_allocated(device),
                               "reserved": torch.cuda.memory_reserved(device),
                               "peak": torch.cuda.max_memory_allocated(device)}

    # Capture off: the loader's own delay, before init, whose hooks stay for the rest of the process.
    off_data = SleepyBatches(args.off_steps, args.batch, args.dim, args.wait_ms / 1000)
    off_loader = torch.utils.data.DataLoader(off_data, batch_size=None, sampler=range(args.off_steps), num_workers=0)
    own_off = train_steps(model, optimizer, off_loader, device, args.off_steps)
    with tensorquill.init(model, run_id="system-check", every=args.every, sink=tensorquill.JsonlSink(jsonl),
                          routers=routers(model), system=True, data_wait=True):
        with smi if smi else _nothing():
            own_wait = train_steps(model, optimizer, loader, device, args.steps, after=read_memory)

    records = tensorquill.load_records(jsonl)
    systems = {r["step"]: r["system"] for r in records if r.get("system")}
    checks = Checks()
    result: dict[str, Any] = {"device": name, "torch": torch.__version__, "steps": args.steps, "every": args.every,
                              "system_records_at": sorted(systems)}
    result["data_wait"] = check_data_wait(checks, records, data, args.wait_slack_ms / 1000, own_wait,
                                          own_delay=loader_own_delay(own_off, off_data))
    result["data_wait"]["loader_own_delay"] = own_delay_summary(own_off, off_data)
    result["cuda_memory"] = check_memory(checks, systems, alongside, device, smi)
    result["nvml"] = check_nvml(checks, systems, device, smi)
    if smi is not None:
        result["nvidia_smi"] = {"samples": len(smi.samples), "errors": smi.errors[:3]}
    failed = [c for c in checks.items if not c["ok"]]
    status = "fail" if failed else "not_proven" if checks.not_proven else "pass"
    result.update(status=status, checks=checks.items, not_proven=checks.not_proven)
    (args.out / "system.json").write_text(json.dumps(result, indent=1, default=str))
    for part in ("data_wait", "cuda_memory", "nvml"):
        mine = [c for c in checks.items if c["part"] == part]
        print(f"{part}: {sum(c['ok'] for c in mine)} of {len(mine)} checks hold")
    for c in failed:
        print(f"FAIL [{c['part']}] {c['name']}: {c['detail']}")
    for why in checks.not_proven:
        print(f"NOT PROVEN {why}")
    print(f"{status.upper().replace('_', ' ')}: result {args.out / 'system.json'}")
    return 1 if failed else NOT_PROVEN if checks.not_proven else 0


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return None


if __name__ == "__main__":
    sys.exit(main())
