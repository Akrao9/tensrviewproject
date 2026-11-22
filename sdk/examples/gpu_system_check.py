"""GPU memory, NVML and data-wait readings, checked against readings taken alongside in the same run.

    python examples/gpu_system_check.py --out runs/system                 # CUDA, nvidia-ml-py, nvidia-smi
    python examples/gpu_system_check.py --out runs/system-cpu             # CPU: data wait only, exit 3

Trains the small MoE of `gpu_check.py` from a `torch.utils.data.DataLoader` with
`tensorquill.init(..., system=True, data_wait=True)`, flushing every `--every` steps, and checks
what TensorQuill recorded against what the script reads itself:

- **Data wait.** The loader sleeps a known time before each batch: `--wait-ms` on every step, four
  times that on steps 5, 15, 25, ... TensorQuill's `data_wait` on every step must be at least the
  sleep; at most the sleep plus `--wait-slack-ms` on 98% of steps, and the sleep plus
  `OUTLIER_MS` (100 ms) on every step, the steps over the slack listed (host jitter: on this Mac
  one step in 200 to 300 came in 40 to 70 ms late, at no fixed flush); `step_time` at least
  `data_wait`. This part is a real check on any machine.
- **CUDA memory** (the record's `system.cuda`, at each flush). Right after `optimizer.step()` on
  that step the script reads `torch.cuda.memory_allocated`, `memory_reserved` and
  `max_memory_allocated`. These are the same allocator's counters, so they check that TensorQuill
  read them and when, within 2% or 32 MiB. The independent ones: `total` against the device's
  properties (exactly) and `nvidia-smi`'s `memory.total` (within 3%: the driver keeps some back), and
  `nvidia-smi`'s `memory.used`, which must be at least TensorQuill's `reserved` less 2%.
- **NVML** (`system.nvml`, the `nvml` extra: `pip install nvidia-ml-py`). A thread runs `nvidia-smi`
  for this GPU, chosen by UUID, every `--smi-interval` seconds during training. Each NVML reading
  is paired with the `nvidia-smi` sample nearest its `read_at` (within 1 s): temperature within 3 C,
  SM clock within 15%, power within 25% or 25 W, the enforced power limit within 1 W, the volatile
  ECC counts equal where `nvidia-smi` reports them. NVML readings are one flush late and the last
  is dropped, so `--steps` must give at least four flushes.

The result is `system.json`, with every pairing. Exit status: 1 if any check fails; else 3 (not
proven) when a part could not be checked here: no CUDA, `nvidia-ml-py` absent, or no `nvidia-smi`;
else 0. On a CPU the script says NVML and CUDA memory are absent, checks that TensorQuill said so
too, checks the data wait, and exits 3.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
WITHIN_SLACK = 0.98  # the share of steps whose data_wait must be within the slack


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


def check_data_wait(checks: Checks, records: list[dict], data: SleepyBatches, slack_s: float) -> dict:
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
        elif got > want + slack_s:
            high.append((step, got, want))
        if isinstance(r.get("step_time"), (int, float)) and r["step_time"] < got:
            short.append(step)
    checks.add("data_wait", "every step has a data_wait", not missing, f"{len(missing)} without, first {missing[:3]}")
    checks.add("data_wait", "data_wait is at least the injected sleep", not low,
               f"{len(low)} below" + (f", first step {low[0][0]}: {low[0][1] * 1e3:.2f} ms < {low[0][2] * 1e3:.2f} ms" if low else ""))
    late = [(step, got, want) for step, got, want in high if got > want + OUTLIER_MS / 1000]
    listed = ", ".join(f"step {st}: {g * 1e3:.1f} ms for {w * 1e3:.0f} ms" for st, g, w in high[:5])
    checks.add("data_wait", f"data_wait is at most the sleep plus {slack_s * 1e3:g} ms on {100 * WITHIN_SLACK:g}% of steps",
               len(high) <= (1 - WITHIN_SLACK) * data.steps, f"{len(high)} of {data.steps} above" + (f": {listed}" if high else ""))
    checks.add("data_wait", f"data_wait is at most the sleep plus {OUTLIER_MS:g} ms on every step", not late,
               f"{len(late)} above" + (f", first step {late[0][0]}: {late[0][1] * 1e3:.1f} ms" if late else ""))
    checks.add("data_wait", "step_time is at least data_wait", not short, f"{len(short)} steps shorter, first {short[:3]}")
    long_steps = [s for s in range(data.steps) if s % 10 == 5 and s in by_step]
    excess = sorted((r["data_wait"] - data.delay(s)) * 1e3 for s, r in by_step.items() if isinstance(r.get("data_wait"), (int, float)))
    return {"steps": data.steps, "long_wait_steps": len(long_steps),
            "excess_ms_median": excess[len(excess) // 2] if excess else None, "excess_ms_max": excess[-1] if excess else None,
            "over_slack": [{"step": st, "data_wait_ms": g * 1e3, "injected_ms": w * 1e3} for st, g, w in high],
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
    return pairs


def check_nvml(checks: Checks, systems: dict[int, dict], device, smi: Smi | None) -> list:
    notes = [s["nvml"] for s in systems.values() if "nvml" in s]
    unavailable = [n["unavailable"] for n in notes if isinstance(n.get("unavailable"), str)]
    if device.type != "cuda":
        checks.add("nvml", "TensorQuill says NVML is unavailable off CUDA", bool(unavailable), str(unavailable[:1]))
        checks.unproven("nvml", "NVML is absent: no CUDA device here")
        return []
    if unavailable:
        # Absent here (no nvidia-ml-py, no driver): not a failure of TensorQuill's, and nothing proven.
        checks.unproven("nvml", f"TensorQuill: {unavailable[0]}")
        return []
    readings = [n for n in notes if "read_at" in n]
    checks.add("nvml", "NVML readings on at least three flushes", len(readings) >= 3, f"{len(readings)} readings")
    if smi is None:
        checks.unproven("nvml", "nvidia-smi is not on PATH: nothing to compare NVML with")
        return []
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
            ("power_w", "power.draw", dict(rel=0.25, absolute=25.0)),
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
    return pairs


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
    parser.add_argument("--wait-slack-ms", type=float, default=15.0, help="how far over the sleep data_wait may be")
    parser.add_argument("--smi-interval", type=float, default=0.25, help="seconds between nvidia-smi samples")
    args = parser.parse_args(argv)
    if args.steps < 4 * args.every:
        parser.error("--steps must give at least four flushes (NVML readings are one flush late)")
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
    with tensorquill.init(model, run_id="system-check", every=args.every, sink=tensorquill.JsonlSink(jsonl),
                          routers=routers(model), system=True, data_wait=True):
        with smi if smi else _nothing():
            for step, x in enumerate(loader):
                x = x.to(device, non_blocking=True)
                loss = (model(x) - x.sum(dim=1, keepdim=True)).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cuda and (step + 1) % args.every == 0:
                    alongside[step] = {"t": time.time(), "allocated": torch.cuda.memory_allocated(device),
                                       "reserved": torch.cuda.memory_reserved(device),
                                       "peak": torch.cuda.max_memory_allocated(device)}

    records = tensorquill.load_records(jsonl)
    systems = {r["step"]: r["system"] for r in records if r.get("system")}
    checks = Checks()
    result: dict[str, Any] = {"device": name, "torch": torch.__version__, "steps": args.steps, "every": args.every,
                              "system_records_at": sorted(systems)}
    result["data_wait"] = check_data_wait(checks, records, data, args.wait_slack_ms / 1000)
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
