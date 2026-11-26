"""Detectors over the GPU memory, NVML and data-wait readings (`system.py`, `datawait.py`).

Each kind is reported once, at the first record that meets its rule, and every rule reads
the records forward only, so the server's Rust twins (`server/src/detect/system.rs`) give
the same incidents after any prefix of a rank's records. The readings belong to one
process, so every rule starts again when the attempt changes (a resume).

Per rank, in Python and Rust:

- `memory_growth`: allocated memory rose at each of `MEMORY_GROWTH_READINGS` flush readings
  in a row, by at least `MEMORY_GROWTH_FRACTION` of the first and `MEMORY_GROWTH_BYTES`:
  a leak's trend, such as a loss kept with its graph. A one-off jump that then stays flat
  is not a trend and is not reported.
- `memory_pressure`: reserved memory reached `MEMORY_RESERVED_FRACTION` of the device.
- `allocator_retries`: the caching allocator failed an allocation and had to free its cache
  and retry (which synchronizes the device), or raised an out-of-memory error, since the
  session started.
- `hardware_errors`: the volatile uncorrected ECC count rose above its first reading of the
  attempt, or NVML delivered an Xid event. Hardware evidence, not a verdict on the job.
- `gpu_throttled`: NVML gave a thermal, hardware-slowdown or power-brake clock-event
  reason. An observation. `sw_power_cap`, the GPU running at its configured power limit,
  is what a busy GPU does and is recorded but never reported.
- `input_starvation`: over `INPUT_WINDOW` steps, the median share of the step spent waiting
  for the next batch (`data_wait / step_time`) is at least `INPUT_SHARE`, or has risen by
  `INPUT_RISE` over the first such window: the input pipeline, not the GPU, sets the pace.

Across ranks, in Python only: `memory_imbalance` (`detect_memory_imbalance`), from each
rank's latest memory reading, which the server's per-rank report carries as `memory`.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

MEMORY_GROWTH_READINGS = 6
MEMORY_GROWTH_FRACTION = 0.05
MEMORY_GROWTH_BYTES = 64 * 1024 * 1024
MEMORY_RESERVED_FRACTION = 0.95
THROTTLE_REPORTED = ("sw_thermal", "hw_thermal", "hw_slowdown", "hw_power_brake")
INPUT_WINDOW = 20
INPUT_SHARE = 0.3
INPUT_RISE = 0.2
MEMORY_IMBALANCE_FACTOR = 1.25
MEMORY_IMBALANCE_BYTES = 1024 * 1024 * 1024

SYSTEM_KINDS = (
    "memory_growth", "memory_pressure", "allocator_retries", "hardware_errors", "gpu_throttled", "input_starvation",
)
_MEMORY_KINDS = ["memory_growth", "memory_pressure", "allocator_retries"]
_NVML_KINDS = ["hardware_errors", "gpu_throttled"]
# The NVML metrics a kind reads: when one cannot be read, that kind is not checked.
_METRIC_KINDS = {"ecc_uncorrected": ["hardware_errors"], "xid": ["hardware_errors"], "throttle": ["gpu_throttled"]}


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    """A finite number, else None; a bool is not a number (DET-16), as Rust's `system::number`. An
    integer too large for a float is None too, as `evidence._finite` reads it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _part(record: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    system = record.get("system")
    part = system.get(name) if isinstance(system, dict) else None
    return part if isinstance(part, dict) else None


def _in_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """In step order, which after `resolve_attempts` is also attempt order."""
    return sorted(records, key=lambda r: int(r["step"]))


def detect_system_incidents(records: list[dict[str, Any]], not_checked: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The incidents of `SYSTEM_KINDS` in that order, each the first that meets its rule.
    Readings that could not be taken are appended to `not_checked`, once per check."""
    found: dict[str, dict[str, Any]] = {}
    noted: set[tuple[str, str]] = set()
    attempt: Any = object()
    growth: list[tuple[int, int]] = []
    ecc_start: int | None = None
    shares: list[tuple[float, float, float]] = []
    baseline: float | None = None
    for record in _in_order(records):
        step = int(record["step"])
        if record.get("attempt", 0) != attempt:
            attempt = record.get("attempt", 0)
            growth, ecc_start, shares, baseline = [], None, [], None
        cuda = _part(record, "cuda")
        nvml = _part(record, "nvml")
        if not_checked is not None:
            _note_unavailable(step, cuda, nvml, noted, not_checked)
        if cuda is not None:
            _memory(step, cuda, growth, found)
        if nvml is not None:
            ecc = _count(nvml.get("ecc_uncorrected"))
            if ecc_start is None:
                ecc_start = ecc
            xid = nvml.get("xid")
            xid = [x for x in xid if _count(x) is not None] if isinstance(xid, list) else []
            if "hardware_errors" not in found and (xid or (ecc is not None and ecc_start is not None and ecc > ecc_start)):
                found["hardware_errors"] = {
                    "kind": "hardware_errors", "step": step, "xid": xid,
                    "ecc_uncorrected": ecc, "ecc_uncorrected_at_start": ecc_start,
                }
            reasons = nvml.get("throttle")
            reasons = [r for r in reasons if isinstance(r, str)] if isinstance(reasons, list) else []
            if "gpu_throttled" not in found and any(r in THROTTLE_REPORTED for r in reasons):
                found["gpu_throttled"] = {
                    "kind": "gpu_throttled", "step": step, "reasons": reasons,
                    **{key: _number(nvml.get(key)) for key in ("temperature_c", "sm_clock_mhz", "power_w", "power_limit_w")},
                }
        wait, took = _number(record.get("data_wait")), _number(record.get("step_time"))
        if wait is not None and took is not None and took > 0 and wait >= 0:
            shares.append((wait / took, wait, took))
            if len(shares) > INPUT_WINDOW:
                shares.pop(0)
            if len(shares) == INPUT_WINDOW:
                share = _median([s for s, _, _ in shares])
                if baseline is None:
                    baseline = share
                rule = "share" if share >= INPUT_SHARE else "rise" if share - baseline >= INPUT_RISE else None
                if rule is not None and "input_starvation" not in found:
                    found["input_starvation"] = {
                        "kind": "input_starvation", "step": step, "rule": rule, "median_share": share,
                        "baseline_share": baseline, "median_data_wait": _median([w for _, w, _ in shares]),
                        "median_step_time": _median([t for _, _, t in shares]), "window": INPUT_WINDOW,
                    }
    return [found[kind] for kind in SYSTEM_KINDS if kind in found]


def _memory(step: int, cuda: Mapping[str, Any], growth: list[tuple[int, int]], found: dict[str, dict[str, Any]]) -> None:
    allocated, reserved, total = (_count(cuda.get(key)) for key in ("allocated", "reserved", "total"))
    if allocated is not None:
        if growth and allocated <= growth[-1][1]:
            growth.clear()
        growth.append((step, allocated))
        if len(growth) > MEMORY_GROWTH_READINGS:
            growth.pop(0)
        if "memory_growth" not in found and len(growth) == MEMORY_GROWTH_READINGS:
            (first_step, first), rise = growth[0], allocated - growth[0][1]
            if rise >= MEMORY_GROWTH_BYTES and rise >= MEMORY_GROWTH_FRACTION * first:
                # A rate per step needs the readings to span steps: readings at one repeated step
                # (itself `repeated_steps`) are growth with no rate (lane U7, Nov 19, 2026: this
                # divided by zero and took the whole report down).
                per_step = rise / (step - first_step) if step > first_step else None
                found["memory_growth"] = {
                    "kind": "memory_growth", "step": step, "first_step": first_step, "allocated_first": first,
                    "allocated": allocated, "readings": MEMORY_GROWTH_READINGS, "bytes_per_step": per_step,
                    "total": total,
                    "steps_to_total": max(total - allocated, 0) / per_step if total is not None and per_step else None,
                }
    if "memory_pressure" not in found and reserved is not None and total is not None and total > 0:
        if reserved >= MEMORY_RESERVED_FRACTION * total:
            found["memory_pressure"] = {
                "kind": "memory_pressure", "step": step, "reserved": reserved, "total": total,
                "fraction": reserved / total, "allocated": allocated,
            }
    retries, ooms = _count(cuda.get("alloc_retries")), _count(cuda.get("ooms"))
    if "allocator_retries" not in found and ((retries or 0) > 0 or (ooms or 0) > 0):
        found["allocator_retries"] = {"kind": "allocator_retries", "step": step, "alloc_retries": retries, "ooms": ooms}


def _note_unavailable(
    step: int, cuda: Mapping[str, Any] | None, nvml: Mapping[str, Any] | None,
    noted: set[tuple[str, str]], not_checked: list[dict[str, Any]],
) -> None:
    def note(check: str, metric: str, kinds: list[str], reason: Any) -> None:
        if (check, metric) in noted:
            return
        noted.add((check, metric))
        entry: dict[str, Any] = {"check": check, "kinds": kinds, "reason": str(reason), "step": step}
        if metric:
            entry["metric"] = metric
        not_checked.append(entry)

    if cuda is not None and "unavailable" in cuda:
        note("cuda_memory", "", _MEMORY_KINDS, cuda["unavailable"])
    if nvml is not None and "unavailable" in nvml:
        unavailable = nvml["unavailable"]
        if isinstance(unavailable, dict):
            for metric, reason in unavailable.items():
                if metric in _METRIC_KINDS:
                    note("nvml", metric, _METRIC_KINDS[metric], reason)
        else:
            note("nvml", "", _NVML_KINDS, unavailable)


def latest_memory(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The last memory reading, as the server's report carries it (`memory`): what
    `detect_memory_imbalance` compares across ranks."""
    latest = None
    for record in _in_order(records):
        cuda = _part(record, "cuda")
        if cuda is not None and _count(cuda.get("allocated")) is not None:
            latest = {"step": int(record["step"]), **{
                key: _count(cuda.get(key)) for key in ("allocated", "reserved", "peak_allocated", "total")
            }}
    return latest


def detect_memory_imbalance(latest: Mapping[int, Mapping[str, Any] | None]) -> dict[str, Any] | None:
    """Ranks whose peak allocated memory is at least `MEMORY_IMBALANCE_FACTOR` times the
    median across ranks and `MEMORY_IMBALANCE_BYTES` above it, from each rank's latest
    reading (`latest_memory`, or `memory` in the server's per-rank report). The rank that
    runs out of memory first is the one with the highest peak. Expected under pipeline
    parallelism or uneven sharding, so an observation. Python only: the server's detectors
    see one rank at a time."""
    readings = {int(rank): reading for rank, reading in latest.items() if isinstance(reading, Mapping)}
    peaks = {rank: peak for rank, reading in readings.items() if (peak := _count(reading.get("peak_allocated"))) is not None}
    if len(peaks) < 2:
        return None
    median = _median([float(p) for p in peaks.values()])
    high = sorted(r for r, p in peaks.items() if p >= MEMORY_IMBALANCE_FACTOR * median and p - median >= MEMORY_IMBALANCE_BYTES)
    if not high:
        return None
    steps = [s for r in peaks if isinstance(s := readings[r].get("step"), int)]
    return {
        "kind": "memory_imbalance", "step": max(steps) if steps else None, "ranks": high,
        "median_peak_allocated": median, "peak_allocated": {str(r): peaks[r] for r in sorted(peaks)},
    }


def _gib(value: Any) -> str:
    return f"{value / 2**30:.2f} GiB" if isinstance(value, (int, float)) else "unknown"


def describe_system(incident: Mapping[str, Any]) -> str | None:
    """One sentence for an incident of `SYSTEM_KINDS` or `memory_imbalance`, or None."""
    kind, step = incident["kind"], incident.get("step")
    if kind == "memory_growth":
        rate, left = incident.get("bytes_per_step"), incident.get("steps_to_total")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            # No rate: every reading at one repeated step (the Rust detectors write a null rate there).
            rate = left = None
        return (
            f"GPU memory kept growing up to step {step}: allocated rose at each of {incident['readings']} readings"
            f" from step {incident['first_step']}, {_gib(incident['allocated_first'])} to {_gib(incident['allocated'])}"
            + (f" (about {rate / 2**20:.1f} MiB per step)" if rate is not None
               else f" (every reading at step {step}, so no rate per step)")
            + ", as when tensors are kept across steps"
            + (f"; at this rate the device is full in about {left:.0f} steps" if isinstance(left, (int, float)) else "")
        )
    if kind == "memory_pressure":
        return (
            f"GPU memory near the device's total at step {step}: {_gib(incident['reserved'])} reserved of"
            f" {_gib(incident['total'])} ({incident['fraction']:.0%}); a larger batch or sequence may not fit"
        )
    if kind == "allocator_retries":
        return (
            f"The CUDA allocator ran short by step {step}: {incident.get('alloc_retries') or 0} allocation retries"
            f" (each frees the cache and synchronizes) and {incident.get('ooms') or 0} out-of-memory errors since"
            " the session started"
        )
    if kind == "hardware_errors":
        parts = []
        if incident.get("xid"):
            parts.append(f"Xid {', '.join(str(x) for x in incident['xid'])}")
        start, now = incident.get("ecc_uncorrected_at_start"), incident.get("ecc_uncorrected")
        if isinstance(start, int) and isinstance(now, int) and now > start:
            parts.append(f"uncorrected ECC errors {start} → {now}")
        return f"GPU hardware errors at step {step}: {'; '.join(parts)}. Hardware evidence: check the node before blaming the job"
    if kind == "gpu_throttled":
        temperature = incident.get("temperature_c")
        return (
            f"GPU clocks throttled at step {step} ({', '.join(incident['reasons'])})"
            + (f" at {temperature:.0f} °C" if isinstance(temperature, (int, float)) else "")
            + ". An observation: steps may be slower than the hardware allows"
        )
    if kind == "input_starvation":
        why = (
            f"the median step spent {incident['median_share']:.0%} of its time waiting for data"
            if incident["rule"] == "share"
            else f"the share of each step spent waiting for data rose from {incident['baseline_share']:.0%}"
            f" to {incident['median_share']:.0%}"
        )
        return (
            f"Input pipeline starving the GPU by step {step}: over {incident['window']} steps {why}"
            f" ({incident['median_data_wait'] * 1000:.1f} ms of a {incident['median_step_time'] * 1000:.1f} ms step)."
            " The data loader, not the GPU, sets the pace: more workers, prefetching or faster storage"
        )
    if kind == "memory_imbalance":
        ranks = ", ".join(str(r) for r in incident["ranks"])
        return (
            f"GPU memory uneven across ranks: peak allocated on rank{'s' if len(incident['ranks']) > 1 else ''} {ranks}"
            f" is well above the median of {_gib(incident['median_peak_allocated'])}; those ranks run out of memory"
            " first. Expected under pipeline parallelism or uneven sharding"
        )
    return None
