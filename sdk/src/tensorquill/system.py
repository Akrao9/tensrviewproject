"""GPU memory and NVML readings, taken once per flush and never on the training path's GPU.

`init(..., system=True)` turns them on. Each flush (every `every` steps) adds one `system`
object to the last record of the block:

- **CUDA memory** (`"cuda"`), read on the training thread when the block is flushed: the
  caching allocator's own counters (`torch.cuda.memory_stats_as_nested_dict`), which are
  host-side numbers kept by PyTorch, so reading them neither launches work on the GPU nor
  waits for it. `allocated`, `reserved` and `peak_allocated` are bytes; `peak_allocated` is
  what `torch.cuda.max_memory_allocated` returns, and is never reset here, since the peak
  belongs to the user. `alloc_retries` and `ooms` count since the session started (the
  allocator's counters are per process, and a batch-size finder may cause OOMs on purpose
  before training). `total` is the device's memory from its properties, read once.
- **NVML** (`"nvml"`), read on the writer thread through `pynvml` (the `nvidia-ml-py`
  package, the `nvml` extra): temperature, SM clock, power and its limit, the volatile ECC
  counters, the current clock-event (throttle) reasons and the Xid events since the last
  reading. The device is found by its UUID, so `CUDA_VISIBLE_DEVICES` is respected.

A reading that cannot be taken is written once as `{"unavailable": reason}` and then left
out; one NVML metric that fails is listed once under `"unavailable"` and not asked again,
as wandb's system monitor does. Nothing here raises into training: every failure becomes
such a note. An NVML call slower than `NVML_SLOW_SECONDS` turns NVML off for the session,
because the writer thread's queue is bounded and a hung driver call would in the end make
training wait for it.
"""

from __future__ import annotations

import time
from typing import Any

import torch

# Clock-event reasons by bit (NVML's nvmlClocksEventReason*, formerly ClocksThrottleReason*).
# GPU idle is left out: an idle GPU lowering its clocks is not throttling.
THROTTLE_REASONS = {
    0x2: "applications_clocks",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clocks",
}
_IDLE = 0x1
NVML_SLOW_SECONDS = 2.0  # one NVML reading slower than this turns NVML off for the session
XID_PER_READING = 64  # Xid events drained per reading at most
_NVML_ERROR_TIMEOUT = 10  # NVML_ERROR_TIMEOUT: no event waiting


def _reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _cuda_device(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


class CudaMemory:
    """The caching allocator's counters for one device. Read on the training thread."""

    def __init__(self, device: torch.device) -> None:
        self.unavailable: str | None = None
        self._reported = False
        self.index: int | None = None
        self.total: int | None = None
        self._base_retries = 0
        self._base_ooms = 0
        try:
            index = _cuda_device(device)
            if index is None:
                self.unavailable = f"the model is on {device.type}, not a CUDA device"
                return
            self.index = index
            self.total = int(torch.cuda.get_device_properties(index).total_memory)
            stats = self._stats()
            if stats:
                self._base_retries = int(stats.get("num_alloc_retries", 0))
                self._base_ooms = int(stats.get("num_ooms", 0))
        except Exception as error:  # noqa: BLE001
            self.unavailable = _reason(error)

    def _stats(self) -> dict[str, Any]:
        # Returns {} before CUDA is initialised instead of initialising it.
        return torch.cuda.memory_stats_as_nested_dict(self.index)

    def read(self) -> dict[str, Any] | None:
        """The reading, `{"unavailable": reason}` the first time it cannot be taken, then None."""
        if self.unavailable is None:
            try:
                stats = self._stats()
                if not stats:
                    return None  # CUDA not initialised yet: nothing allocated to read
                return {
                    "device": self.index,
                    "allocated": int(stats["allocated_bytes"]["all"]["current"]),
                    "reserved": int(stats["reserved_bytes"]["all"]["current"]),
                    "peak_allocated": int(stats["allocated_bytes"]["all"]["peak"]),
                    "total": self.total,
                    "alloc_retries": int(stats.get("num_alloc_retries", 0)) - self._base_retries,
                    "ooms": int(stats.get("num_ooms", 0)) - self._base_ooms,
                }
            except Exception as error:  # noqa: BLE001
                self.unavailable = _reason(error)
        if self._reported:
            return None
        self._reported = True
        return {"unavailable": self.unavailable}


def _import_pynvml() -> Any:
    import pynvml  # the nvidia-ml-py package

    return pynvml


class Nvml:
    """NVML readings for one CUDA device. Opened and read on the writer thread only."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._nvml: Any = None
        self._handle: Any = None
        self._events: Any = None
        self._opened = False
        self.unavailable: str | None = None
        self._reported = False
        self._off: set[str] = set()  # metrics that failed once and are not asked again
        self._failed: dict[str, str] = {}  # metrics that failed in this reading

    def _open(self) -> None:
        self._opened = True
        index = _cuda_device(self._device)
        if index is None:
            self.unavailable = f"the model is on {self._device.type}, not a CUDA device"
            return
        try:
            nvml = _import_pynvml()
        except Exception as error:  # noqa: BLE001
            self.unavailable = f"pynvml is not installed (pip install nvidia-ml-py): {_reason(error)}"
            return
        try:
            nvml.nvmlInit()
        except Exception as error:  # noqa: BLE001
            self.unavailable = f"nvmlInit failed: {_reason(error)}"
            return
        self._nvml = nvml
        try:
            self._handle = self._find_handle(nvml, index)
        except Exception as error:  # noqa: BLE001
            self.unavailable = f"no NVML device for cuda:{index}: {_reason(error)}"
            self.close()
            return
        try:
            events = nvml.nvmlEventSetCreate()
            try:
                nvml.nvmlDeviceRegisterEvents(self._handle, nvml.nvmlEventTypeXidCriticalError, events)
            except Exception:
                nvml.nvmlEventSetFree(events)
                raise
            self._events = events
        except Exception as error:  # noqa: BLE001
            self._fail("xid", error)

    @staticmethod
    def _find_handle(nvml: Any, index: int) -> Any:
        uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
        if uuid is not None:
            text = str(uuid)
            if not text.startswith(("GPU-", "MIG-")):
                text = f"GPU-{text}"
            try:
                return nvml.nvmlDeviceGetHandleByUUID(text)
            except Exception:  # noqa: BLE001
                pass
        nvml_index = torch.cuda._get_nvml_device_index(index)  # honours CUDA_VISIBLE_DEVICES
        return nvml.nvmlDeviceGetHandleByIndex(nvml_index)

    def _fail(self, metric: str, error: BaseException) -> None:
        self._off.add(metric)
        self._failed[metric] = _reason(error)

    def _metric(self, out: dict[str, Any], metric: str, read: Any) -> None:
        if metric in self._off:
            return
        try:
            out[metric] = read()
        except Exception as error:  # noqa: BLE001
            self._fail(metric, error)

    def read(self) -> dict[str, Any] | None:
        """The reading, `{"unavailable": reason}` the first time NVML cannot be used, then None."""
        self._failed: dict[str, str] = {}
        if not self._opened:
            self._open()
        if self.unavailable is not None:
            if self._reported:
                return None
            self._reported = True
            return {"unavailable": self.unavailable}
        nvml, handle = self._nvml, self._handle
        started = time.monotonic()
        out: dict[str, Any] = {}
        self._metric(out, "temperature_c", lambda: int(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)))
        self._metric(out, "sm_clock_mhz", lambda: int(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)))
        self._metric(out, "power_w", lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        self._metric(out, "power_limit_w", lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
        # Volatile counters, as wandb reads them: they count since the driver last loaded, while
        # the aggregate ones stay non-zero for the board's lifetime after rows are remapped.
        self._metric(out, "ecc_uncorrected", lambda: int(nvml.nvmlDeviceGetTotalEccErrors(
            handle, nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nvml.NVML_VOLATILE_ECC)))
        self._metric(out, "ecc_corrected", lambda: int(nvml.nvmlDeviceGetTotalEccErrors(
            handle, nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, nvml.NVML_VOLATILE_ECC)))
        self._metric(out, "throttle", lambda: _throttle_names(self._reasons(nvml, handle)))
        if self._events is not None and "xid" not in self._off:
            self._metric(out, "xid", self._drain_xid)
        if self._failed:
            out["unavailable"] = self._failed
        if time.monotonic() - started > NVML_SLOW_SECONDS:
            self.unavailable = f"an NVML reading took over {NVML_SLOW_SECONDS:g} s; NVML is off for this session"
            self.close()
        return out

    @staticmethod
    def _reasons(nvml: Any, handle: Any) -> int:
        reasons = getattr(nvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
        if reasons is None:  # nvidia-ml-py before 12.535
            reasons = nvml.nvmlDeviceGetCurrentClocksThrottleReasons
        return int(reasons(handle))

    def _drain_xid(self) -> list[int]:
        nvml, found = self._nvml, []
        for _ in range(XID_PER_READING):
            try:
                event = nvml.nvmlEventSetWait_v2(self._events, 0)
            except nvml.NVMLError as error:
                if getattr(error, "value", None) == _NVML_ERROR_TIMEOUT:
                    break  # no more events waiting
                raise
            found.append(int(event.eventData))
        return found

    def close(self) -> None:
        nvml, self._nvml = self._nvml, None
        if nvml is None:
            return
        try:
            if self._events is not None:
                nvml.nvmlEventSetFree(self._events)
        except Exception:  # noqa: BLE001
            pass
        self._events = None
        try:
            nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _throttle_names(mask: int) -> list[str]:
    mask &= ~_IDLE
    names = [name for bit, name in THROTTLE_REASONS.items() if mask & bit]
    rest = mask & ~sum(THROTTLE_REASONS)
    if rest:
        names.append(hex(rest))
    return names


class SystemPart(dict):
    """The memory part read at a flush, waiting in the writer queue for its NVML part."""


class SystemReadings:
    """One session's readings: CUDA memory at each flush, NVML on the writer thread."""

    def __init__(self, device: torch.device) -> None:
        self.memory = CudaMemory(device)
        self.nvml = Nvml(device)

    def flush(self) -> SystemPart:
        """On the training thread, when a block is flushed: the memory part."""
        reading = self.memory.read()
        return SystemPart() if reading is None else SystemPart(cuda=reading)

    def write(self, system: dict[str, Any]) -> dict[str, Any] | None:
        """On the writer thread: add the NVML part to what `flush` read. None when empty."""
        try:
            reading = self.nvml.read()
        except Exception as error:  # noqa: BLE001
            # A reading that fails in a way the per-metric checks did not catch: NVML goes
            # off for the session rather than the writer thread failing, which would stop training.
            self.nvml.unavailable = _reason(error)
            self.nvml.close()
            reading = None if self.nvml._reported else {"unavailable": self.nvml.unavailable}
            self.nvml._reported = True
        system = dict(system)  # a plain dict in the record
        if reading is not None:
            system["nvml"] = reading
        return system or None

    def close(self) -> None:
        self.nvml.close()
