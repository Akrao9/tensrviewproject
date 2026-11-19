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
- **NVML** (`"nvml"`), asked for by the writer thread and read through `pynvml` (the
  `nvidia-ml-py` package, the `nvml` extra) on a daemon thread of its own: temperature, SM
  clock, power and its limit, the volatile ECC counters, the current clock-event (throttle)
  reasons and the Xid events since the last reading. The device is found by its UUID, so
  `CUDA_VISIBLE_DEVICES` is respected.

A reading that cannot be taken is written once as `{"unavailable": reason}` and then left
out; one NVML metric that fails is listed once under `"unavailable"` and not asked again,
as wandb's system monitor does. Nothing here raises into training: every failure becomes
such a note.

A hung or slow driver call never holds the writer, and so never training: the writer never waits
for NVML (the author's decision on L2). Every NVML call is made on the NVML thread
(`SystemReadings`). Each flush attaches the newest reading that has finished and asks for the next,
so the first flush carries no NVML part and a reading is at most one flush old; `read_at` in it is
the wall-clock time it finished. A reading still in the driver longer than `NVML_SLOW_SECONDS`
(`NVML_OPEN_SECONDS` for the first, which opens NVML) is found at a later flush and written as the
note, and NVML is off for the session. The stuck thread is never joined and never asked again; if
its call ever returns, it shuts NVML down and exits. A reading that comes back in time but took over
`NVML_SLOW_SECONDS` inside NVML is kept, and NVML goes off after it.
"""

from __future__ import annotations

import queue
import threading
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
# How long the first reading, which also opens NVML (nvmlInit, the device, the Xid event set), may
# stay in the driver before a later flush turns NVML off. The writer never waits for it. Opening gets
# more room than a reading rather than turning off an NVML that is only slow to start.
NVML_OPEN_SECONDS = 10.0
XID_PER_READING = 64  # Xid events drained per reading at most
_NVML_ERROR_TIMEOUT = 10  # NVML_ERROR_TIMEOUT: no event waiting


def _reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _cuda_device_type(device: torch.device) -> bool:
    """Whether `device` is a CUDA device, from its type alone: no CUDA call."""
    return device.type == "cuda"


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
    """NVML readings for one CUDA device. Opened, read and closed on the NVML thread only
    (`SystemReadings`)."""

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
        """The reading, `{"unavailable": reason}` the first time NVML cannot be used, then None.
        None too once every metric has failed: the reading that saw them fail listed them."""
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
        # Empty once every metric failed in an earlier reading, which said why: nothing to add.
        return out or None

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


class _Ask:
    """One reading the writer thread asks of the NVML thread. The writer never waits on it: it looks
    at `done` at its next flush (`SystemReadings._take`)."""

    __slots__ = ("done", "reading", "off", "asked", "first")

    def __init__(self, first: bool) -> None:
        self.done = threading.Event()
        self.reading: dict[str, Any] | None = None
        self.off = False  # NVML is off for the session and its note is written: ask no more
        self.asked = time.monotonic()
        self.first = first  # the first reading, which opens NVML


class SystemReadings:
    """One session's readings: CUDA memory at each flush on the training thread, NVML asked for by the
    writer thread and read on the NVML thread, a daemon thread started by the first `write` on a CUDA
    model. Once that thread exists, only it touches `self.nvml`.

    The writer never waits for NVML (the author's decision on L2). Each `write` attaches the newest
    reading that has finished and was not attached yet, then asks for the next one if none is in
    flight. So the first flush carries no NVML part, and a reading rides on the flush after the one
    that asked for it: at most one flush old. `read_at` in the reading is the wall-clock time it
    finished, so a reader can place it. A reading still in the driver longer than
    `NVML_SLOW_SECONDS` (`NVML_OPEN_SECONDS` for the first) is found at a later flush, never waited
    for: that flush writes the note once and NVML is off for the session."""

    def __init__(self, device: torch.device) -> None:
        self.memory = CudaMemory(device)
        self.nvml = Nvml(device)
        self._cuda = _cuda_device_type(device)
        self._asks: queue.SimpleQueue[_Ask | None] = queue.SimpleQueue()  # None: shut down and exit
        self._thread: threading.Thread | None = None
        self._pending: _Ask | None = None  # the reading asked for and not attached yet
        self._off = False  # writer side: no more readings are asked for
        self._stuck = False  # a reading did not come back in time: its thread is never waited for
        self._closed = False

    def flush(self) -> SystemPart:
        """On the training thread, when a block is flushed: the memory part."""
        reading = self.memory.read()
        return SystemPart() if reading is None else SystemPart(cuda=reading)

    def write(self, system: dict[str, Any]) -> dict[str, Any] | None:
        """On the writer thread: add the newest finished NVML part to what `flush` read. Never waits.
        None when empty."""
        reading = self._take()
        system = dict(system)  # a plain dict in the record
        if reading is not None:
            system["nvml"] = reading
        return system or None

    def _take(self) -> dict[str, Any] | None:
        """On the writer thread: the reading that finished since the last flush (or the note), and the
        next one asked for. Nothing here waits."""
        if self._off:
            return None
        if not self._cuda:
            # No NVML call is needed to know there is no reading: said at the first flush, no thread.
            self._off = True
            return self.nvml.read()
        reading = None
        pending = self._pending
        if pending is not None:
            if not pending.done.is_set():
                bound = NVML_OPEN_SECONDS if pending.first else NVML_SLOW_SECONDS
                if time.monotonic() - pending.asked <= bound:
                    return None  # still in the driver, within its bound: nothing new to attach
                # A hung driver call. The thread is left to it: never joined, never asked again,
                # and what it reads, if it ever returns, is dropped with `pending`.
                self._off = self._stuck = True
                self._pending = None
                self._asks.put(None)  # if the call returns: shut NVML down and exit
                if pending.first:
                    return {"unavailable": f"the first NVML reading, which opens NVML, took over {bound:g} s; "
                                           "NVML is off for this session"}
                return {"unavailable": f"an NVML reading took over {bound:g} s; NVML is off for this session"}
            self._pending = None
            reading = pending.reading
            if pending.off:
                self._off = True
                self._asks.put(None)  # nothing more to read: the thread exits
                return reading
        first = self._thread is None
        if first:
            thread = threading.Thread(target=self._serve, name="tensorquill-nvml", daemon=True)
            try:
                thread.start()
            except Exception as error:  # noqa: BLE001
                self._off = True
                return {"unavailable": f"the NVML thread did not start: {_reason(error)}"}
            self._thread = thread
        self._pending = _Ask(first)
        self._asks.put(self._pending)
        return reading

    def _serve(self) -> None:
        """The NVML thread: every NVML call of the session is made here."""
        try:
            while True:
                ask = self._asks.get()
                if ask is None:
                    return
                try:
                    reading = self._read()
                    if reading is not None and set(reading) - {"unavailable"}:
                        reading["read_at"] = time.time()  # when it finished: it is attached a flush later
                    ask.reading = reading
                    ask.off = self.nvml.unavailable is not None and self.nvml._reported
                finally:
                    ask.done.set()
        finally:
            self.nvml.close()

    def _read(self) -> dict[str, Any] | None:
        """On the NVML thread: one reading, the note when it fails."""
        try:
            return self.nvml.read()
        except Exception as error:  # noqa: BLE001
            # A reading that fails in a way the per-metric checks did not catch: NVML goes
            # off for the session rather than the thread failing.
            self.nvml.unavailable = _reason(error)
            self.nvml.close()
            reading = None if self.nvml._reported else {"unavailable": self.nvml.unavailable}
            self.nvml._reported = True
            return reading

    def close(self) -> None:
        """On the writer thread, at the end: the NVML thread shuts NVML down and exits. It is waited for
        at most `NVML_SLOW_SECONDS`, and not at all once a reading was left behind. A reading that
        finished after the last flush is dropped: no record is left to carry it."""
        if self._closed:
            return
        self._closed = True
        if self._thread is None:
            return  # never opened: NVML is only opened on the NVML thread
        if not self._off:
            self._asks.put(None)
        if not self._stuck:
            self._thread.join(NVML_SLOW_SECONDS)
