"""GPU memory and NVML readings (`system.py`) with CUDA and NVML mocked: available,
unavailable and failing, and never a synchronize or an error raised into training."""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import system as system_module
from tensorquill.system import CudaMemory, Nvml, SystemReadings

GIB = 1 << 30
CUDA0 = torch.device("cuda", 0)


def _stats(allocated=2 * GIB, reserved=3 * GIB, peak=4 * GIB, retries=0, ooms=0):
    return {
        "allocated_bytes": {"all": {"current": allocated, "peak": peak}},
        "reserved_bytes": {"all": {"current": reserved}},
        "num_alloc_retries": retries,
        "num_ooms": ooms,
    }


@pytest.fixture
def cuda(monkeypatch):
    """A CUDA device 0 of 16 GiB whose allocator counters the test sets, and a synchronize
    that fails the test if anything calls it."""
    state = {"stats": _stats(retries=5, ooms=1), "calls": []}

    def stats(index):
        state["calls"].append(threading.get_ident())
        if isinstance(state["stats"], BaseException):
            raise state["stats"]
        return state["stats"]

    def no_sync(*args, **kwargs):
        raise AssertionError("torch.cuda.synchronize was called")

    monkeypatch.setattr(torch.cuda, "memory_stats_as_nested_dict", stats)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda index: types.SimpleNamespace(total_memory=16 * GIB, uuid="1234-abcd"),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", no_sync)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return state


class FakeNvml(types.ModuleType):
    """The parts of `pynvml` that `system.py` calls."""

    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_MEMORY_ERROR_TYPE_CORRECTED = 0
    NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
    NVML_VOLATILE_ECC = 0
    nvmlEventTypeXidCriticalError = 8

    class NVMLError(Exception):
        def __init__(self, value):
            super().__init__(f"NVML error {value}")
            self.value = value

    def __init__(self):
        super().__init__("pynvml")
        self.ecc = [0, 7]  # uncorrected, corrected
        self.reasons = 0x4 | 0x1  # power cap and idle
        self.xid: list[int] = []
        self.fail: dict[str, BaseException] = {}
        self.uuids: list[str] = []
        self.shutdowns = 0
        self.threads: set[int] = set()
        self.delay = 0.0

    def _call(self, name):
        self.threads.add(threading.get_ident())
        if name in self.fail:
            raise self.fail[name]

    def nvmlInit(self):
        self._call("init")

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self._call("uuid")
        self.uuids.append(uuid)
        return "handle"

    def nvmlDeviceGetHandleByIndex(self, index):
        return "handle"

    def nvmlDeviceGetTemperature(self, handle, sensor):
        self._call("temperature")
        time.sleep(self.delay)
        return 71

    def nvmlDeviceGetClockInfo(self, handle, clock):
        self._call("clock")
        return 1410

    def nvmlDeviceGetPowerUsage(self, handle):
        self._call("power")
        return 250_000

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        return 400_000

    def nvmlDeviceGetTotalEccErrors(self, handle, kind, counter):
        self._call("ecc")
        return self.ecc[0] if kind == self.NVML_MEMORY_ERROR_TYPE_UNCORRECTED else self.ecc[1]

    def nvmlDeviceGetCurrentClocksEventReasons(self, handle):
        self._call("throttle")
        return self.reasons

    def nvmlEventSetCreate(self):
        self._call("events")
        return "events"

    def nvmlDeviceRegisterEvents(self, handle, kinds, events):
        self._call("register")

    def nvmlEventSetFree(self, events):
        pass

    def nvmlEventSetWait_v2(self, events, timeout):
        self._call("wait")
        if not self.xid:
            raise self.NVMLError(10)  # NVML_ERROR_TIMEOUT
        return types.SimpleNamespace(eventData=self.xid.pop(0))


@pytest.fixture
def nvml(monkeypatch):
    fake = FakeNvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    return fake


# --- CUDA memory ---------------------------------------------------------------------------


def test_memory_reads_the_allocator_counters_since_the_session_started(cuda):
    memory = CudaMemory(CUDA0)  # 5 retries and 1 OOM before the session: not counted
    cuda["stats"] = _stats(allocated=2 * GIB, reserved=3 * GIB, peak=4 * GIB, retries=6, ooms=1)
    assert memory.read() == {
        "device": 0, "allocated": 2 * GIB, "reserved": 3 * GIB, "peak_allocated": 4 * GIB, "total": 16 * GIB,
        "alloc_retries": 1, "ooms": 0,
    }


def test_memory_before_cuda_is_initialised_is_no_reading(cuda):
    memory = CudaMemory(CUDA0)
    cuda["stats"] = {}  # what memory_stats_as_nested_dict returns before CUDA starts
    assert memory.read() is None


def test_memory_on_cpu_is_unavailable_once(cuda):
    memory = CudaMemory(torch.device("cpu"))
    assert memory.read() == {"unavailable": "the model is on cpu, not a CUDA device"}
    assert memory.read() is None and memory.read() is None
    assert cuda["calls"] == []


def test_a_failing_memory_read_is_unavailable_once_and_never_raises(cuda):
    memory = CudaMemory(CUDA0)
    cuda["stats"] = RuntimeError("allocator backend has no stats")
    assert memory.read() == {"unavailable": "RuntimeError: allocator backend has no stats"}
    cuda["stats"] = _stats()
    assert memory.read() is None  # it stays off: one note, not a reading that comes and goes


def test_memory_with_a_failing_device_query_at_init_is_unavailable(cuda, monkeypatch):
    def broken(index):
        raise RuntimeError("no CUDA GPUs are available")

    monkeypatch.setattr(torch.cuda, "get_device_properties", broken)
    assert CudaMemory(CUDA0).read() == {"unavailable": "RuntimeError: no CUDA GPUs are available"}


# --- NVML ----------------------------------------------------------------------------------


def test_nvml_reads_every_metric_and_finds_the_device_by_uuid(cuda, nvml):
    reader = Nvml(CUDA0)
    nvml.xid = [79, 48]
    reading = reader.read()
    assert reading == {
        "temperature_c": 71, "sm_clock_mhz": 1410, "power_w": 250.0, "power_limit_w": 400.0,
        "ecc_uncorrected": 0, "ecc_corrected": 7, "throttle": ["sw_power_cap"], "xid": [79, 48],
    }
    assert nvml.uuids == ["GPU-1234-abcd"]
    nvml.reasons = 0x40 | 0x8 | 0x400
    assert reader.read()["throttle"] == ["hw_slowdown", "hw_thermal", "0x400"]
    assert reader.read()["xid"] == []
    reader.close()
    assert nvml.shutdowns == 1


def test_nvml_without_the_library_is_unavailable_once(cuda, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import pynvml raises ImportError
    reader = Nvml(CUDA0)
    first = reader.read()
    assert set(first) == {"unavailable"} and first["unavailable"].startswith("pynvml is not installed")
    assert reader.read() is None


def test_nvml_without_a_gpu_is_unavailable_once(cuda, nvml):
    nvml.fail["init"] = FakeNvml.NVMLError(9)  # NVML_ERROR_DRIVER_NOT_LOADED
    reader = Nvml(CUDA0)
    assert reader.read() == {"unavailable": "nvmlInit failed: NVMLError: NVML error 9"}
    assert reader.read() is None


def test_nvml_on_cpu_never_imports_the_library(nvml):
    reader = Nvml(torch.device("cpu"))
    assert reader.read() == {"unavailable": "the model is on cpu, not a CUDA device"}
    assert nvml.threads == set()


def test_one_failing_metric_is_listed_once_and_not_asked_again(cuda, nvml):
    nvml.fail["ecc"] = FakeNvml.NVMLError(3)  # NVML_ERROR_NOT_SUPPORTED, as on consumer boards
    nvml.fail["register"] = FakeNvml.NVMLError(3)
    reader = Nvml(CUDA0)
    first = reader.read()
    assert first["unavailable"] == {
        "xid": "NVMLError: NVML error 3", "ecc_uncorrected": "NVMLError: NVML error 3",
        "ecc_corrected": "NVMLError: NVML error 3",
    }
    assert "ecc_uncorrected" not in first and first["temperature_c"] == 71
    second = reader.read()
    assert "unavailable" not in second and "ecc_uncorrected" not in second and "xid" not in second


def test_a_slow_nvml_turns_nvml_off(cuda, nvml, monkeypatch):
    monkeypatch.setattr(system_module, "NVML_SLOW_SECONDS", 0.01)
    nvml.delay = 0.05
    reader = Nvml(CUDA0)
    assert reader.read()["temperature_c"] == 71  # the slow reading is kept
    assert reader.read() == {"unavailable": "an NVML reading took over 0.01 s; NVML is off for this session"}
    assert reader.read() is None and nvml.shutdowns == 1


def test_an_unexpected_error_inside_a_reading_turns_nvml_off_and_never_raises(cuda, nvml, monkeypatch):
    readings = SystemReadings(CUDA0)
    assert "nvml" in readings.write({})
    monkeypatch.setattr(Nvml, "_drain_xid", lambda self: (_ for _ in ()).throw(KeyError("odd")))
    monkeypatch.setattr(Nvml, "_metric", lambda self, out, metric, read: (_ for _ in ()).throw(ValueError("bad")))
    assert readings.write({"cuda": {"allocated": 1}}) == {"cuda": {"allocated": 1}, "nvml": {"unavailable": "ValueError: bad"}}
    assert readings.write({}) is None


# --- through a session ---------------------------------------------------------------------


def _train(session, model, steps):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()


def _written(sink, count):
    """Wait for the writer thread, which reads NVML, to have written `count` records."""
    deadline = time.monotonic() + 10
    while len(sink.records) < count:
        assert time.monotonic() < deadline, "the writer thread did not write the records"
        time.sleep(0.005)


def _session(every=5, **kwargs):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=every, run_id="sys", **kwargs)
    return session, model, sink


def test_off_by_default_and_on_a_cpu_model_unavailable_once():
    session, model, sink = _session()
    _train(session, model, 10)
    session.close()
    assert not any("system" in r for r in sink.records)
    session, model, sink = _session(system=True)
    _train(session, model, 15)
    session.close()
    systems = [(r["step"], r["system"]) for r in sink.records if "system" in r]
    assert systems == [(4, {
        "cuda": {"unavailable": "the model is on cpu, not a CUDA device"},
        "nvml": {"unavailable": "the model is on cpu, not a CUDA device"},
    })]


def test_each_flush_adds_one_reading_memory_on_the_training_thread_nvml_on_the_writer(cuda, nvml):
    session, model, sink = _session(system=True)
    session._system = SystemReadings(CUDA0)  # as on a CUDA model
    trainer_thread = threading.get_ident()
    cuda["calls"].clear()
    _train(session, model, 6)  # a full ring's newest step is written when the next step ends
    _written(sink, 5)
    nvml.ecc[0] = 2
    nvml.xid = [79]
    _train(session, model, 5)
    session.close()
    systems = [(r["step"], r["system"]) for r in sink.records if "system" in r]
    assert [step for step, _ in systems] == [4, 9, 10]  # each flush's last step, and the close
    assert systems[0][1]["cuda"]["allocated"] == 2 * GIB and systems[0][1]["nvml"]["ecc_uncorrected"] == 0
    assert systems[1][1]["nvml"]["ecc_uncorrected"] == 2 and systems[1][1]["nvml"]["xid"] == [79]
    assert set(cuda["calls"]) == {trainer_thread}
    assert trainer_thread not in nvml.threads and len(nvml.threads) == 1
    assert nvml.shutdowns == 1


def test_readings_that_fail_mid_run_never_stop_training(cuda, nvml, monkeypatch):
    session, model, sink = _session(system=True)
    session._system = SystemReadings(CUDA0)
    _train(session, model, 6)  # a full ring's newest step is written when the next step ends
    _written(sink, 5)
    cuda["stats"] = RuntimeError("allocator gone")
    monkeypatch.setattr(Nvml, "read", lambda self: (_ for _ in ()).throw(Exception("driver fell off the bus")))
    _train(session, model, 9)
    session.close()  # raises if the writer thread failed
    assert [r["step"] for r in sink.records] == list(range(15))
    notes = [r["system"] for r in sink.records if "system" in r]
    assert notes[1] == {"cuda": {"unavailable": "RuntimeError: allocator gone"},
                        "nvml": {"unavailable": "Exception: driver fell off the bus"}}
    assert len(notes) == 2  # each note once, then nothing


def test_readings_survive_the_flight_columns():
    pytest.importorskip("pyarrow")
    from tensorquill.flight import record_to_row, row_to_record

    record = {
        "schema_version": 1, "run_id": "r", "rank": 0, "attempt": 5, "step": 9, "ts": 1.0, "grad_norm": None,
        "data_wait": 0.25, "step_time": 0.5,
        "system": {"cuda": {"allocated": 2 * GIB, "total": 16 * GIB}, "nvml": {"xid": [79], "unavailable": {"ecc": "x"}}},
    }
    row = record_to_row(record)
    assert row["data_wait"] == 0.25 and row["step_time"] == 0.5 and row["system_json"] and row["extra_json"] is None
    assert row_to_record(row) == record
