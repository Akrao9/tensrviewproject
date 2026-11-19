"""Lane W1: capture and sender, the P1 rows C06, F1 (with C06), C03 (the SDK half) and L2.

Written first, before the fix, against `main` at 66b52d7. Each test says whether it fails there
(fail-before) or guards behaviour the fix must keep (guard).

- C06: a router's integer ids outside its experts. A negative id is no assignment: not counted,
  not a refusal, and how many there were is recorded. An id at or above `n_experts` refuses the
  router with a reason. Never a raise. The default integer path and the `indices=` path follow
  the same rule (the `indices=` path refused a -1 before).
- F1: a bool routing map's entries left out of the counts (their paired float is exactly zero)
  are recorded per expert, so Megatron's own count (`routing_map.sum(dim=0)`,
  `token_dispatcher.py:558`) is the recorded count plus the left-out count.
- C03, SDK half: after capture stops, the heartbeat says so at once, and a stop followed straight
  by `close()` still delivers that beat. Contract 3 (lane W2) names the field; until it is signed
  off these tests check only that a stopped rank's beat differs from a live one, never a name.
- L2: the writer never waits for an NVML reading; it attaches the newest finished one.

The two record fields are proposals pending the top session's sign-off; each name is one constant
below. No server binary is needed; the heartbeat tests skip without pyarrow.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture
from tensorquill import system as system_module
from tensorquill.system import SystemReadings

DEADLINE = 20.0  # seconds: a call that waits for good never returns, so a deadline is the only way to see it

# Proposed record fields (schema, pending sign-off by the top session; travel in `extra_json`):
UNASSIGNED = "moe_unassigned"  # {router: n}: negative ids in the step, no assignment; written when n > 0
LEFT_OUT = "moe_left_out"  # {router: [n per expert]}: map entries left out of `moe`; written when any > 0
EXPERTS = 4


# --- C06: integer ids outside the experts --------------------------------------------------------


class IdsRouter(nn.Module):
    """A router whose output is (logits, ids) with the ids the test sets, one row per token."""

    def __init__(self, ids: list[list[int]]) -> None:
        super().__init__()
        self.register_buffer("ids", torch.tensor(ids, dtype=torch.int64))
        self.proj = nn.Linear(4, EXPERTS)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proj(h), self.ids.clone()


class OneRouter(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.inp = nn.Linear(4, 4)
        self.router = router

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        logits = self.router(h)[0]
        return h.sum() + logits.sum()


ROUTES = {
    # The default reads the output's first integer tensor; `indices=` is given the same ids.
    "default": lambda: tensorquill.Router(EXPERTS, 1),
    "indices": lambda: tensorquill.Router(EXPERTS, 1, indices=lambda out: out[1]),
}


def _run(router: nn.Module, spec: tensorquill.Router, steps: int = 2, every: int = 1, backend: str | None = None,
         tokens: int | None = None) -> list[dict]:
    torch.manual_seed(0)
    model = OneRouter(router)
    sink = tensorquill.MemorySink()
    tokens = tokens or (router.ids if hasattr(router, "ids") else router.routing).shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, sink=sink, every=every, routers={"router": spec})
        stepped = torch.compile(model, backend=backend) if backend else model
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(steps):
            opt.zero_grad()
            stepped(torch.ones(tokens, 4)).backward()
            opt.step()
        session.close()
    assert session.writer_error is None
    return sink.records


def _refusal(records: list[dict]) -> str | None:
    found = [r["moe_refused"]["router"] for r in records if "router" in r.get("moe_refused", {})]
    return found[0] if found else None


@pytest.mark.parametrize("route", ROUTES)
def test_c06_an_id_at_or_above_n_experts_refuses_the_router(route):
    """The audit's probe: ids [-1, 8] for four experts. Fail-before on the default path (four zero
    counts, no refusal); a guard on `indices=` (refused there already, as outside 0..3)."""
    records = _run(IdsRouter([[-1], [8]]), ROUTES[route]())
    assert [r["step"] for r in records] == [0, 1]
    assert not any("router" in r.get("moe", {}) for r in records), "zeros were recorded for ids no expert has"
    why = _refusal(records)
    assert why is not None, "the router was not refused"


@pytest.mark.parametrize("route", ROUTES)
def test_c06_a_negative_id_is_no_assignment_and_the_rest_are_counted(route):
    """-1 is Megatron's sentinel for no route (`router.py:1002-1019`). Fail-before on `indices=`
    (it refused the router for the -1); a guard on the default path (counted, as here)."""
    records = _run(IdsRouter([[-1], [2], [2], [-1], [0]]), ROUTES[route]())
    assert _refusal(records) is None
    assert [r["moe"]["router"] for r in records] == [[1, 0, 2, 0]] * 2


@pytest.mark.parametrize("route", ROUTES)
def test_c06_how_many_ids_were_no_assignment_is_recorded(route):
    """Fail-before on both paths: nothing records the two -1 ids today."""
    records = _run(IdsRouter([[-1], [2], [2], [-1], [0]]), ROUTES[route]())
    assert [r.get(UNASSIGNED) for r in records] == [{"router": 2}] * 2


@pytest.mark.parametrize("route", ROUTES)
def test_c06_a_step_with_every_id_in_range_records_no_unassigned_count(route):
    """Guard: the field is written only when there is something to say."""
    records = _run(IdsRouter([[1], [2], [2], [3], [0]]), ROUTES[route]())
    assert [r["moe"]["router"] for r in records] == [[1, 1, 2, 1]] * 2
    assert not any(UNASSIGNED in r for r in records) and _refusal(records) is None


@pytest.mark.parametrize("route", ROUTES)
def test_c06_a_negative_id_and_one_too_high_in_one_step_refuse(route):
    """The refusal wins over the no-assignment count. Fail-before on the default path."""
    records = _run(IdsRouter([[-1], [2], [4], [1]]), ROUTES[route]())
    assert _refusal(records) is not None
    assert not any("router" in r.get("moe", {}) for r in records)


def test_c06_an_id_out_of_range_from_a_later_block_refuses_from_that_block():
    """Judged at the flush, from fixed slots the hook fills: the first block is counted, the router
    is refused from the block in which an id reached n_experts. Fail-before (never refused)."""

    class Drifting(IdsRouter):
        calls = 0

        def forward(self, h):
            Drifting.calls += 1
            logits, ids = super().forward(h)
            return logits, ids + EXPERTS if Drifting.calls == 3 else ids

    records = _run(Drifting([[0], [1], [2], [3]]), tensorquill.Router(EXPERTS, 1), steps=4, every=2)
    assert [r["moe"]["router"] for r in records[:2]] == [[1, 1, 1, 1]] * 2
    assert not any("router" in r.get("moe", {}) for r in records[2:])
    assert [r["step"] for r in records if "moe_refused" in r] == [2]


@pytest.mark.parametrize("route", ROUTES)
def test_c06_nothing_raises_under_warnings_as_errors(route):
    """Guard on raising (none on base either); fail-before on the default path's refusal."""
    torch.manual_seed(0)
    model = OneRouter(IdsRouter([[-1], [8]]))
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session = tensorquill.init(model, sink=sink, every=1, routers={"router": ROUTES[route]()}, auto_step=False)
        for _ in range(2):
            model(torch.ones(2, 4)).backward()
            session.step()
        session.close()
    assert _refusal(sink.records) is not None


def test_c06_logits_through_top_k_are_counted_as_before():
    """Guard: a router that returns only logits is counted from their top-k, with no new field."""

    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, EXPERTS)

        def forward(self, h):
            return (self.proj(h),)

    records = _run(Logits(), tensorquill.Router(EXPERTS, 2), tokens=5)
    assert [sum(r["moe"]["router"]) for r in records] == [10, 10]
    assert not any(UNASSIGNED in r or LEFT_OUT in r for r in records) and _refusal(records) is None


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_c06_compiled_counts_and_refuses_as_eager_does(backend):
    """Counted in fixed slots in the hook and judged at the flush, so a compiled run records what an
    eager one does and nothing raises. Fail-before (the unassigned count; the refusal)."""
    torch.manual_seed(0)
    control = torch.compile(OneRouter(IdsRouter([[0], [1]])), backend=backend)
    try:
        control(torch.ones(2, 4)).backward()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"this PyTorch cannot compile the model without TensorQuill: {type(error).__name__}")
    torch._dynamo.reset()
    eager = _run(IdsRouter([[-1], [2], [2], [-1], [0]]), tensorquill.Router(EXPERTS, 1))
    torch._dynamo.reset()
    compiled = _run(IdsRouter([[-1], [2], [2], [-1], [0]]), tensorquill.Router(EXPERTS, 1), backend=backend)
    assert [(r["moe"], r.get(UNASSIGNED)) for r in compiled] == [(r["moe"], r.get(UNASSIGNED)) for r in eager]
    assert compiled[0][UNASSIGNED] == {"router": 2}
    torch._dynamo.reset()
    refused = _run(IdsRouter([[-1], [8]]), tensorquill.Router(EXPERTS, 1), backend=backend)
    assert _refusal(refused) is not None and not any("router" in r.get("moe", {}) for r in refused)


# --- F1: the routing map's entries left out ----------------------------------------------------


class MapRouter(nn.Module):
    """Megatron's `TopKRouter` shape: (probs, routing_map), both [tokens, n_experts]. The map routes
    each token to one expert; entries the test pads in have a probability of exactly zero, as
    `pad_to_capacity` leaves them."""

    def __init__(self, routed: list[int], padded: list[tuple[int, int]]) -> None:
        super().__init__()
        tokens = len(routed)
        routing = torch.zeros(tokens, EXPERTS, dtype=torch.bool)
        probs = torch.zeros(tokens, EXPERTS)
        for token, expert in enumerate(routed):
            routing[token, expert] = True
            probs[token, expert] = 1.0
        for token, expert in padded:
            routing[token, expert] = True  # in the map, with a zero probability
        self.register_buffer("routing", routing)
        self.register_buffer("probs", probs)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, h):
        return self.probs * self.scale, self.routing.clone()


def test_f1_counts_keep_real_assignments():
    """Guard: the author's decision on F1 (lane U1) stands."""
    records = _run(MapRouter([0, 1, 1, 3], padded=[(0, 2), (2, 2), (3, 0)]), tensorquill.Router(EXPERTS, 1))
    assert [r["moe"]["router"] for r in records] == [[1, 2, 0, 1]] * 2


def test_f1_the_map_entries_left_out_are_recorded_per_expert():
    """Fail-before: nothing records the three padded entries. The recorded count plus the left-out
    count is Megatron's own `routing_map.sum(dim=0)`."""
    router = MapRouter([0, 1, 1, 3], padded=[(0, 2), (2, 2), (3, 0)])
    records = _run(router, tensorquill.Router(EXPERTS, 1))
    megatron = router.routing.sum(0).tolist()
    assert [r.get(LEFT_OUT) for r in records] == [{"router": [1, 0, 2, 0]}] * 2
    assert [a + b for a, b in zip(records[0]["moe"]["router"], records[0][LEFT_OUT]["router"])] == megatron


def test_f1_a_map_with_nothing_left_out_records_no_left_out_count():
    """Guard: written only when an entry was left out."""
    records = _run(MapRouter([0, 1, 1, 3], padded=[]), tensorquill.Router(EXPERTS, 1))
    assert not any(LEFT_OUT in r for r in records)


# --- C03, SDK half: the heartbeat after capture stops -------------------------------------------


def _beat_server():
    """An in-process stand-in for the server that keeps every heartbeat body and acks every batch."""
    pa = pytest.importorskip("pyarrow")
    flight = pytest.importorskip("pyarrow.flight")

    class Beats(flight.FlightServerBase):
        def __init__(self) -> None:
            super().__init__("grpc://127.0.0.1:0")
            self.beats: list[dict] = []
            self.seen = threading.Condition()

        def do_action(self, context, action):
            if action.type == "heartbeat":
                with self.seen:
                    self.beats.append(json.loads(action.body.to_pybytes()))
                    self.seen.notify_all()
            return iter(())

        def do_put(self, context, descriptor, reader, writer):
            while True:
                try:
                    chunk = reader.read_chunk()
                except StopIteration:
                    return
                end = json.loads(chunk.app_metadata.to_pybytes())["end_byte_offset"]
                writer.write(pa.py_buffer(json.dumps({"end_byte_offset": end}).encode()))

        def wait_for(self, condition, timeout: float = DEADLINE) -> bool:
            with self.seen:
                return self.seen.wait_for(lambda: condition(list(self.beats)), timeout)

    return Beats()


class FailingJsonl(tensorquill.JsonlSink):
    """Writes its first `good` blocks, then raises, as a disk that fills up."""

    def __init__(self, path: Path, good: int) -> None:
        super().__init__(path)
        self.good = good

    def write(self, records):
        if self.good <= 0:
            raise OSError("disk gone")
        self.good -= 1
        super().write(records)


def _state(beat: dict) -> dict:
    """What a beat says beyond the work done and where the threads are."""
    return {k: v for k, v in beat.items() if k not in ("work", "stack")}


def _streamed_session_that_stops(tmp_path, server):
    """A streamed session (one beat a minute, so any later beat is one sent at once) whose sink fails
    at its second write, stepped until capture has stopped. Returns it and the beats before the stop."""
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(
            model, run_id="w1-stop", every=1, sink=FailingJsonl(tmp_path / "w1-stop.rank0.jsonl", good=1),
            endpoint=f"grpc://127.0.0.1:{server.port}", heartbeat_seconds=60.0, stack_seconds=None,
        )
        assert server.wait_for(lambda beats: any("work" in b for b in beats)), "no beat named the session's work"
        live = [_state(b) for b in server.beats]
        deadline = time.monotonic() + DEADLINE
        while not session._stopped and time.monotonic() < deadline:
            opt.zero_grad()
            model(torch.ones(2, 4)).sum().backward()
            opt.step()
            if session._error is None:
                time.sleep(0.01)  # the writer's turn
    assert session._stopped, "capture did not stop"
    return session, model, live


def test_c03_after_capture_stops_a_beat_saying_so_goes_at_once(tmp_path):
    """Fail-before: no beat is sent at the stop, and every later one reads as a live rank's."""
    server = _beat_server()
    try:
        session, model, live = _streamed_session_that_stops(tmp_path, server)
        told = server.wait_for(lambda beats: any(_state(b) not in live for b in beats), timeout=10.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.close()
        assert told, "no heartbeat after the stop said anything a live rank's does not"
    finally:
        server.shutdown()


def test_c03_a_stop_followed_at_once_by_close_still_delivers_the_stopped_beat(tmp_path):
    """Fail-before. `Heartbeat.close` ends its loop without a last beat; the stopped state must reach the
    server before the session's heartbeat ends, so a closed stopped attempt reads unlike a finished one."""
    server = _beat_server()
    try:
        session, model, live = _streamed_session_that_stops(tmp_path, server)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.close()
        assert any(_state(b) not in live for b in server.beats), "close() ended the heartbeat before the stop was told"
    finally:
        server.shutdown()


def test_c03_training_goes_on_after_the_stop_and_nothing_raises(tmp_path):
    """Guard: the stopped beat is sent from the heartbeat's thread; training is neither held nor raised into."""
    server = _beat_server()
    try:
        session, model, live = _streamed_session_that_stops(tmp_path, server)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _ in range(5):
                model(torch.ones(2, 4)).sum().backward()
            session.set_phase("eval")
            session.close()
    finally:
        server.shutdown()


# --- L2: the writer never waits for an NVML reading ----------------------------------------------

CUDA0 = torch.device("cuda", 0)
LONG = 30.0  # bounds far beyond any deadline here: a writer that waits for NVML is seen waiting


class HangingNvml(types.ModuleType):
    """The parts of `pynvml` that `system.py` calls; `hang` names calls that block until `release`."""

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

    def __init__(self) -> None:
        super().__init__("pynvml")
        self.hang: set[str] = set()
        self.release = threading.Event()
        self.entered = threading.Event()  # a hung call has been entered
        self.xid: list[int] = []
        self.temperature = 71
        self.calls: list[str] = []

    def _call(self, name):
        self.calls.append(name)
        if name in self.hang:
            self.entered.set()
            self.release.wait()

    def nvmlInit(self):
        self._call("init")

    def nvmlShutdown(self):
        self._call("shutdown")

    def nvmlDeviceGetHandleByUUID(self, uuid):
        return "handle"

    def nvmlDeviceGetHandleByIndex(self, index):
        return "handle"

    def nvmlDeviceGetTemperature(self, handle, sensor):
        self._call("temperature")
        return self.temperature

    def nvmlDeviceGetClockInfo(self, handle, clock):
        return 1410

    def nvmlDeviceGetPowerUsage(self, handle):
        return 250_000

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        return 400_000

    def nvmlDeviceGetTotalEccErrors(self, handle, kind, counter):
        return 0

    def nvmlDeviceGetCurrentClocksEventReasons(self, handle):
        return 0

    def nvmlEventSetCreate(self):
        return "events"

    def nvmlDeviceRegisterEvents(self, handle, kinds, events):
        pass

    def nvmlEventSetFree(self, events):
        pass

    def nvmlEventSetWait_v2(self, events, timeout):
        if not self.xid:
            raise self.NVMLError(10)  # NVML_ERROR_TIMEOUT
        return types.SimpleNamespace(eventData=self.xid.pop(0))


@pytest.fixture
def fake_gpu(monkeypatch):
    """CUDA device 0 as `system.py` sees it, NVML faked, and both of T3's bounds far beyond any deadline."""
    monkeypatch.setattr(torch.cuda, "memory_stats_as_nested_dict", lambda index: {
        "allocated_bytes": {"all": {"current": 1, "peak": 1}}, "reserved_bytes": {"all": {"current": 1}}})
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: types.SimpleNamespace(total_memory=1 << 30, uuid="1234-abcd"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(system_module, "NVML_SLOW_SECONDS", LONG, raising=False)
    monkeypatch.setattr(system_module, "NVML_OPEN_SECONDS", LONG, raising=False)
    fake = HangingNvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    yield fake
    fake.release.set()  # no call stays hung past its test


def _on_a_thread(call):
    """Run `call` on a thread of its own; returns the thread and where its result goes."""
    out: dict = {}
    thread = threading.Thread(target=lambda: out.update(result=call()), daemon=True)
    thread.start()
    return thread, out


def test_l2_the_writer_returns_while_an_nvml_reading_hangs(fake_gpu):
    """Fail-before: `write` waits for the reading (up to the bound, here 30 s)."""
    readings = SystemReadings(CUDA0)
    fake_gpu.hang.add("temperature")
    thread, out = _on_a_thread(lambda: readings.write({"cuda": {"allocated": 1}}))
    thread.join(5.0)
    assert not thread.is_alive(), "the writer waited for a hung NVML reading"
    assert out["result"] == {"cuda": {"allocated": 1}}  # no reading had finished: nothing attached
    thread, out = _on_a_thread(lambda: readings.write({}))
    thread.join(5.0)
    assert not thread.is_alive(), "the writer waited for the reading still in the driver"


def test_l2_a_finished_reading_is_attached_once_at_a_later_flush(fake_gpu):
    """Guard on base; after the fix the reading taken after one flush rides on a later one. The newest
    finished reading is attached once: an Xid event is never written twice."""
    readings = SystemReadings(CUDA0)
    fake_gpu.xid = [79]
    parts = []
    deadline = time.monotonic() + DEADLINE
    while not parts and time.monotonic() < deadline:
        written = readings.write({}) or {}
        if "nvml" in written:
            parts.append(written["nvml"])
        else:
            time.sleep(0.01)
    assert parts and parts[0]["temperature_c"] == 71
    for _ in range(20):
        written = readings.write({}) or {}
        if "nvml" in written:
            parts.append(written["nvml"])
        time.sleep(0.005)
    readings.close()
    assert sum(p.get("xid", []).count(79) for p in parts) == 1


def test_l2_a_hung_nvml_reading_never_stops_capture_however_fast_the_steps(fake_gpu, monkeypatch):
    """The row: T3's bounded wait holds the writer, and a writer a full queue behind stops capture
    (lane U1's decision on C03). With a queue of two steps and the bound at 30 s, the hung reading
    stops capture on base. Fail-before."""
    monkeypatch.setattr(capture, "QUEUE_STEPS", 2, raising=False)
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, sink=sink, every=1, system=True)
        session._system = SystemReadings(CUDA0)  # as on a CUDA model
        fake_gpu.hang.add("temperature")
        done = threading.Event()

        def train():
            for _ in range(40):
                opt.zero_grad()
                model(torch.ones(2, 4)).sum().backward()
                opt.step()
            done.set()

        threading.Thread(target=train, daemon=True).start()
        finished = done.wait(DEADLINE)
        stopped = session.writer_error
        fake_gpu.release.set()
        session.close()
    assert finished, "training waited"
    assert stopped is None, f"capture stopped: {stopped}"
    assert [r["step"] for r in sink.records] == list(range(40))
    assert all("cuda" in r.get("system", {}) for r in sink.records)  # memory is read at every flush


def test_l2_a_reading_still_in_the_driver_past_the_bound_turns_nvml_off_once(fake_gpu, monkeypatch):
    """Guard: the note and NVML going off stay, judged at a later flush instead of waited for. Short
    bounds here; the note is written once and nothing is asked of NVML after it."""
    monkeypatch.setattr(system_module, "NVML_SLOW_SECONDS", 0.05, raising=False)
    monkeypatch.setattr(system_module, "NVML_OPEN_SECONDS", 0.05, raising=False)
    readings = SystemReadings(CUDA0)
    fake_gpu.hang.add("init")
    notes = []
    deadline = time.monotonic() + DEADLINE
    while not notes and time.monotonic() < deadline:
        written = readings.write({}) or {}
        if "nvml" in written:
            notes.append(written["nvml"])
        else:
            time.sleep(0.01)
    assert notes and "NVML is off for this session" in notes[0]["unavailable"]
    asked = len(fake_gpu.calls)
    assert all("nvml" not in (readings.write({}) or {}) for _ in range(5))
    readings.close()
    assert len(fake_gpu.calls) == asked


def test_l2_a_cpu_model_is_told_at_the_first_flush():
    """Guard: what needs no NVML call (a model not on CUDA) is said at the first flush, as before."""
    readings = SystemReadings(torch.device("cpu"))
    written = readings.write({})
    assert written["nvml"] == {"unavailable": "the model is on cpu, not a CUDA device"}
    assert readings.write({}) is None
    readings.close()
