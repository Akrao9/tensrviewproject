"""Lane W1: capture and sender, the P1 rows C06, F1 (with C06), C03 (the SDK half) and L2.

Written first, before the fix, against `main` at 66b52d7. Each test says whether it fails there
(fail-before) or guards behaviour the fix must keep (guard).

- C06: a router's integer ids outside its experts. A negative id is no assignment: not counted,
  not a refusal, and how many there were is recorded (`moe_unassigned`). An id at or above `n_experts` refuses the
  router with a reason. Never a raise. The default integer path and the `indices=` path follow
  the same rule (the `indices=` path refused a -1 before).
- F1: a bool routing map's entries left out of the counts (their paired float is exactly zero)
  are recorded per expert, so Megatron's own count (`routing_map.sum(dim=0)`,
  `token_dispatcher.py:558`) is the recorded count plus the left-out count (`moe_left_out`).
- The author's rule for both fields: an explicit zero whenever the router was counted and the check
  applied (ids checked; a routing map with a paired float checked); no key for a router the check did
  not apply to or that was refused. A missing value, every older record's included, is unknown, not 0.
- C03, SDK half (contract 3, signed): after capture stops, every beat carries
  `"capture_stopped": "<reason>"` (`sink_error`, `writer_behind`, `close_timeout`) and no `stack`; the
  stop sends a beat at once, and `Heartbeat.close()` sends one bounded last beat if it was not yet sent.
- Contract 1, the sender's part (signed): each batch's `app_metadata` names `sender_attempt`. A beat or
  batch the server refuses as UNAVAILABLE is handled as today, which the contract keeps: the heartbeat
  retries with the same attempt at its next beat, silently; the sender prints one line per outage and
  resends from the acknowledged offset.
- L2: the writer never waits for an NVML reading; it attaches the newest finished one.

The record fields and the heartbeat key are signed off. No server binary is needed; the heartbeat
and sender tests skip without pyarrow.
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

# Record fields, signed off (they travel in `extra_json`; no Arrow or Parquet column):
UNASSIGNED = "moe_unassigned"  # {router: n}: negative ids in the step, no assignment; 0 when ids were checked
LEFT_OUT = "moe_left_out"  # {router: [n per expert]}: map entries left out of `moe`; zeros when the map was checked
STOPPED = "capture_stopped"  # heartbeat key, contract 3
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
    assert not any(LEFT_OUT in r for r in records)  # ids, not a routing map: that check does not apply


@pytest.mark.parametrize("route", ROUTES)
def test_c06_ids_all_in_range_record_an_explicit_zero(route):
    """Fail-before. "Checked and found none" is a result, written as 0, unlike a writer that did not check."""
    records = _run(IdsRouter([[1], [2], [2], [3], [0]]), ROUTES[route]())
    assert [r["moe"]["router"] for r in records] == [[1, 1, 2, 1]] * 2
    assert [r.get(UNASSIGNED) for r in records] == [{"router": 0}] * 2 and _refusal(records) is None


@pytest.mark.parametrize("route", ROUTES)
def test_c06_a_negative_id_and_one_too_high_in_one_step_refuse(route):
    """The refusal wins over the no-assignment count. Fail-before on the default path."""
    records = _run(IdsRouter([[-1], [2], [4], [1]]), ROUTES[route]())
    assert _refusal(records) is not None
    assert not any("router" in r.get("moe", {}) for r in records)
    assert not any("router" in r.get(UNASSIGNED, {}) for r in records)  # a refused router is not counted


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
    """Guard: a router that returns only logits is counted from their top-k. It has no ids to check and no
    routing map, so neither check applies and neither key is written."""

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


def test_f1_a_map_with_nothing_left_out_records_explicit_zeros():
    """Fail-before. The map was checked and nothing was left out: a vector of zeros, one per expert."""
    records = _run(MapRouter([0, 1, 1, 3], padded=[]), tensorquill.Router(EXPERTS, 1))
    assert [r.get(LEFT_OUT) for r in records] == [{"router": [0, 0, 0, 0]}] * 2
    assert not any(UNASSIGNED in r for r in records)  # a map, not ids: that check does not apply


def test_f1_a_map_with_no_paired_float_has_no_left_out_key():
    """Guard. With no float tensor of the map's shape, every true entry is counted (U1's rule) and no entry
    can be left out: the check does not apply, so no key, not zeros."""

    class Unpaired(MapRouter):
        def forward(self, h):
            return (self.probs * self.scale).sum(-1), self.routing.clone()  # [tokens]: not the map's shape

    records = _run(Unpaired([0, 1, 1, 3], padded=[(0, 2)]), tensorquill.Router(EXPERTS, 1))
    assert [r["moe"]["router"] for r in records] == [[1, 2, 1, 1]] * 2
    assert not any(LEFT_OUT in r or UNASSIGNED in r for r in records)


def test_a_record_without_the_new_keys_reads_back_without_them_and_zeros_stay_zeros():
    """Guard, on the sender's columns (`flight.py`): a record from before this change has no key and comes
    back from the Arrow row with no key, never a filled-in 0, so a reader sees "unknown". An explicit 0 and
    a zero vector come back as written. What the evidence makes of a missing key is lane W3's."""
    pytest.importorskip("pyarrow")
    from tensorquill.flight import record_to_row, row_to_record

    older = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": 3, "ts": 1.0, "grad_norm": None,
             "layers": {}, "moe": {"router": [1, 2, 0, 1]}}
    back = row_to_record(record_to_row(older))
    assert UNASSIGNED not in back and LEFT_OUT not in back
    newer = {**older, UNASSIGNED: {"router": 0}, LEFT_OUT: {"router": [0, 0, 0, 0]}}
    back = row_to_record(record_to_row(newer))
    assert back[UNASSIGNED] == {"router": 0} and back[LEFT_OUT] == {"router": [0, 0, 0, 0]}


# --- C03, SDK half: the heartbeat after capture stops -------------------------------------------


def _beat_server(fail_beats: int = 0, fail_batches: int = 0):
    """An in-process stand-in for the server that keeps every heartbeat body, every batch's metadata and
    rows, and acks every batch. The first `fail_beats` heartbeats and the first `fail_batches` batches get
    a retryable failure (UNAVAILABLE), as the server answers when it cannot write `attempts.json`
    (contract 1, the author's amendment 2): such a beat or batch is not taken."""
    pa = pytest.importorskip("pyarrow")
    flight = pytest.importorskip("pyarrow.flight")

    class Beats(flight.FlightServerBase):
        def __init__(self) -> None:
            super().__init__("grpc://127.0.0.1:0")
            self.beats: list[dict] = []
            self.refused_beats: list[dict] = []
            self.metadata: list[dict] = []  # each batch taken: its app_metadata
            self.rows: list[dict] = []
            self.refused_batches = 0
            self.seen = threading.Condition()

        def do_action(self, context, action):
            if action.type == "heartbeat":
                beat = json.loads(action.body.to_pybytes())
                with self.seen:
                    if len(self.refused_beats) < fail_beats:
                        self.refused_beats.append(beat)
                        self.seen.notify_all()
                        raise flight.FlightUnavailableError("could not write attempts.json")
                    self.beats.append(beat)
                    self.seen.notify_all()
            return iter(())

        def do_put(self, context, descriptor, reader, writer):
            while True:
                try:
                    chunk = reader.read_chunk()
                except StopIteration:
                    return
                metadata = json.loads(chunk.app_metadata.to_pybytes())
                with self.seen:
                    if self.refused_batches < fail_batches:
                        self.refused_batches += 1
                        self.seen.notify_all()
                        raise flight.FlightUnavailableError("could not write attempts.json")
                    self.metadata.append(metadata)
                    self.rows.extend(chunk.data.to_pylist())
                    self.seen.notify_all()
                writer.write(pa.py_buffer(json.dumps({"end_byte_offset": metadata["end_byte_offset"]}).encode()))

        def wait_for(self, condition, timeout: float = DEADLINE) -> bool:
            with self.seen:
                return self.seen.wait_for(lambda: condition(list(self.beats)), timeout)

        def wait_until(self, condition, timeout: float = DEADLINE) -> bool:
            with self.seen:
                return self.seen.wait_for(condition, timeout)

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


def _streamed_session_that_stops(tmp_path, server, heartbeat_seconds: float = 60.0, stack_seconds=None):
    """A streamed session whose sink fails at its second write, stepped until capture has stopped. With one
    beat a minute (the default here) any later beat is one sent at once. Returns it, its model and the beats
    taken before the stop."""
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(
            model, run_id="w1-stop", every=1, sink=FailingJsonl(tmp_path / "w1-stop.rank0.jsonl", good=1),
            endpoint=f"grpc://127.0.0.1:{server.port}", heartbeat_seconds=heartbeat_seconds, stack_seconds=stack_seconds,
        )
        assert server.wait_for(lambda beats: any("work" in b for b in beats)), "no beat named the session's work"
        live = list(server.beats)
        deadline = time.monotonic() + DEADLINE
        while not session._stopped and time.monotonic() < deadline:
            opt.zero_grad()
            model(torch.ones(2, 4)).sum().backward()
            opt.step()
            if session._error is None:
                time.sleep(0.01)  # the writer's turn
    assert session._stopped, "capture did not stop"
    return session, model, live


def _close(session) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.close()


def test_c03_after_capture_stops_a_beat_saying_so_goes_at_once(tmp_path):
    """Fail-before: no beat is sent at the stop, and none says `capture_stopped`. Live beats never carry it."""
    server = _beat_server()
    try:
        session, model, live = _streamed_session_that_stops(tmp_path, server)
        told = server.wait_for(lambda beats: any(STOPPED in b for b in beats), timeout=10.0)
        _close(session)
        assert not any(STOPPED in b for b in live)
        assert told, "no heartbeat said capture_stopped within 10 s of the stop (the interval is 60 s)"
        stopped = [b for b in server.beats if STOPPED in b]
        assert stopped[0][STOPPED] == "sink_error" and stopped[0]["attempt"] == session.attempt
    finally:
        server.shutdown()


def test_c03_every_beat_after_the_stop_says_so_and_none_carries_a_stack(tmp_path):
    """Contract 3: the key goes on every beat after the stop, so a restarted server learns it again, and no
    beat of a stopped session carries `stack` (its work stands still while training goes on normally).
    Beats every 0.1 s and stacks after 0.05 s of no work, so on base the stopped beats would carry stacks.
    Fail-before."""
    server = _beat_server()
    try:
        session, model, live = _streamed_session_that_stops(tmp_path, server, heartbeat_seconds=0.1,
                                                            stack_seconds=0.05)
        told = server.wait_for(lambda beats: sum(STOPPED in b for b in beats) >= 4, timeout=10.0)
        _close(session)
        assert told, "fewer than four beats said capture_stopped"
        first = next(i for i, b in enumerate(server.beats) if STOPPED in b)
        after = server.beats[first:]
        assert all(b.get(STOPPED) == "sink_error" for b in after)
        assert not any("stack" in b for b in after)
    finally:
        server.shutdown()


def test_c03_a_stop_followed_at_once_by_close_still_delivers_the_stopped_beat(tmp_path):
    """Fail-before. `Heartbeat.close` ends its loop without a last beat; the stopped state must reach the
    server before the session's heartbeat ends, so a closed stopped attempt reads unlike a finished one."""
    server = _beat_server()
    try:
        session, model, live = _streamed_session_that_stops(tmp_path, server)
        _close(session)
        assert [b[STOPPED] for b in server.beats if STOPPED in b][:1] == ["sink_error"], \
            "close() ended the heartbeat before the stop was told"
    finally:
        server.shutdown()


def test_c03_a_writer_a_full_queue_behind_is_told_as_writer_behind(tmp_path, monkeypatch):
    """Fail-before. A sink that stalls (not fails) with a queue of two steps: capture stops as `_put` finds the
    queue full, and the beat says `writer_behind`."""
    monkeypatch.setattr(capture, "QUEUE_STEPS", 2, raising=False)
    release = threading.Event()

    class Stalling(tensorquill.JsonlSink):
        def write(self, records):
            release.wait()
            super().write(records)

    server = _beat_server()
    try:
        model = nn.Linear(4, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = tensorquill.init(model, run_id="w1-behind", every=1, sink=Stalling(tmp_path / "b.rank0.jsonl"),
                                       endpoint=f"grpc://127.0.0.1:{server.port}", heartbeat_seconds=60.0,
                                       stack_seconds=None)
            deadline = time.monotonic() + DEADLINE
            while not session._stopped and time.monotonic() < deadline:
                _train(model, 1)
        assert session._stopped, "capture did not stop"
        told = server.wait_for(lambda beats: any(STOPPED in b for b in beats), timeout=10.0)
        release.set()
        _close(session)
        assert told and [b[STOPPED] for b in server.beats if STOPPED in b][0] == "writer_behind"
    finally:
        release.set()
        server.shutdown()


def test_c03_a_close_that_runs_out_of_time_is_told_as_close_timeout(tmp_path, monkeypatch):
    """Fail-before. The writer is still in the sink when close()'s deadline passes: capture stops at close,
    and the bounded last beat says `close_timeout`."""
    monkeypatch.setattr(capture, "CLOSE_SECONDS", 0.3)
    release = threading.Event()

    class Slow(tensorquill.JsonlSink):
        def write(self, records):
            release.wait()
            super().write(records)

    server = _beat_server()
    try:
        model = nn.Linear(4, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = tensorquill.init(model, run_id="w1-close", every=50, sink=Slow(tmp_path / "c.rank0.jsonl"),
                                       endpoint=f"grpc://127.0.0.1:{server.port}", heartbeat_seconds=60.0,
                                       stack_seconds=None, auto_step=False)
            session.step()
            session.close()
        release.set()
        assert session._stopped
        assert [b[STOPPED] for b in server.beats if STOPPED in b][:1] == ["close_timeout"]
    finally:
        release.set()
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


# --- Contract 1, the sender's part: `sender_attempt`, and a refused beat or batch is retried -----


def _train(model, steps: int) -> None:
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()


def _streamed(server, path: Path, run_id: str = "w1-claim", **options):
    model = nn.Linear(4, 4)
    options = {"heartbeat_seconds": 60.0, "stack_seconds": None, **options}
    session = tensorquill.init(model, run_id=run_id, every=1, sink=tensorquill.JsonlSink(path),
                               endpoint=f"grpc://127.0.0.1:{server.port}", **options)
    return session, model


def test_contract1_every_batch_names_the_attempt_of_the_session_that_sends_it(tmp_path):
    """Fail-before: no batch's metadata names `sender_attempt`."""
    server = _beat_server()
    try:
        session, model = _streamed(server, tmp_path / "w1-claim.rank0.jsonl")
        _train(model, 6)
        session.close()
        assert server.metadata and {r["attempt"] for r in server.rows} == {session.attempt}
        assert [m.get("sender_attempt") for m in server.metadata] == [session.attempt] * len(server.metadata)
        assert {b.get("attempt") for b in server.beats} == {session.attempt}  # the same attempt the beats claim
    finally:
        server.shutdown()


def test_contract1_an_earlier_attempts_backlog_goes_under_the_sending_sessions_attempt(tmp_path):
    """The contract's backlog case: a session streams a file that still holds an earlier attempt's unsent
    rows. Those rows keep their own `attempt`; the batches name the session sending them, so the server
    places the earlier attempt just before it (rule 3). Fail-before."""
    path = tmp_path / "w1-claim.rank0.jsonl"
    model = nn.Linear(4, 4)
    first = tensorquill.init(model, run_id="w1-claim", every=1, sink=tensorquill.JsonlSink(path))  # not streamed
    _train(model, 3)
    first.close()
    server = _beat_server()
    try:
        session, model = _streamed(server, path)
        _train(model, 3)
        session.close()
        attempts = [r["attempt"] for r in server.rows]
        assert first.attempt in attempts and session.attempt in attempts and first.attempt != session.attempt
        assert {m.get("sender_attempt") for m in server.metadata} == {session.attempt}
    finally:
        server.shutdown()


def test_contract1_a_sender_with_no_heartbeat_still_names_its_attempt(tmp_path):
    """Fail-before. With `heartbeat_seconds=None` the batches are the only claim the server gets."""
    server = _beat_server()
    try:
        session, model = _streamed(server, tmp_path / "w1-claim.rank0.jsonl", heartbeat_seconds=None)
        _train(model, 4)
        session.close()
        assert not server.beats and server.metadata
        assert {m.get("sender_attempt") for m in server.metadata} == {session.attempt}
    finally:
        server.shutdown()


def test_contract1_a_refused_claim_is_sent_again_at_the_next_beat_silently(tmp_path, capfd):
    """Contract 1 keeps today's handling of a beat the server refuses as UNAVAILABLE (it could not write the
    attempt order): the heartbeat catches it, and its next beat, one interval later, names the same attempt.
    Nothing is printed or raised, the stream still opens, and training goes on. Guard."""
    server = _beat_server(fail_beats=2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session, model = _streamed(server, tmp_path / "w1-claim.rank0.jsonl", heartbeat_seconds=0.2)
            _train(model, 5)
            claimed = server.wait_for(lambda beats: any(b.get("attempt") == session.attempt for b in beats))
            _train(model, 5)
            session.close()
        assert claimed, "the refused claim was not sent again"
        assert [b.get("attempt") for b in server.refused_beats] == [session.attempt] * 2
        assert sorted({r["step"] for r in server.rows}) == list(range(10))
        assert session.writer_error is None
        assert "tensorquill" not in capfd.readouterr().err
    finally:
        server.shutdown()


def test_contract1_a_refused_batch_is_retried_and_every_row_arrives(tmp_path, capfd):
    """Guard: a batch refused as UNAVAILABLE is handled as today, which contract 1 keeps. The sender says so
    once per outage on stderr and resends from the acknowledged offset with backoff; capture does not stop,
    training goes on, and nothing raises. The rows are waited for before close(), so the outage is a
    streaming one (at close() the sender reports on its own)."""
    server = _beat_server(fail_batches=2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            session, model = _streamed(server, tmp_path / "w1-claim.rank0.jsonl")
            _train(model, 10)
            arrived = server.wait_until(lambda: {r["step"] for r in server.rows} >= set(range(9)))
            session.close()
        assert arrived and server.refused_batches == 2
        assert sorted({r["step"] for r in server.rows}) == list(range(10))
        assert session.writer_error is None
        assert capfd.readouterr().err.count("is failing") == 1
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
