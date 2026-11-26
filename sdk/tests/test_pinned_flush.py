"""Lane AQ-1: the pinned-memory flush (`init(pinned_flush=True)`).

With the switch on, a flush packs the ring into one fresh staging tensor on the device, copies it into a
pinned host buffer on a copy stream of its own, and leaves every host read of the block to the writer
thread, which waits for the copy's event first. The default (`pinned_flush=False`) is today's blocking
`.cpu()` path, unchanged.

This Mac has no CUDA. The tests stand a CPU copier in for the CUDA one (`FakeCopies`): its host buffers
start poisoned (every byte 0xFF), and its copy lands either only when the writer waits for the event
(`late`) or at once when it is issued (`eager`). What that can show:

- the writer reads a block only after waiting for its event (`late`: a read before the wait sees the
  poison), and the training thread never waits for one (asserted by the event);
- the copy reads a snapshot, not the ring (`late`: the next steps overwrite the ring before the copy runs);
- a host buffer is not reused while the writer has not taken a block out of it (`eager` with a slow sink:
  the next block's copy would land on the earlier one);
- the records are the same as the blocking path's, value for value, in every capture mode tried;
- no host read of a tensor on the training thread at all, close included;
- every failure falls back to the blocking transfer and says so once.

What it cannot show: that CUDA orders the copy stream after the training stream (it has no streams), that
the caching allocator keeps the staging tensor while the copy is in flight, or any timing. The CUDA copier's
own sequence of calls is checked against a stand-in for `torch.cuda` (`test_the_cuda_copier_...`); what the
GPU does with them is for `examples/pinned_flush_check.py` on a GPU.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import warnings

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture

from conftest import ToyMoE

# What differs between any two runs of the same seeded loop.
VOLATILE = {"ts", "attempt", "run_id", "step_time", "data_wait", "system"}


def _clean(value):
    """The record without what differs between runs, and NaN as a string so that it equals itself."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if key not in VOLATILE}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and value != value:
        return "nan"
    return value


class FakeEvent:
    def __init__(self, copies: "FakeCopies", staging: torch.Tensor, host: torch.Tensor) -> None:
        self.copies, self.staging, self.host, self.done = copies, staging, host, False

    def run(self) -> None:
        if not self.done:
            self.host[: self.staging.numel()].copy_(self.staging)
            self.done = True

    def synchronize(self) -> None:
        assert threading.get_ident() != self.copies.training, "the training thread waited for a copy"
        self.copies.waits += 1
        if self.copies.fail_wait is not None and self.copies.fail_wait(self.copies):
            raise RuntimeError("CUDA error: an illegal memory access was encountered (stand-in)")
        if self.copies.wait_seconds:
            time.sleep(self.copies.wait_seconds)
        self.run()


class FakeCopies:
    """The CUDA copier's interface (`buffer`, `copy`) on the CPU."""

    def __init__(self, mode: str = "late", *, fail_buffer=None, fail_copy=None, fail_wait=None,
                 wait_seconds: float = 0.0) -> None:
        assert mode in ("late", "eager")
        self.mode = mode
        self.training = threading.get_ident()
        self.buffers = self.copies = self.waits = 0
        self.fail_buffer, self.fail_copy, self.fail_wait = fail_buffer, fail_copy, fail_wait
        self.wait_seconds = wait_seconds

    def buffer(self, nbytes: int) -> torch.Tensor:
        if self.fail_buffer is not None and self.fail_buffer(self):
            raise RuntimeError("CUDA error: out of memory (pinned, stand-in)")
        self.buffers += 1
        return torch.full((nbytes,), 0xFF, dtype=torch.uint8)

    def copy(self, staging: torch.Tensor, host: torch.Tensor) -> FakeEvent:
        assert threading.get_ident() == self.training, "a copy was issued off the training thread"
        assert staging.dtype == torch.uint8 and host.dtype == torch.uint8 and host.numel() >= staging.numel()
        self.copies += 1
        if self.fail_copy is not None and self.fail_copy(self):
            raise RuntimeError("CUDA error: invalid resource handle (stand-in)")
        event = FakeEvent(self, staging, host)
        if self.mode == "eager":
            event.run()
        return event


class SlowSink(tensorquill.MemorySink):
    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    def write(self, records):
        time.sleep(self.seconds)
        super().write(records)


@pytest.fixture
def use_fake(monkeypatch):
    """Install a `FakeCopies` as the session's copier; returns a function that makes one."""
    made: list[FakeCopies] = []

    def install(copies: FakeCopies | None = None) -> FakeCopies:
        copies = copies or FakeCopies()
        made.append(copies)
        monkeypatch.setattr(capture, "_pinned_backend", lambda device: copies)
        return copies

    return install


def _out_of_range(state):
    def indices(output):
        _, chosen = output
        return torch.full_like(chosen, 99) if state["bad"] else chosen
    return indices


def _train(*, pinned: bool, scenario: str = "full", every: int = 4, steps: int = 11, sink=None, compiled=False,
           **init_kwargs):
    """A seeded run of `scenario`; returns the cleaned records, TensorQuill's warnings and the session."""
    torch.manual_seed(0)
    model = ToyMoE()
    sink = sink if sink is not None else tensorquill.MemorySink()
    state = {"bad": False}
    kwargs: dict = {"routers": {"router": 4}, "capture": "full"}
    params = list(model.parameters())
    opt2 = None
    scaler = None
    if scenario == "routers":
        kwargs["capture"] = "routers"
    elif scenario == "grads_n":
        kwargs.update(grads=2, weights=3)
    elif scenario == "updates":
        kwargs.update(updates=True)
    elif scenario == "follower":
        kwargs.update(updates=True, weights=1)
        params = list(model.inp.parameters()) + list(model.router.parameters())
        opt2 = torch.optim.Adam(list(model.experts.parameters()) + list(model.out.parameters()), lr=0.01)
    elif scenario == "scaler":
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
        kwargs.update(scaler=scaler)
    elif scenario == "refusal":
        kwargs["routers"] = {"router": tensorquill.Router(n_experts=4, indices=_out_of_range(state))}
    elif scenario == "unfrozen":
        for p in model.out.parameters():
            p.requires_grad_(False)
    elif scenario == "plain":
        pass
    else:
        assert scenario == "full", scenario
    kwargs.update(init_kwargs)
    opt_cls = torch.optim.Adam if scenario in ("updates", "follower", "checkpoint") else torch.optim.SGD
    opt = opt_cls(params, lr=0.01)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, every=every, pinned_flush=pinned, **kwargs)
        forward = torch.compile(model, backend="eager") if compiled else model
        for i in range(steps):
            if scenario == "unfrozen" and i == 5:
                for p in model.out.parameters():
                    p.requires_grad_(True)  # the optimizer held them all along
            state["bad"] = scenario == "refusal" and i == 6
            opt.zero_grad()
            if opt2 is not None:
                opt2.zero_grad()
            loss = forward(torch.randn(16, 8)).pow(2).mean()
            if scenario == "scaler" and i == 5:
                loss = loss * float("inf")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if scenario == "updates":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # the clip rows of the ring
                opt.step()
                if opt2 is not None:
                    opt2.step()  # after the lead ended the step: a follower
            session.log({"train/loss": loss.detach(), "train/lr": 0.01})
            if i % 3 == 1:
                session.log({"eval/loss": loss.detach() * 0.5}, step=i)  # a scalar-only record with a tensor
            if i == 3:  # a save right after the step that filled the ring (held back)
                session.checkpoint({"model": model.state_dict(), "optimizer": opt.state_dict()})
        session.close()
    told = sorted(str(w.message) for w in said if "TensorQuill" in str(w.message))
    return [_clean(r) for r in getattr(sink, "records", [])], told, session


SCENARIOS = ["full", "routers", "grads_n", "updates", "follower", "scaler", "refusal", "unfrozen"]


@pytest.mark.parametrize("mode", ["late", "eager"])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_records_are_the_same_as_the_blocking_paths_value_for_value(use_fake, scenario, mode):
    expected, expected_said, blocking = _train(pinned=False, scenario=scenario)
    copies = use_fake(FakeCopies(mode))
    got, said, session = _train(pinned=True, scenario=scenario)
    assert blocking.writer_error is None and session.writer_error is None
    assert session.pinned_flush, "the pinned path was not used"
    assert copies.copies > 0 and (mode == "eager" or copies.waits == copies.copies)
    assert got == expected
    assert said == expected_said


@pytest.mark.parametrize("every", [1, 3, 50])
def test_every_ring_size_and_a_flush_before_the_ring_fills(use_fake, every):
    expected, _, _ = _train(pinned=False, every=every, steps=7)
    use_fake(FakeCopies("late"))
    got, _, _ = _train(pinned=True, every=every, steps=7)
    assert got == expected


def test_a_copy_that_lands_late_holds_the_flushs_values_not_the_next_steps(use_fake):
    """The writer waits 30 ms for each copy, and the copy lands only then: by that time training has
    written the next steps into the ring. A copy that read the ring itself, not a snapshot of it, would
    write those. The update and clip rows are parts read straight from the ring (`updates=True`)."""
    expected, _, _ = _train(pinned=False, scenario="updates", every=2, steps=9)
    use_fake(FakeCopies("late", wait_seconds=0.03))
    got, _, session = _train(pinned=True, scenario="updates", every=2, steps=9)
    assert session.writer_error is None and got == expected


def test_a_writer_behind_takes_more_buffers_and_never_reuses_one_it_has_not_read(use_fake):
    """A sink that sleeps, every=1: the writer falls several blocks behind. The copy lands as it is issued
    (`eager`), so a buffer reused before the writer took its block out would overwrite that block."""
    expected, _, _ = _train(pinned=False, every=1, steps=12)
    copies = use_fake(FakeCopies("eager"))
    got, _, session = _train(pinned=True, every=1, steps=12, sink=SlowSink(0.05))
    assert session.writer_error is None
    assert got == expected
    assert copies.buffers > 2  # two buffers of its own, then more while the writer was behind


def test_the_training_thread_reads_no_tensor_value_on_the_pinned_path(use_fake):
    """Every read of a tensor's value (`item`, `tolist`, `cpu`, `numpy`, `float`, `int`, `bool`) is
    recorded with its thread. On the blocking path the training thread reads at each flush; on the pinned
    path it reads none, `close()` included: the writer reads them all."""
    names = ("item", "tolist", "cpu", "numpy", "__float__", "__int__", "__bool__")
    reads: list[tuple[int, str]] = []
    saved = {name: getattr(torch.Tensor, name) for name in names}

    def recorded(name):
        real = saved[name]

        def read(self, *args, **kwargs):
            reads.append((threading.get_ident(), name))
            return real(self, *args, **kwargs)
        return read

    def run(pinned: bool) -> list[str]:
        torch.manual_seed(0)
        model = ToyMoE()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        session = tensorquill.init(model, sink=tensorquill.MemorySink(), every=3, routers={"router": 4},
                                   capture="full", pinned_flush=pinned)
        reads.clear()
        for name in names:
            setattr(torch.Tensor, name, recorded(name))
        try:
            for i in range(8):
                opt.zero_grad()
                loss = model(torch.randn(16, 8)).pow(2).mean()
                loss.backward()
                opt.step()
                session.log({"train/loss": loss.detach()})
                if i == 4:
                    session.log({"eval/loss": loss.detach()}, step=i)
            session.close()
        finally:
            for name, real in saved.items():
                setattr(torch.Tensor, name, real)
        here = threading.get_ident()
        return [name for thread, name in reads if thread == here]

    assert run(pinned=False)  # the recorder sees the blocking path's reads
    use_fake(FakeCopies("late"))
    assert run(pinned=True) == []


def test_the_default_is_the_blocking_path_and_no_copier_is_made(monkeypatch):
    def refuse(device):
        raise AssertionError("a copier was made with the switch off")

    monkeypatch.setattr(capture, "_pinned_backend", refuse)
    _, said, session = _train(pinned=False)
    assert not session.pinned_flush
    assert not [s for s in said if "pinned" in s]


def test_the_switch_is_a_bool():
    with pytest.raises(ValueError, match="pinned_flush"):
        tensorquill.init(nn.Linear(2, 2), sink=tensorquill.MemorySink(), pinned_flush="yes")


def test_on_a_device_without_cuda_it_says_so_once_and_writes_what_the_blocking_path_writes():
    expected, expected_said, _ = _train(pinned=False)
    got, said, session = _train(pinned=True)  # the real copier: this Mac's session is on the CPU
    assert got == expected
    assert not session.pinned_flush
    extra = [s for s in said if s not in expected_said]
    assert len(extra) == 1 and "pinned_flush" in extra[0] and "CUDA" in extra[0]


@pytest.mark.parametrize("where", ["buffer", "copy", "copy_later", "wait"])
def test_a_failure_falls_back_to_the_blocking_transfer_and_says_so_once(use_fake, where):
    expected, expected_said, _ = _train(pinned=False, steps=14)
    calls = {"n": 0}

    def on_second(_copies):
        calls["n"] += 1
        return calls["n"] >= 2

    fake = {
        "buffer": FakeCopies("late", fail_buffer=lambda c: True),
        "copy": FakeCopies("late", fail_copy=lambda c: True),
        "copy_later": FakeCopies("late", fail_copy=on_second),
        "wait": FakeCopies("late", fail_wait=on_second),
    }[where]
    copies = use_fake(fake)
    got, said, session = _train(pinned=True, steps=14)
    assert session.writer_error is None
    assert got == expected  # nothing lost: the failed flush itself is read with a blocking copy
    extra = [s for s in said if s not in expected_said]
    assert len(extra) == 1 and "pinned" in extra[0], extra
    assert not session.pinned_flush
    if where == "copy_later":
        assert copies.copies == 2  # none after the failure


def test_close_waits_for_the_last_copies_and_leaves_the_end_marker(use_fake, tmp_path):
    path = tmp_path / "run.jsonl"
    use_fake(FakeCopies("late", wait_seconds=0.2))
    _, _, session = _train(pinned=True, steps=10, sink=tensorquill.JsonlSink(path))
    assert session.writer_error is None
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["step"] for r in lines if "record" not in r] == list(range(10))  # step records
    assert [r["step"] for r in lines if "record" in r] == [1, 4, 7]  # and the scalar-only ones
    assert (tmp_path / "run.end.json").exists()


def test_a_stop_writes_what_was_queued_before_it(use_fake):
    def run(pinned: bool):
        torch.manual_seed(0)
        model = ToyMoE()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sink = tensorquill.MemorySink()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = tensorquill.init(model, sink=sink, every=3, routers={"router": 4}, pinned_flush=pinned)
            for i in range(8):
                opt.zero_grad()
                model(torch.randn(16, 8)).pow(2).mean().backward()
                opt.step()
                if i == 6:
                    session._stop(RuntimeError("stopped by the test"), reason="flush_error")
            session.close()
        return [_clean(r) for r in sink.records]

    expected = run(False)
    use_fake(FakeCopies("late", wait_seconds=0.05))
    got = run(True)
    assert [r["step"] for r in got] == [0, 1, 2, 3, 4, 5]  # the held step 5 went with step 6's end
    assert got == expected


def test_a_writer_a_full_queue_behind_stops_capture_at_the_same_step_as_on_the_blocking_path(use_fake, monkeypatch):
    """The queue holds four blocks (`QUEUE_STEPS=4`, every=1) and the sink is stuck until training ends: the
    pinned path queues one item a block, as the blocking path does, so both stop at the same step."""
    monkeypatch.setattr(capture, "QUEUE_STEPS", 4)

    class Stuck(tensorquill.MemorySink):
        def __init__(self):
            super().__init__()
            self.go = threading.Event()

        def write(self, records):
            self.go.wait(10)
            super().write(records)

    def run(pinned: bool):
        torch.manual_seed(0)
        model = ToyMoE()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sink = Stuck()
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter("always")
            session = tensorquill.init(model, sink=sink, every=1, routers={"router": 4}, pinned_flush=pinned)
            for _ in range(10):
                opt.zero_grad()
                model(torch.randn(16, 8)).pow(2).mean().backward()
                opt.step()
            sink.go.set()
            session.close()
        assert session._stopped and "behind training" in str(session.writer_error)
        return [_clean(r) for r in sink.records], sorted(str(w.message) for w in said if "TensorQuill" in str(w.message))

    expected = run(False)
    use_fake(FakeCopies("late"))
    assert run(True) == expected
    assert [r["step"] for r in expected[0]] == [0, 1, 2, 3, 4]


def test_a_sink_that_fails_stops_capture_as_on_the_blocking_path(use_fake):
    class Failing(tensorquill.MemorySink):
        def write(self, records):
            if self.records:
                raise OSError("disk full")
            super().write(records)

    expected, expected_said, blocking = _train(pinned=False, sink=Failing(), steps=14)
    use_fake(FakeCopies("late"))
    got, said, session = _train(pinned=True, sink=Failing(), steps=14)
    assert "disk full" in str(session.writer_error) and "disk full" in str(blocking.writer_error)
    assert got == expected
    assert said == expected_said


def test_in_a_forked_process_no_copy_is_issued(use_fake):
    copies = use_fake(FakeCopies("late"))
    torch.manual_seed(0)
    model = ToyMoE()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), every=2, routers={"router": 4},
                               pinned_flush=True)
    real_pid = session._pid
    session._pid = real_pid + 1  # as in a child forked from the session's process
    for _ in range(4):
        opt.zero_grad()
        model(torch.randn(16, 8)).pow(2).mean().backward()
        opt.step()
    assert copies.copies == 0
    session._pid = real_pid
    session.close()


def test_under_torch_compile_the_records_are_the_same(use_fake):
    expected, _, _ = _train(pinned=False, scenario="plain", compiled=True, steps=6)
    use_fake(FakeCopies("late"))
    got, _, session = _train(pinned=True, scenario="plain", compiled=True, steps=6)
    assert session.pinned_flush and got == expected


# ---- the CUDA copier, against a stand-in for torch.cuda ------------------------------------------------


class _Log(list):
    pass


class _Stream:
    def __init__(self, log: _Log, name: str) -> None:
        self.log, self.name = log, name

    def wait_event(self, event) -> None:
        self.log.append(("wait_event", self.name, event.name))


class _Event:
    count = 0

    def __init__(self, log: _Log, blocking: bool = False) -> None:
        _Event.count += 1
        self.log, self.blocking, self.name = log, blocking, f"event{_Event.count}"

    def record(self, stream) -> None:
        self.log.append(("record", self.name, stream.name, self.blocking))

    def synchronize(self) -> None:
        self.log.append(("synchronize", self.name))


class _FakeCuda:
    """Only what the copier may call: no `synchronize`, no `current_device`, nothing else."""

    def __init__(self) -> None:
        self.log = _Log()
        self.active: list[str] = []

    def Stream(self, device=None):  # noqa: N802
        self.log.append(("Stream", str(device)))
        return _Stream(self.log, "copy")

    def Event(self, blocking: bool = False):  # noqa: N802
        return _Event(self.log, blocking=blocking)

    def current_stream(self, device=None):
        return _Stream(self.log, "training")

    @contextlib.contextmanager
    def stream(self, stream):
        self.log.append(("enter", stream.name))
        self.active.append(stream.name)
        try:
            yield
        finally:
            self.active.pop()
            self.log.append(("exit", stream.name))


def test_the_cuda_copier_orders_the_copy_after_the_training_stream_and_waits_on_the_writer(monkeypatch):
    cuda = _FakeCuda()
    copier = capture._CudaCopies(torch.device("cpu"), cuda=cuda)
    copies: list[tuple[str | None, bool]] = []
    kept: list[tuple[str, str]] = []

    class Copy(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.Tensor.copy_:
                copies.append((cuda.active[-1] if cuda.active else None, kwargs.get("non_blocking", False)))
                cuda.log.append(("copy",))
            return func(*args, **kwargs)

    monkeypatch.setattr(capture, "_record_stream", lambda tensor, stream: kept.append(("record_stream", stream.name)))
    staging = torch.arange(32, dtype=torch.uint8)
    host = torch.zeros(64, dtype=torch.uint8)
    with Copy():
        done = copier.copy(staging, host)
    names = [entry[0] for entry in cuda.log]
    assert names == ["Stream", "record", "wait_event", "enter", "copy", "exit", "record"]
    ready = cuda.log[1]
    assert ready[2] == "training"  # recorded on the stream the ring was written on
    assert cuda.log[2] == ("wait_event", "copy", ready[1])  # the copy stream waits for it
    assert copies == [("copy", True)]  # one copy, on the copy stream, non-blocking
    assert cuda.log[-1][2] == "copy" and cuda.log[-1][3] is True  # `done`, a blocking-sync event
    assert kept == [("record_stream", "copy")]
    assert torch.equal(host[:32], staging)
    done.synchronize()
    assert cuda.log[-1][0] == "synchronize"


def test_the_cuda_copier_asks_for_pinned_memory(monkeypatch):
    asked = []
    real = torch.empty

    def empty(*args, **kwargs):
        asked.append(kwargs.get("pin_memory"))
        kwargs.pop("pin_memory", None)
        return real(*args, **kwargs)

    copier = capture._CudaCopies(torch.device("cpu"), cuda=_FakeCuda())
    monkeypatch.setattr(torch, "empty", empty)
    buffer = copier.buffer(100)
    assert asked == [True] and buffer.dtype == torch.uint8 and buffer.numel() >= 100


def test_the_backend_is_refused_off_cuda():
    with pytest.raises(RuntimeError, match="CUDA"):
        capture._pinned_backend(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="CUDA"):
        capture._pinned_backend(torch.device("meta"))


def test_pid_is_the_sessions_own():
    session = tensorquill.init(nn.Linear(2, 2), sink=tensorquill.MemorySink())
    assert session._pid == os.getpid()
    session.close()

