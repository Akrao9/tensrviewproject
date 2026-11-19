"""Lane V1 (wave V): the capture rows of `OPEN_BUGS.md`.

Each test below fails on `main` at 66b52d7 and passes with the lane's change in `capture.py`,
`datawait.py` or `flight.py`, except the guards named at the end:

- C12: an optimizer that held none of the model's parameters at its first step, and is later given
  them with `add_param_group`, ends steps again; a parameter unfrozen after `init` is named in a
  warning instead of being left out in silence.
- C13: the cache of on-device index tensors for partial gradient sets is bounded.
- C15: a replica-spread table of the wrong shape is refused, not broadcast to other parameters.
- C20 (SDK-12): time under a marked pause is out of `step_time`, and `time_data` fetches count only
  where the `DataLoader` timer's would (training phase, the session's thread, not nested).
- C23: a session's heartbeat carries that session's attempt, even when another session took an
  attempt in between.
- C24: a model that is one leaf module with parameters is observed, under its class name.
- SDK-15: a trainable parameter with no elements is "not known", never "unchanged".
- SDK-26: `init`'s docstring for `auto_step` agrees with `scaler`.

- C14 (contract 2): `init` states what `weights_unchanged` asserts (an equal finite float64 norm:
  necessary, not sufficient), and a parameter whose norm is not finite is not listed.

SDK-06 is left open by the author's decision; its reproduction is in the lane's hand-over note. The C24 end-marker test
passes on `main` too: that half was already fixed. Other guards also pass on `main`: a model with
children adds no root layer, a counter set by hand still takes the newest attempt, a phase left
behind at a step counts as training time, and the C14 rules the capture already kept (no list at the
first check, an empty list only after a comparison, a changed or unread parameter not listed, rows
swapped under one norm listed as the stated limit). No server binary.
"""

from __future__ import annotations

import math
import os
import threading
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture
from tensorquill.flight import Heartbeat
from tensorquill.sink import JsonlSink, read_end_marker


def _steps(records: list[dict]) -> list[dict]:
    return [r for r in records if "step" in r]


# --- C12: optimizer membership changes after init --------------------------------------------------


def test_an_optimizer_given_the_model_with_add_param_group_ends_steps():
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    other = nn.Parameter(torch.ones(3))  # not the model's: the optimizer starts out foreign to it
    opt = torch.optim.SGD([other], lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, weights=1)
    other.grad = torch.ones(3)
    opt.step()  # its first step: it holds none of the model's parameters
    opt.add_param_group({"params": list(model.parameters())})
    before = model.weight.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        model(torch.ones(4, 2)).sum().backward()
        opt.step()
    session.close()
    assert not torch.equal(model.weight.detach(), before)
    assert [r["step"] for r in _steps(sink.records)] == [0, 1, 2]


def test_a_parameter_unfrozen_after_init_is_named_in_a_warning():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=2, weights=1)
    model[0].weight.requires_grad_(True)  # a transfer-learning schedule unfreezes the backbone
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(4):
            opt.zero_grad()
            model(torch.ones(4, 2)).sum().backward()
            opt.step()
    session.close()
    said = [str(w.message) for w in caught if "0.weight" in str(w.message)]
    assert len(said) == 1, [str(w.message) for w in caught]
    assert "after tensorquill.init" in said[0]


# --- C13: the on-device index cache ------------------------------------------------------------------


class Eight(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ps = nn.ParameterList([nn.Parameter(torch.full((3,), float(i + 1))) for i in range(8)])


def test_the_index_cache_for_partial_gradient_sets_is_bounded_and_norms_stay_right():
    model = Eight()
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), every=1000, auto_step=False, weights=False)
    for mask in range(1, 255):  # every nonempty proper subset of eight parameters
        for i, p in enumerate(model.ps):
            p.grad = torch.full((3,), float(i + 1)) if mask >> i & 1 else None
        session.record_grads()
        expected = [math_norm(i) if mask >> i & 1 else 0.0 for i in range(8)]
        got = session._grad[:, 0].tolist()
        assert got == pytest.approx(expected), mask
        assert session._grad[:, 1].tolist() == [1.0 if mask >> i & 1 else 0.0 for i in range(8)]
    retained = sum(t.numel() for t in session._grad_idx_cache.values())
    session.close()
    # 254 index tensors of 1,016 entries on main; a bounded few now.
    assert len(session._grad_idx_cache) <= 32
    assert retained <= 32 * 7


def math_norm(i: int) -> float:
    return float(torch.linalg.vector_norm(torch.full((3,), float(i + 1))))


# --- C15: a replica-spread table of the wrong shape --------------------------------------------------


@pytest.mark.parametrize("table", [
    torch.tensor([[0.1, 0.2, 0.3]]),  # one row for two parameters: broadcast on main
    torch.zeros(3, 3),  # three rows for two: raised into training on main
    torch.zeros(2, 4),  # four columns
    torch.zeros(2),  # one dimension
])
def test_a_spread_table_of_the_wrong_shape_is_refused_and_keeps_what_was_recorded(table):
    model = nn.Linear(2, 1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, auto_step=False, grads=False, weights=False)
    good = torch.tensor([[1e-3, 2e-3, 3e-3], [4e-3, 5e-3, 6e-3]], dtype=torch.float64)
    session.record_replica_spread(good)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.record_replica_spread(table)  # never raises
    session.step()
    session.close()
    assert any("replica" in str(w.message) and "shape" in str(w.message) for w in caught)
    (record,) = _steps(sink.records)
    assert record["replica_spread"] == {"weight": pytest.approx([1e-3, 2e-3, 3e-3]),
                                        "bias": pytest.approx([4e-3, 5e-3, 6e-3])}


def test_a_wrong_spread_table_alone_records_no_spread():
    sink = tensorquill.MemorySink()
    session = tensorquill.init(nn.Linear(2, 1), sink=sink, every=1, auto_step=False, grads=False, weights=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.record_replica_spread(torch.tensor([[0.1, 0.2, 0.3]]))
    session.step()
    session.close()
    (record,) = _steps(sink.records)
    assert "replica_spread" not in record


# --- C20 / SDK-12: pauses in step_time and data_wait ------------------------------------------------


class Clock:
    """`time.perf_counter`, moved only by the test."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(time, "perf_counter", fake)
    return fake


def _batches(clock: Clock, n: int, cost: float):
    for _ in range(n):
        clock.now += cost  # the fetch
        yield torch.ones(2, 2)


def _train_step(session, model, opt, clock, batch):
    clock.now += 1.0  # forward, backward and the optimizer
    opt.zero_grad()
    model(batch).sum().backward()
    opt.step()


def test_a_marked_pause_is_not_in_step_time(clock):
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, grads=False, weights=False)
    loader = iter(session.time_data(_batches(clock, 3, 1.0)))
    for step in range(3):
        if step == 2:
            with session.phase("eval"):
                clock.now += 99.0  # validation, a checkpoint save
        _train_step(session, model, opt, clock, next(loader))
    session.close()
    rows = _steps(sink.records)
    assert [r["data_wait"] for r in rows] == [1.0, 1.0, 1.0]
    assert [r.get("step_time") for r in rows] == [None, 2.0, 2.0]


def test_time_data_fetches_under_an_eval_phase_are_not_data_wait(clock):
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, grads=False, weights=False)
    loader = iter(session.time_data(_batches(clock, 3, 1.0)))
    evaluation = session.time_data(_batches(clock, 3, 33.0))
    for step in range(3):
        if step == 2:
            with session.phase("eval"), torch.no_grad():
                for batch in evaluation:
                    model(batch)
        _train_step(session, model, opt, clock, next(loader))
    session.close()
    rows = _steps(sink.records)
    assert [r["data_wait"] for r in rows] == [1.0, 1.0, 1.0]
    assert rows[2]["step_time"] == 2.0


def test_a_nested_time_data_counts_a_fetch_once(clock):
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, grads=False, weights=False)
    loader = iter(session.time_data(session.time_data(_batches(clock, 2, 1.0))))
    for _ in range(2):
        _train_step(session, model, opt, clock, next(loader))
    session.close()
    assert [r["data_wait"] for r in _steps(sink.records)] == [1.0, 1.0]


def test_time_data_on_another_thread_is_not_data_wait(clock):
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, grads=False, weights=False)
    loader = iter(session.time_data(_batches(clock, 2, 1.0)))
    prefetch = iter(session.time_data(_batches(clock, 1, 50.0)))
    worker = threading.Thread(target=lambda: next(prefetch))  # a prefetch thread, as the global timer leaves out
    _train_step(session, model, opt, clock, next(loader))
    worker.start()
    worker.join()
    _train_step(session, model, opt, clock, next(loader))
    session.close()
    assert [r["data_wait"] for r in _steps(sink.records)] == [1.0, 1.0]


def test_an_unmarked_evaluation_loop_is_not_data_wait_but_a_training_fetch_in_eval_mode_is(clock):
    from torch.utils.data import DataLoader

    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, grads=False, weights=False, data_wait=True)

    class Slow(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            clock.now += 10.0
            return torch.ones(2)

    train = iter(session.time_data(_batches(clock, 3, 1.0)))
    _train_step(session, model, opt, clock, next(train))
    model.eval()  # a hand-written evaluation, no session.phase: the DataLoader timer and time_data
    with torch.no_grad():
        for batch in DataLoader(Slow(), batch_size=2):
            model(batch)
        for batch in session.time_data(_batches(clock, 2, 10.0)):
            model(batch)
    _train_step(session, model, opt, clock, next(train))  # still in eval mode, gradients on: training
    model.train()
    _train_step(session, model, opt, clock, next(train))
    session.close()
    assert [r["data_wait"] for r in _steps(sink.records)] == [1.0, 1.0, 1.0]


def test_a_phase_left_behind_at_a_step_is_training_time(clock):
    # A callback that set "eval" and missed its end: the step that ends it ran training all along.
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, grads=False, weights=False)
    loader = iter(session.time_data(_batches(clock, 3, 1.0)))
    _train_step(session, model, opt, clock, next(loader))
    session.set_phase("eval")
    clock.now += 5.0
    _train_step(session, model, opt, clock, torch.ones(2, 2))
    _train_step(session, model, opt, clock, next(loader))
    session.close()
    rows = _steps(sink.records)
    assert [r.get("step_time") for r in rows] == [None, 6.0, 2.0]


# --- C23: two sessions starting at once ---------------------------------------------------------------


def _beat() -> Heartbeat:
    """A `Heartbeat` with its attempt logic and no thread or connection (no pyarrow needed)."""
    beat = Heartbeat.__new__(Heartbeat)
    beat._counter = None
    beat.attempt = None
    beat._changed = threading.Event()
    beat.phase = "train"
    beat._pid = os.getpid()
    return beat


class BeatSink(tensorquill.MemorySink):
    """A sink whose heartbeat, when the session reaches for it, first lets `meanwhile` run."""

    def __init__(self, meanwhile=None) -> None:
        super().__init__()
        self._beat = _beat()
        self._meanwhile = meanwhile

    @property
    def heartbeat(self) -> Heartbeat:
        if self._meanwhile is not None:
            meanwhile, self._meanwhile = self._meanwhile, None
            meanwhile()
        return self._beat


def test_a_heartbeat_carries_its_own_session_attempt_when_another_was_taken_in_between():
    sink = BeatSink(meanwhile=capture._next_attempt)  # another session takes an attempt here
    session = tensorquill.init(nn.Linear(2, 1), sink=sink, every=1)
    assert sink._beat.attempt == session.attempt
    session.close()


def test_two_sessions_started_on_two_threads_each_beat_with_their_own_attempt():
    b_started = threading.Event()
    sessions: dict[str, object] = {}

    def start_b() -> None:
        sinks["b"] = BeatSink()
        sessions["b"] = tensorquill.init(nn.Linear(2, 1), run_id="run-b", sink=sinks["b"], every=1)
        b_started.set()

    def a_waits_for_b() -> None:
        other = threading.Thread(target=start_b)
        other.start()
        assert b_started.wait(30)
        other.join(30)

    sinks: dict[str, BeatSink] = {"a": BeatSink(meanwhile=a_waits_for_b)}
    sessions["a"] = tensorquill.init(nn.Linear(2, 1), run_id="run-a", sink=sinks["a"], every=1)
    try:
        assert sessions["a"].attempt != sessions["b"].attempt
        assert sinks["a"]._beat.attempt == sessions["a"].attempt
        assert sinks["b"]._beat.attempt == sessions["b"].attempt
    finally:
        for s in sessions.values():
            s.close()


def test_a_heartbeat_given_a_counter_by_hand_still_takes_the_newest_attempt():
    beat = _beat()
    attempt = capture._next_attempt()
    beat.counter = lambda: 0
    assert beat.attempt == attempt


# --- C24: what is observed, and what an end marker says -------------------------------------------


def test_a_model_that_is_one_leaf_module_is_observed_under_its_class_name():
    model = nn.Linear(2, 1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.zero_grad()
    model(torch.ones(4, 2)).sum().backward()
    opt.step()
    session.close()
    assert session.layer_names == ["Linear"]
    (record,) = _steps(sink.records)
    assert set(record["layers"]) == {"Linear"}


@pytest.mark.parametrize("layers", ["blocks", "all"])
def test_a_root_leaf_is_named_for_both_default_layer_choices(layers):
    session = tensorquill.init(nn.Linear(2, 1), sink=tensorquill.MemorySink(), every=1, layers=layers)
    session.close()
    assert session.layer_names == ["Linear"]


def test_a_compiled_root_leaf_gives_the_eager_statistics_without_recompiling():
    from torch._dynamo.testing import CompileCounter

    results, frames = [], None
    for compile_it in (False, True):
        torch._dynamo.reset()
        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        sink = tensorquill.MemorySink()
        session = tensorquill.init(model, sink=sink, every=1, auto_step=False)
        counter = CompileCounter()
        run = torch.compile(model, backend=counter) if compile_it else model
        for _ in range(3):
            run(torch.randn(8, 4)).sum().backward()
            session.step()
        session.close()
        results.append([r["layers"] for r in _steps(sink.records)])
        frames = counter.frame_count
    eager, compiled = results
    assert set(eager[0]) == {"Linear"}
    for e, c in zip(eager, compiled):
        assert c["Linear"] == pytest.approx(e["Linear"], rel=1e-5, abs=1e-7)
    assert frames == 1


def test_a_model_with_children_does_not_add_its_root():
    session = tensorquill.init(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1)), sink=tensorquill.MemorySink(),
                               every=1, layers="all")
    session.close()
    assert session.layer_names == ["0", "1"]


def test_an_end_marker_whose_records_file_is_gone_is_not_closed(tmp_path: Path):
    # Passes on main at 66b52d7 too: this half of C24 was fixed before this lane.
    path = tmp_path / "r.jsonl"
    session = tensorquill.init(nn.Linear(2, 1), sink=JsonlSink(path), every=1)
    session.close()
    assert read_end_marker(path)["state"] == "closed"
    path.unlink()
    assert read_end_marker(path)["state"] == "unreadable"


# --- SDK-15: a parameter with no elements ------------------------------------------------------------


class WithEmpty(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.empty = nn.Parameter(torch.zeros(0, 4))  # an empty local shard
        self.b = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


def test_an_empty_parameter_is_never_weights_unchanged():
    torch.manual_seed(0)
    model = WithEmpty()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=2, weights=1, grads=1)
    for _ in range(4):
        opt.zero_grad()
        model(torch.randn(2, 4)).pow(2).mean().backward()
        opt.step()
    session.close()
    checked = [r for r in _steps(sink.records) if "weights_unchanged" in r]
    assert len(checked) == 3  # the first check has nothing to compare with
    for record in checked:
        assert "empty" not in record["weights_unchanged"]
        assert record["weight_norms"]["empty"] == 0.0  # its norm is still its norm


# --- SDK-26: the auto_step docstring ------------------------------------------------------------------


def _arg(doc: str, name: str) -> str:
    lines = doc.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(f"{name}:"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith("            "))
    return " ".join(line.strip() for line in lines[start:end])


def test_init_docstring_auto_step_agrees_with_scaler():
    doc = tensorquill.init.__doc__
    auto_step = _arg(doc, "auto_step")
    assert "scaler=" in auto_step and "scaler.update()" in auto_step
    assert "Pass False and call `session.step()` yourself when using `torch.amp.GradScaler`" not in auto_step
    step_doc = " ".join(capture.Session.step.__doc__.split())
    assert "scaler=" in step_doc


# --- C14: what weights_unchanged asserts (contract 2) ------------------------------------------------


def _checks(model: nn.Module, change, steps: int = 3, **options) -> list[dict]:
    """A weight check at each step (`session.step()`), `change(step)` applied to the weights before it."""
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, auto_step=False, weights=1, grads=1, **options)
    for step in range(steps):
        with torch.no_grad():
            change(step)
        model.zero_grad()
        model(torch.ones(1, model.in_features)).sum().backward()
        session.step()
    session.close()
    return _steps(sink.records)


def test_swapped_rows_share_a_norm_and_are_listed_the_limit_the_docstring_states():
    # An equal float64 norm is necessary, not sufficient: no second witness is read (it would be a
    # second pass over every weight), so this changed weight is listed, and `init` says it can be.
    model = nn.Linear(2, 2, bias=False)
    rows = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[0.0, 1.0], [1.0, 0.0]])]
    records = _checks(model, lambda step: model.weight.copy_(rows[step % 2]), steps=2)
    assert records[1]["weights_unchanged"] == ["weight"]
    doc = " ".join(_arg(tensorquill.init.__doc__, "weights").split())
    assert "necessary for an unchanged weight, not sufficient" in doc
    assert "two different weights can share a norm" in doc


def test_the_first_check_writes_no_list_and_a_changed_weight_is_not_listed():
    model = nn.Linear(2, 1)
    records = _checks(model, lambda step: model.bias.add_(1.0))
    assert "weights_unchanged" not in records[0]  # nothing compared: no key, never an empty list
    assert records[1]["weights_unchanged"] == ["weight"]
    assert records[2]["weights_unchanged"] == ["weight"]


def test_a_check_that_shows_nothing_unchanged_writes_an_empty_list():
    model = nn.Linear(2, 1)
    records = _checks(model, lambda step: (model.weight.add_(1.0), model.bias.add_(1.0)))
    assert [r.get("weights_unchanged") for r in records] == [None, [], []]


def test_an_infinite_norm_at_both_checks_is_not_listed():
    model = nn.Linear(2, 1)
    model.weight.data.copy_(torch.tensor([[math.inf, 1.0]]))
    records = _checks(model, lambda step: model.weight[0, 1].add_(1.0))  # changed; the norm stays inf
    assert records[1]["weights_unchanged"] == ["bias"]
    assert records[2]["weights_unchanged"] == ["bias"]


class WithUnread(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)
        self.extra = nn.Parameter(torch.ones(4, device="meta"))  # never read
        self.in_features = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_a_parameter_that_is_not_read_is_never_listed():
    model = WithUnread()
    records = _checks(model, lambda step: None)
    assert [r.get("weights_unchanged") for r in records] == [None, ["lin.weight", "lin.bias"], ["lin.weight", "lin.bias"]]
