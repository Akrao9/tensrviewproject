"""Lane AL-5, change 1: a value logged late inside a compiled function, then a flush for a newly
trainable parameter (`WAVE_AL_CONTRACT.md` section 4.8, row L1).

Cases `scalar_compiled_late_before_unfreeze` (the fix), `scalar_eager_late_before_unfreeze` (the
reference, kept) and `scalar_compiled_late_at_a_stop` (a stated limit, pinned).

The rule the fix makes true: every path that hands the newest step to the writer drains
`_late_traced` first, so a value in that buffer always belongs to a step not yet handed over. The
drain is at the start of `Session._flush`, after its stop check, so `_take_unfrozen` (which writes
the ring from the optimizer's pre-step hook before the step's gradients are read) drains too.
"""

from __future__ import annotations

import time
import warnings

import pytest
import torch
from torch import nn

import tensorquill

LEFT_OUT = "left out the scalar"
UNFROZEN = "frozen when the session started"
STOPPED = "the steps already handed to the writer are written, no later step is"


def _late_before_unfreeze(compiled: bool, every: int, unfreeze: bool = True):
    """The contract's scenario: the first layer frozen at `init`; steps 0 and 1, each followed by an
    evaluation under "eval" that logs `eval/x`; the layer made trainable before step 2, whose
    optimizer pre-step hook flushes the ring (`_take_unfrozen`) before step 2 ends.

    Returns the records, the value logged after step 1, the warnings and the compile counter (or
    None). Raises `RuntimeError` when the scenario is not reached, so a test cannot pass for a wrong
    reason."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    model[0].requires_grad_(False)
    sink = tensorquill.MemorySink()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    session = None

    def evaluate(x):
        with torch.no_grad():
            value = model(x).pow(2).mean()
        session.log({"eval/x": value})
        return value

    counter = None
    if compiled:
        from torch._dynamo.testing import CompileCounter

        torch._dynamo.reset()
        counter = CompileCounter()
        evaluate = torch.compile(evaluate, backend=counter, fullgraph=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, every=every, weights=1, updates=True, run_id="al5_late",
                                   heartbeat_seconds=None, stack_seconds=None)
        try:
            for step in range(3):
                if step == 2 and unfreeze:
                    model[0].requires_grad_(True)  # trainable now: this step's pre-step hook flushes the ring
                model(torch.randn(8, 4)).pow(2).mean().backward()
                opt.step()  # the TensorQuill step ends here (auto_step)
                opt.zero_grad()
                if step in (0, 1):
                    with session.phase("eval"):
                        logged = evaluate(torch.full((8, 4), float(step + 1))).item()
        finally:
            session.close()
            if compiled:
                torch._dynamo.reset()
    messages = [str(w.message) for w in caught]
    if compiled and counter.frame_count != 1:
        raise RuntimeError(f"the evaluation compiled {counter.frame_count} frames, not one")
    if unfreeze and sum(UNFROZEN in m for m in messages) != 1:
        raise RuntimeError(f"no flush for a newly trainable parameter: {messages}")
    return sink.records, logged, messages, counter


def _step_scalars(records, step):
    return next(r for r in records if r["step"] == step and r.get("record") != "scalars").get("scalars", {})


@pytest.mark.parametrize("every", [1, 4])
def test_a_compiled_late_value_reaches_its_step_before_an_unfreeze_flush(every):
    """`scalar_compiled_late_before_unfreeze`: on step 1's own record, as eagerly, with no left-out
    warning, the unfreeze said once, and the evaluation compiled once for both calls."""
    records, logged, messages, counter = _late_before_unfreeze(True, every)
    assert not [m for m in messages if LEFT_OUT in m]
    assert _step_scalars(records, 1).get("eval/x") == pytest.approx(logged)
    assert sum(UNFROZEN in m for m in messages) == 1
    assert counter.frame_count == 1
    # The value after step 0 is on step 0, and no record but step 1's holds the second value.
    assert "eval/x" in _step_scalars(records, 0)
    assert "eval/x" not in _step_scalars(records, 2)
    assert [r["step"] for r in records] == [0, 1, 2]


@pytest.mark.parametrize("every", [1, 4])
def test_the_eager_twin_still_reaches_its_step(every):
    """`scalar_eager_late_before_unfreeze`, the reference: unchanged by the fix."""
    records, logged, messages, _ = _late_before_unfreeze(False, every)
    assert not [m for m in messages if LEFT_OUT in m]
    assert _step_scalars(records, 1).get("eval/x") == pytest.approx(logged)
    assert sum(UNFROZEN in m for m in messages) == 1


@pytest.mark.parametrize("every", [1, 4])
def test_compiled_and_eager_write_the_same_scalars(every):
    """Compiling the evaluation changes nothing that is recorded: the eval value of each step is the
    same, compiled or not (the seed and inputs are fixed)."""
    eager, _, _, _ = _late_before_unfreeze(False, every)
    other, _, _, _ = _late_before_unfreeze(True, every)
    assert [r.get("scalars", {}).get("eval/x") for r in other] == pytest.approx(
        [r.get("scalars", {}).get("eval/x") for r in eager])


def test_the_drain_comes_after_the_stop_check_and_before_the_flush_writes():
    """The drain is in `_flush` itself, so any later caller that writes the newest step drains too:
    a compiled late value, then a direct `_flush(hold_newest=False)` (as `_take_unfrozen` calls it)."""
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=4, run_id="al5_flush", heartbeat_seconds=None,
                               stack_seconds=None)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    model(torch.randn(8, 4)).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()
    session._late_traced["eval/x"] = 0.5  # what a traced `_log_late` leaves
    session._flush(hold_newest=False)
    assert session._late_traced == {}
    session.close()
    assert sink.records[0]["scalars"]["eval/x"] == 0.5


@pytest.mark.parametrize("every", [1, 4])
def test_a_compiled_late_value_is_lost_with_its_step_at_a_stop(every):
    """`scalar_compiled_late_at_a_stop`, a stated limit: the compiled evaluation logs `eval/x` after
    step 1, then capture stops before step 1 is handed to the writer (the writer's sink failed, which
    the next step's end turns into a stop). Step 1 and its value are lost together; the stop's own
    warning is the one said, and no left-out warning."""
    from torch._dynamo.testing import CompileCounter

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    sink = tensorquill.MemorySink()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    session = None

    def evaluate(x):
        with torch.no_grad():
            value = model(x).pow(2).mean()
        session.log({"eval/x": value})
        return value

    torch._dynamo.reset()
    counter = CompileCounter()
    evaluate = torch.compile(evaluate, backend=counter, fullgraph=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, every=every, run_id="al5_stop", heartbeat_seconds=None,
                                   stack_seconds=None)
        try:
            for step in range(3):
                if step == 2:
                    # Step 0 is handed over and written at every=1 (step 1 ended it); then the sink fails,
                    # as the writer thread records it, before step 1 is handed over.
                    deadline = time.monotonic() + 10
                    while every == 1 and not sink.records and time.monotonic() < deadline:
                        time.sleep(0.01)
                    session._error = OSError("disk gone")
                model(torch.randn(8, 4)).pow(2).mean().backward()
                opt.step()
                opt.zero_grad()
                if step in (0, 1):
                    with session.phase("eval"):
                        evaluate(torch.full((8, 4), float(step + 1)))
        finally:
            session.close()
            torch._dynamo.reset()
    messages = [str(w.message) for w in caught]
    assert counter.frame_count == 1
    assert session._stopped
    assert [r["step"] for r in sink.records] == ([0] if every == 1 else [])
    assert not [r for r in sink.records if r["step"] == 1]
    assert not [m for m in messages if LEFT_OUT in m]
    stops = [m for m in messages if STOPPED in m]
    assert len(stops) == 1
