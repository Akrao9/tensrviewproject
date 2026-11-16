"""The compile check of `examples/trainer_check.py`, which crashed on its first GPU run."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from trainer_check import recompiled_frames, reset_compile_state  # noqa: E402


def metric(compile_id, forward=True, name="forward", line=10, filename="/x/model.py"):
    """What `get_compilation_metrics()` held under Inductor on torch 2.14.1: a backward graph has no code object."""
    if not forward:
        return SimpleNamespace(compile_id=compile_id, is_forward=False, co_name=None, co_filename=None, co_firstlineno=None)
    return SimpleNamespace(compile_id=compile_id, is_forward=True, co_name=name, co_filename=filename, co_firstlineno=line)


def test_backward_graphs_are_not_frames_and_do_not_look_like_a_recompile():
    # One frame compiled once, and several backward graphs without a code object (the crash).
    metrics = [metric("0/0"), metric("0/0", forward=False), metric("1/0", forward=False), metric("2/0", forward=False)]
    assert recompiled_frames(metrics) == (1, [])


def test_a_frame_compiled_twice_is_named():
    metrics = [metric("0/0"), metric("0/0", forward=False), metric("0/1"), metric("0/1", forward=False)]
    count, lines = recompiled_frames(metrics)
    assert count == 2
    assert lines == ["forward (model.py:10) compiled 2 times"]


def test_two_different_frames_are_not_a_recompile():
    assert recompiled_frames([metric("0/0", line=10), metric("1/0", line=99, name="resume")]) == (2, [])


def test_nothing_compiled_counts_zero():
    assert recompiled_frames([]) == (0, [])
    assert recompiled_frames([metric("0/0", forward=False)]) == (0, [])


def test_a_metric_without_the_forward_flag_still_counts():
    # Older torch versions have no `is_forward`.
    old = SimpleNamespace(co_name="forward", co_filename="/x/model.py", co_firstlineno=10)
    assert recompiled_frames([old, old]) == (2, ["forward (model.py:10) compiled 2 times"])


@pytest.mark.skipif(not hasattr(torch._dynamo.utils, "get_compilation_metrics"), reason="needs Dynamo's metrics")
def test_against_the_real_metrics_of_a_compiled_training_step():
    # Other tests in this process have compiled the same frames (conftest's `forward`), which
    # the script would count as recompiles unless it clears the metrics first.
    reset_compile_state()
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 1))
    compiled = torch.compile(model, backend="inductor")
    try:
        for _ in range(3):
            compiled(torch.randn(4, 8)).sum().backward()
    except Exception as error:  # no C++ compiler for Inductor on this machine
        pytest.skip(f"Inductor is not usable here: {type(error).__name__}")
    count, lines = recompiled_frames(torch._dynamo.utils.get_compilation_metrics())
    assert count >= 1 and lines == []


def test_reset_forgets_frames_compiled_earlier_in_the_process():
    compiled = torch.compile(torch.nn.Linear(4, 4), backend="aot_eager")
    compiled(torch.randn(2, 4))
    assert torch._dynamo.utils.get_compilation_metrics()
    reset_compile_state()
    assert torch._dynamo.utils.get_compilation_metrics() == []
