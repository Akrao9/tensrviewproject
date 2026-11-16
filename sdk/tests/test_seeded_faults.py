"""The faults the dense checks seed, and which of them a block summary can see.

The first A100 run of `trainer_check.py` scaled two ordinary channels of SmolLM2's block 16 by 100
and the activation detector found nothing: the block's output already held massive activations
(`abs_max` near 20,000, `std` near 32), so its summary moved 1.1 times. These tests build that
situation in miniature and run the real detector on it.
"""

import sys
from pathlib import Path

import pytest
import torch

from tensorquill.evidence import detect_activation_anomalies

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from dense_outlier_check import FAULTS, install_outlier, outlier_channels  # noqa: E402

HIDDEN = 960
MASSIVE = [5, 11]
ORDINARY = [7, 300]
BASELINE_STEPS, SEED_STEP = 60, 60


class Stub(torch.nn.Module):
    """Enough of a Hugging Face model for `outlier_channels`: one block, called with `input_ids`."""

    def __init__(self):
        super().__init__()
        self.block = torch.nn.Identity()

    def forward(self, input_ids, use_cache=False):
        return self.block(input_ids)


def activations():
    """A block output: noise everywhere and, on each sequence's first token, two massive channels."""
    x = torch.randn(4, 64, HIDDEN)
    x[:, 0, MASSIVE] = 2e4
    return x


def summary(block, x):
    out = block(x)
    return {"std": float(out.std()), "abs_max": float(out.abs().max()), "zero_frac": 0.0}


def run_detector(fault):
    torch.manual_seed(0)
    model = Stub()
    channels = outlier_channels(model, "block", fault, ORDINARY, HIDDEN, activations())
    flag, handle = install_outlier(model.block, HIDDEN, channels, 100.0, "cpu", torch.float32)
    records = []
    for step in range(BASELINE_STEPS + 3):
        if step == SEED_STEP:
            flag.fill_(True)
        records.append({"step": step, "layers": {"block": summary(model.block, activations())}})
    handle.remove()
    return channels, {i["kind"]: i for i in detect_activation_anomalies(records)}


def test_massive_takes_the_channels_with_the_largest_outputs():
    channels = outlier_channels(Stub(), "block", "massive", ORDINARY, HIDDEN, activations())
    assert channels == sorted(MASSIVE)  # two were named, so two are taken


def test_massive_takes_as_many_channels_as_were_named():
    channels = outlier_channels(Stub(), "block", "massive", [1, 2, 3], HIDDEN, activations())
    assert len(channels) == 3 and set(MASSIVE) <= set(channels)


def test_block_takes_every_channel_and_channels_takes_the_ones_named():
    assert outlier_channels(Stub(), "block", "block", ORDINARY, HIDDEN, activations()) == list(range(HIDDEN))
    assert outlier_channels(Stub(), "block", "channels", ORDINARY, HIDDEN, activations()) == ORDINARY


def test_calibration_leaves_no_hook_behind():
    model = Stub()
    outlier_channels(model, "block", "massive", ORDINARY, HIDDEN, activations())
    assert not model.block._forward_hooks


@pytest.mark.parametrize("fault", ["massive", "block"])
def test_a_block_summary_sees_the_massive_and_the_block_wide_fault(fault):
    _, found = run_detector(fault)
    assert found["activation_spike"]["step"] == SEED_STEP
    assert [m["module"] for m in found["activation_spike"]["modules"]] == ["block"]


def test_a_block_summary_does_not_see_two_ordinary_channels_next_to_massive_ones():
    # What the first A100 run found; if the detector ever learns to see this, this test should change.
    _, found = run_detector("channels")
    assert "activation_spike" not in found


def test_every_fault_is_offered_by_the_scripts():
    assert set(FAULTS) == {"massive", "block", "channels"}
