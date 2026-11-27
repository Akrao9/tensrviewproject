"""`examples/megatron_tp_check.py` (the multi-GPU test plan's T9, tensor parallel only): Megatron-core's
MoE layer under TP=2 with sequence parallelism, its router counts against its own `routing_map`, and
the `replica_params` filter. `megatron_routing_check.py`, the one-GPU check, is left as it was.

Megatron needs CUDA and `triton`, so its path cannot run here: the tests check that it says so
and stops before starting anything. The `--cpu` stand-in runs on two Gloo ranks: a router with
Megatron's `TopKRouter` contract fed each rank's slice of the sequence, two linear layers sharded by
hand and marked `tensor_model_parallel` as Megatron marks its own, a replicated norm, and the
non-tensor-parallel gradients summed over the group as Megatron's `finalize_model_grads` does
under sequence parallelism. It shows the driver, the per-rank comparison and the filter on real
ranks, not Megatron.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SDK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK / "examples"))

import megatron_tp_check as mtp  # noqa: E402

needs_distributed = pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")


def _drive(*args: str) -> int:
    saved = dict(os.environ)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(SDK / "src"), saved.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(name, None)
    try:
        return mtp.main(list(args))
    finally:
        os.environ.clear()
        os.environ.update(saved)


@needs_distributed
def test_the_stand_in_on_two_gloo_ranks_passes_both_arms(tmp_path):
    out = tmp_path / "tp"
    code = _drive("--cpu", "--nproc", "2", "--out", str(out))
    verdict = json.loads((out / "verdict.json").read_text())
    assert code == mtp.EXIT_PASS, json.dumps(verdict["lines"], indent=1) + (out / "launcher.log").read_text()[-3000:]
    assert verdict["megatron"] is False and verdict["tp"] == 2 and verdict["world_size"] == 2
    filtered, unfiltered = verdict["arms"]["filtered"], verdict["arms"]["unfiltered"]
    # Each rank routed its half of the sequence, and the router's counts were TensorQuill's.
    tokens = verdict["seq"] * verdict["batch"]
    assert filtered["tokens_per_rank"] == [tokens // 2, tokens // 2]
    assert filtered["count_mismatches"] == 0 and unfiltered["count_mismatches"] == 0
    # The filter keeps only what every rank of the group holds whole.
    assert filtered["replica_names"] == filtered["expected_replicas"]
    assert not any(name in filtered["tensor_parallel"] for name in filtered["replica_names"])
    assert filtered["drift"] is None
    # Without it the sharded weights read as drift, as the README says they do.
    assert unfiltered["drift"] and set(unfiltered["drift"]) <= set(unfiltered["tensor_parallel"])


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the path taken without CUDA")
def test_the_megatron_path_without_cuda_says_what_is_missing(tmp_path, capsys):
    out = tmp_path / "r"
    assert mtp.main(["--out", str(out)]) == mtp.EXIT_USAGE
    assert "CUDA" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("argv, text", [
    (["--tp", "4", "--nproc", "2"], "--nproc"),
    (["--seq", "30", "--tp", "4", "--nproc", "4"], "--seq"),
    (["--tp", "1"], "--tp"),
])
def test_bad_arguments_exit_2(tmp_path, argv, text, capsys):
    out = tmp_path / "bad"
    assert mtp.main(["--cpu", "--out", str(out), *argv]) == mtp.EXIT_USAGE
    assert text in capsys.readouterr().err
    assert not out.exists()


# ---- the arithmetic ----------------------------------------------------------------------------


def test_each_tensor_parallel_rank_takes_its_slice_of_the_sequence():
    assert [mtp.sequence_slice(32, 2, r) for r in range(2)] == [(0, 16), (16, 32)]
    assert [mtp.sequence_slice(32, 4, r) for r in range(4)] == [(0, 8), (8, 16), (16, 24), (24, 32)]
    with pytest.raises(ValueError):
        mtp.sequence_slice(30, 4, 0)


def test_the_expected_replicas_are_the_parameters_megatron_does_not_shard():
    def p(**marks):
        tensor = torch.nn.Parameter(torch.zeros(2))
        for key, value in marks.items():
            setattr(tensor, key, value)
        return tensor

    named = [("norm.weight", p(sequence_parallel=True)), ("moe.router.weight", p(sequence_parallel=True)),
             ("moe.experts.local_experts.0.linear_fc1.weight", p(tensor_model_parallel=True)),
             ("frozen", p(tensor_model_parallel=False))]
    assert mtp.expected_replicas(named) == ["norm.weight", "moe.router.weight", "frozen"]
    assert mtp.REPLICA_FILTER("x", named[2][1]) is False and mtp.REPLICA_FILTER("x", named[0][1]) is True


def test_a_recorded_count_that_differs_from_the_routing_map_is_counted_per_step():
    spy = [[3, 1, 0, 0], [2, 2, 0, 0]]
    records = [{"step": 0, "moe": {"moe.router": [3, 1, 0, 0]}}, {"step": 1, "moe": {"moe.router": [2, 1, 1, 0]}}]
    problems = mtp.count_problems(records, spy, "moe.router")
    assert problems == ["step 1: recorded [2, 1, 1, 0], routing_map sums [2, 2, 0, 0]"]
    assert mtp.count_problems(records[:1], spy, "moe.router") == ["step 1: not recorded"]


def test_a_map_in_another_form_than_tokens_by_experts_is_not_proven():
    good = SimpleNamespace(dtype=torch.bool, shape=(16, 8))
    assert mtp.map_form_problem([good], tokens=16, experts=8) is None
    dense = SimpleNamespace(dtype=torch.int64, shape=(16, 2))  # top-k indices, a Flex backend's form
    assert "top-k" in mtp.map_form_problem([dense], tokens=16, experts=8)
