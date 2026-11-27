"""`examples/fsdp_check.py` (the multi-GPU test plan's T5): a plain loop under FSDP2 (`fully_shard`
per block) and FSDP1 (`FullyShardedDataParallel(use_orig_params=True)`), run here on two Gloo
ranks on the CPU, and the arithmetic it judges with.

A CPU run shows that the harness starts both ranks, plants, computes its own whole-parameter
norms, makes its comparisons and writes its files; the claim about NCCL and GPUs is the GPU run's.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

SDK = Path(__file__).resolve().parents[1]
EXAMPLES = SDK / "examples"
sys.path.insert(0, str(EXAMPLES))

import fsdp_check  # noqa: E402

needs_distributed = pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")


def _cpu_fsdp2_supported() -> bool:
    """As `test_capture_tensors.py` asks it: FSDP2 on the CPU needs `torch.cpu.Stream.record_event` (2.7+)."""
    stream = getattr(getattr(torch, "cpu", None), "Stream", None)
    return stream is not None and hasattr(stream, "record_event")


needs_cpu_fsdp2 = pytest.mark.skipif(not _cpu_fsdp2_supported(), reason="FSDP2 on the CPU needs PyTorch 2.7 or later")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SDK / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    env["OMP_NUM_THREADS"] = "1"
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    return env


def _drive(tmp_path: Path, *args: str) -> tuple[int, dict | None, str]:
    """The driver in this process (it starts the ranks itself), with the variables the tests need."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(_env())
    try:
        code = fsdp_check.main(["--cpu", *args])
    finally:
        os.environ.clear()
        os.environ.update(saved)
    out = Path(args[args.index("--out") + 1])
    verdict = out / "verdict.json"
    log = (out / "launcher.log").read_text(errors="replace") if (out / "launcher.log").exists() else ""
    return code, (json.loads(verdict.read_text()) if verdict.exists() else None), log


# ---- the two arms on two Gloo ranks ------------------------------------------------------------


@needs_distributed
@needs_cpu_fsdp2
def test_fsdp2_on_two_gloo_ranks_passes(tmp_path):
    out = tmp_path / "fsdp2"
    code, verdict, log = _drive(tmp_path, "--arm", "fsdp2", "--nproc", "2", "--out", str(out))
    assert verdict is not None, log[-4000:]
    assert code == fsdp_check.EXIT_PASS, json.dumps(verdict["lines"], indent=1) + log[-3000:]
    assert verdict["status"] == "PASS" and verdict["world_size"] == 2 and verdict["arm"] == "fsdp2"
    assert verdict["captures"] == ["fsdp-check.rank0.jsonl", "fsdp-check.rank1.jsonl"]
    planted = verdict["grad_plant"]["parameter"]
    assert all(size > 0 for size in verdict["grad_plant"]["shard_sizes"])  # every rank holds part of it
    assert planted.startswith(verdict["grad_plant"]["block"] + ".")
    for rank in ("0", "1"):
        facts = verdict["ranks"][rank]
        assert facts["activation_spike"] == {"step": verdict["act_step"], "first": verdict["act_block"]}
        assert facts["probe"]["entered"] == 0 and "FSDP2" in facts["probe"]["not_run"]
    assert verdict["combined"]["grad_norm"]["steps"] == verdict["steps"]
    assert verdict["combined"]["grad_norm"]["worst"] < 1e-4


@needs_distributed
def test_fsdp1_on_two_gloo_ranks_passes(tmp_path):
    out = tmp_path / "fsdp1"
    code, verdict, log = _drive(tmp_path, "--arm", "fsdp1", "--nproc", "2", "--out", str(out))
    assert verdict is not None, log[-4000:]
    assert code == fsdp_check.EXIT_PASS, json.dumps(verdict["lines"], indent=1) + log[-3000:]
    assert verdict["status"] == "PASS" and verdict["arm"] == "fsdp1"
    for rank in ("0", "1"):
        assert "FSDP1" in verdict["ranks"][rank]["probe"]["not_run"]
        assert verdict["ranks"][rank]["probe"]["entered"] == 0


@needs_distributed
def test_fsdp1_with_a_parameter_one_rank_does_not_hold_is_not_proven(tmp_path):
    """FSDP1 shards the flattened unit: with one unit per block on two ranks the start of a block's
    flat parameter, its norm's weight, is rank 0's alone. Planting there leaves rank 1 nothing to
    see, and the row says that is not proven."""
    out = tmp_path / "fsdp1-empty"
    code, verdict, log = _drive(tmp_path, "--arm", "fsdp1", "--nproc", "2", "--grad-param", "norm.weight",
                                "--out", str(out))
    assert verdict is not None, log[-4000:]
    assert code == fsdp_check.EXIT_NOT_PROVEN, json.dumps(verdict["lines"], indent=1)
    assert verdict["status"] == "NOT PROVEN"
    assert 0 in verdict["grad_plant"]["shard_sizes"]
    assert any("empty" in reason for reason in verdict["not_proven"])


# ---- the arithmetic ----------------------------------------------------------------------------


def test_rank_norms_combine_by_the_root_of_the_sum_of_squares():
    assert fsdp_check.combine([3.0, 4.0]) == pytest.approx(5.0)
    assert fsdp_check.combine([3.0, 0.0, 4.0]) == pytest.approx(5.0)
    assert fsdp_check.combine([None, 4.0]) is None  # a rank with no norm: nothing to combine
    assert fsdp_check.combine([]) is None


def test_rms_over_shards_weights_each_square_by_its_element_count():
    # A parameter of 4 elements, 3 on rank 0 with RMS 2 and 1 on rank 1 with RMS 4:
    # mean square (3 * 4 + 1 * 16) / 4 = 7.
    assert fsdp_check.combine_rms([(2.0, 3), (4.0, 1)]) == pytest.approx(math.sqrt(7.0))
    assert fsdp_check.combine_rms([(2.0, 3), (None, 0)]) == pytest.approx(2.0)  # an empty shard has nothing
    assert fsdp_check.combine_rms([(2.0, 3), (None, 1)]) is None  # a shard with values and no reading


def test_the_planted_parameter_is_one_every_rank_holds_part_of():
    sizes = [
        {"blocks.1.norm.weight": 8, "blocks.1.fc1.weight": 256, "blocks.1.fc2.weight": 0},
        {"blocks.1.norm.weight": 0, "blocks.1.fc1.weight": 0, "blocks.1.fc2.weight": 256},
    ]
    name, why = fsdp_check.choose_param(sizes, "blocks.1")
    assert name is None and "every rank" in why
    sizes[1]["blocks.1.fc1.weight"] = 128
    assert fsdp_check.choose_param(sizes, "blocks.1") == ("blocks.1.fc1.weight", None)
    # Named by hand, it is taken as named, and an empty shard is said.
    name, why = fsdp_check.choose_param(sizes, "blocks.1", wanted="fc2.weight")
    assert name == "blocks.1.fc2.weight" and "empty on rank 0" in why
    name, why = fsdp_check.choose_param(sizes, "blocks.1", wanted="nope")
    assert name is None and "no parameter" in why


@pytest.mark.parametrize("argv, text", [
    (["--grad-step", "40", "--act-step", "30"], "--grad-step"),
    (["--grad-step", "5"], "history"),
    (["--act-step", "60", "--steps", "60"], "--act-step"),
    (["--nproc", "1"], "--nproc"),
])
def test_bad_arguments_exit_2_before_anything_starts(tmp_path, argv, text, capsys):
    out = tmp_path / "bad"
    assert fsdp_check.main(["--cpu", "--out", str(out), *argv]) == fsdp_check.EXIT_USAGE
    assert text in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the path taken without CUDA")
def test_without_cuda_and_without_cpu_it_says_what_is_missing(tmp_path, capsys):
    assert fsdp_check.main(["--out", str(tmp_path / "r")]) == fsdp_check.EXIT_USAGE
    assert "CUDA" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()
