"""`examples/split_model_check.py` (the multi-GPU test plan's T4): one model split by layer over two
devices in one process, by Accelerate's `device_map`, by hand with `.to()`, an MoE model split the
same way, and Accelerate's offload.

This machine has no two CUDA devices, so each arm runs here with both halves on the CPU: the
harness, its plants, its own norms and its comparisons run; the cross-device copy, the
synchronisation count and a router on another device than the counts do not, and the run must
say so (exit 3, NOT PROVEN, with nothing failed). The offload arm offloads to disk here (Accelerate
does not offload to the CPU when the main device is the CPU), which is said too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

SDK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK / "examples"))

pytest.importorskip("transformers")
pytest.importorskip("accelerate")

import split_model_check as smc  # noqa: E402

FAST = ["--toy", "--devices", "cpu,cpu", "--steps", "40", "--grad-step", "15", "--act-step", "25",
        "--timed-steps", "6"]


def _run(tmp_path: Path, arm: str, *extra: str) -> tuple[int, dict]:
    out = tmp_path / arm
    code = smc.main([*FAST, "--arm", arm, "--out", str(out), *extra])
    return code, json.loads((out / "verdict.json").read_text())


def _passed(verdict: dict, text: str) -> bool:
    return any(line.startswith("PASS") and text in line for line in verdict["lines"])


@pytest.mark.parametrize("arm", ["device-map", "manual"])
def test_a_dense_split_on_one_cpu_runs_every_check_and_is_not_proven(tmp_path, arm):
    code, verdict = _run(tmp_path, arm)
    assert verdict["failures"] == [], json.dumps(verdict["lines"], indent=1)
    assert code == smc.EXIT_NOT_PROVEN and verdict["status"] == "NOT PROVEN"
    assert set(verdict["not_exercised"]) >= {"cross-device copy", "synchronisation count"}
    assert verdict["devices"] == ["cpu", "cpu"]
    for text in ("every block has a layers entry", "no layers_not_read", "activation spike at step 25",
                 "grad_norm equals the script's own", "grad_spike at step 15", "weight_norms of the"):
        assert _passed(verdict, text), (text, verdict["lines"])
    assert verdict["second_half"]  # the blocks placed on the second device, named
    assert verdict["act_block"] in verdict["second_half"]
    assert verdict["grad_plant"]["parameter"].startswith(verdict["second_half"][-1] + ".")


def test_the_device_map_arm_puts_accelerate_hooks_on_the_blocks(tmp_path):
    """With every entry on one device Accelerate adds no hook unless forced (`dispatch_model`'s
    `force_hooks`): the dry run forces them, so its blocks go through `AlignDevicesHook` as on two GPUs."""
    code, verdict = _run(tmp_path, "device-map")
    assert verdict["accelerate_hooks"] > 0


def test_the_moe_arm_counts_every_router_once_fixed_and_cannot_show_a_refusal_on_one_device(tmp_path):
    code, verdict = _run(tmp_path, "moe")
    assert verdict["failures"] == [], json.dumps(verdict["lines"], indent=1)
    assert code == smc.EXIT_NOT_PROVEN
    assert "router on another device" in verdict["not_exercised"]
    assert verdict["moe"]["refused_without_fix"] == []  # one device: nothing to refuse
    assert verdict["moe"]["routers"] and _passed(verdict, "sum to tokens x top_k")


def test_the_offload_arm_names_the_offloaded_weights_unread(tmp_path):
    code, verdict = _run(tmp_path, "offload")
    assert verdict["failures"] == [], json.dumps(verdict["lines"], indent=1)
    assert code == smc.EXIT_NOT_PROVEN and verdict["offload_to"] == "disk"
    assert verdict["offloaded"]  # parameters on meta
    assert _passed(verdict, "weight_norms is \"nan\" for every offloaded parameter")


# ---- the pieces ------------------------------------------------------------------------------


def test_the_split_map_puts_the_second_half_of_the_blocks_on_the_second_device():
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                                         num_attention_heads=2, num_key_value_heads=1))
    device_map, second = smc.split_map(model, "cuda:0", "cuda:1")
    assert second == ["model.layers.2", "model.layers.3"]
    assert [device_map[f"model.layers.{i}"] for i in range(4)] == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
    # Everything else stays with the embedding, so a tied head is never split from it.
    assert device_map["model.embed_tokens"] == device_map["lm_head"] == device_map["model.norm"] == "cuda:0"


def test_the_sync_rule_allows_no_added_synchronising_call_off_the_flush_steps():
    off = [2, 2, 2, 2]
    assert smc.sync_verdict(off, [2, 2, 2, 2], [False, False, False, True])["passed"]
    assert smc.sync_verdict(off, [2, 2, 2, 9], [False, False, False, True])["flush_extra"] == 7
    bad = smc.sync_verdict(off, [2, 3, 2, 2], [False, False, False, True])
    assert not bad["passed"] and bad["added"] == {1: 1}


@pytest.mark.parametrize("argv, text", [
    (["--devices", "cpu"], "two devices"),
    (["--grad-step", "30", "--act-step", "20"], "--grad-step"),
])
def test_bad_arguments_exit_2(tmp_path, argv, text, capsys):
    out = tmp_path / "bad"
    assert smc.main(["--toy", "--out", str(out), *argv]) == smc.EXIT_USAGE
    assert text in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.skipif(torch.cuda.device_count() >= 2, reason="checks the path taken without two CUDA GPUs")
def test_two_cuda_devices_asked_for_and_missing_exit_2(tmp_path, capsys):
    assert smc.main(["--toy", "--out", str(tmp_path / "r")]) == smc.EXIT_USAGE
    assert "CUDA" in capsys.readouterr().err
