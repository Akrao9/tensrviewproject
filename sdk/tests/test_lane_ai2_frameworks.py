"""Lane AI-2: the framework harnesses `lightning_check.py`, `trainer_ddp_check.py` and
`megatron_routing_check.py`, run on their CPU paths, and the comparisons they judge with.

A CPU run only shows that a harness starts, makes its comparisons and writes its files: the claims
are the GPU run's (Lightning's FSDP and Megatron's own `TopKRouter` cannot run here at all).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SDK = Path(__file__).resolve().parents[1]
EXAMPLES = SDK / "examples"
sys.path.insert(0, str(EXAMPLES))

import megatron_routing_check as mrc  # noqa: E402

CLONE = mrc.DEFAULT_CLONE
needs_distributed = pytest.mark.skipif(not torch.distributed.is_available(), reason="needs torch.distributed")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SDK / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    return env


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(args: list[str], cwd: Path, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=cwd, env=_env(), capture_output=True, text=True, timeout=timeout)


# ---- Megatron's routing map ---------------------------------------------------------------------


def test_megatron_routing_cpu_compares_every_arm(tmp_path):
    out = tmp_path / "routing"
    assert mrc.main(["--cpu", "--out", str(out)]) == mrc.EXIT_PASS
    result = json.loads((out / "result.json").read_text())
    assert result["passed"] and result["megatron_router"] is False
    assert [a["arm"] for a in result["arms"]] == ["off", "on", "on-pad"]
    off, on, pad = result["arms"]
    assert off["totals"]["dropped"] == 0 and on["totals"]["dropped"] > 0 and pad["totals"]["padding"] > 0
    assert all((out / f"{arm}.jsonl").exists() for arm in ("off", "on", "on-pad"))
    expected = "executed from" if CLONE is not None else "inline copy"
    assert expected in result["router"]


def test_megatron_routing_cpu_position_policy(tmp_path):
    assert mrc.main(["--cpu", "--out", str(tmp_path / "r"), "--drop-policy", "position", "--steps", "3"]) == 0


@pytest.mark.skipif(CLONE is None, reason="no Megatron-LM clone beside the project")
def test_megatron_dropping_from_the_clone_equals_the_inline_copy():
    executed, origin = mrc.megatron_dropping(CLONE)
    assert origin.startswith("executed from")
    inline = mrc._inline_dropping()
    g = torch.Generator().manual_seed(0)
    for policy in ("probs", "position"):
        for pad in (False, True):
            logits = torch.randn(40, 8, generator=g)
            top = logits.topk(2, dim=1)
            probs = torch.zeros_like(logits).scatter(1, top.indices, top.values.softmax(-1))
            routing_map = torch.zeros_like(logits).int().scatter(1, top.indices, 1).bool()
            a = executed["apply_router_token_dropping"](probs, routing_map, 2, 0.75, policy, pad)
            b = inline["apply_router_token_dropping"](probs, routing_map, 2, 0.75, policy, pad)
            assert torch.equal(a[1], b[1]) and torch.equal(a[0], b[0])


def _judge_args(steps=1, seq=4, batch=1, experts=4, topk=1):
    return type("A", (), {"steps": steps, "seq": seq, "batch": batch, "experts": experts, "topk": topk})()


def test_megatron_judge_fails_when_the_record_disagrees_with_the_router():
    before = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    after = before.clone()
    after[3, 0] = False  # the router dropped token 3's assignment to expert 0
    probs = after.float()
    good = [{"step": 0, "moe": {"router": [2, 1, 0, 0]}, "moe_left_out": {"router": [0, 0, 0, 0]}}]
    assert mrc.judge("on", _judge_args(), good, [(probs, after)], [before], 1.0)["passed"]
    # Counting the dropped assignment, as a router hooked before its dropping would, is caught.
    bad = [{"step": 0, "moe": {"router": [3, 1, 0, 0]}, "moe_left_out": {"router": [0, 0, 0, 0]}}]
    result = mrc.judge("on", _judge_args(), bad, [(probs, after)], [before], 1.0)
    assert not result["passed"] and any("moe [3, 1, 0, 0] != kept [2, 1, 0, 0]" in f for f in result["failures"])


def test_megatron_judge_fails_when_dropping_on_drops_nothing():
    chosen = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    records = [{"step": 0, "moe": {"router": [1, 1]}, "moe_left_out": {"router": [0, 0]}}]
    result = mrc.judge("on", _judge_args(seq=2, experts=2), records, [(chosen.float(), chosen)], [chosen], 1.0)
    assert not result["passed"] and any("dropping on" in f for f in result["failures"])


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the path taken without CUDA")
def test_megatron_router_path_without_cuda_says_what_is_missing(tmp_path, capsys):
    assert mrc.main(["--out", str(tmp_path / "r")]) == mrc.EXIT_MISSING
    assert "needs CUDA" in capsys.readouterr().out
    assert not (tmp_path / "r").exists()


# ---- Lightning ----------------------------------------------------------------------------------


@needs_distributed
def test_lightning_ddp_on_two_cpu_ranks(tmp_path):
    pytest.importorskip("lightning")
    pytest.importorskip("transformers")
    out = tmp_path / "lightning"
    done = _run([str(EXAMPLES / "lightning_check.py"), "--toy", "--devices", "2", "--steps", "120",
                 "--seed-step", "60", "--out", str(out)], cwd=tmp_path)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    result = json.loads((out / "result.json").read_text())
    assert result["passed"] and result["strategy"] == "DDPStrategy" and result["world_size"] == 2
    assert result["captures"] == ["lightning-check.rank0.jsonl", "lightning-check.rank1.jsonl"]
    assert all(r["spike"] == {"step": 60, "first": result["seeded_block"]} for r in result["ranks"].values())
    assert "All checks passed" in done.stdout


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks Lightning's refusal of FSDP without CUDA")
def test_lightning_fsdp_without_cuda_stops_with_lightnings_own_refusal(tmp_path):
    pytest.importorskip("lightning")
    pytest.importorskip("transformers")
    done = _run([str(EXAMPLES / "lightning_check.py"), "--toy", "--strategy", "fsdp", "--devices", "2",
                 "--steps", "60", "--seed-step", "50", "--out", str(tmp_path / "f")], cwd=tmp_path)
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "requires a GPU accelerator" in done.stdout


# ---- Hugging Face Trainer under DDP -------------------------------------------------------------


def _torchrun(tmp_path, *args):
    return _run(["-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-addr", "127.0.0.1",
                 "--master-port", str(_free_port()), str(EXAMPLES / "trainer_ddp_check.py"), "--toy",
                 "--steps", "120", "--seed-step", "60", "--log-every", "10", "--probe-every", "40", *args],
                cwd=tmp_path)


@needs_distributed
@pytest.mark.parametrize("mode", [["--eager"], ["--backend", "aot_eager"]], ids=["eager", "aot_eager"])
def test_trainer_ddp_on_two_cpu_ranks(tmp_path, mode):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    out = tmp_path / "trainer-ddp"
    done = _torchrun(tmp_path, *mode, "--out", str(out))
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    result = json.loads((out / "result.json").read_text())
    assert result["passed"] and result["world_size"] == 2 and result["distributed_type"] == "DistributedType.MULTI_CPU"
    assert result["captures"] == ["trainer-ddp-check.rank0.jsonl", "trainer-ddp-check.rank1.jsonl"]
    assert all(r["spike"]["step"] == 60 for r in result["ranks"].values())


@needs_distributed
def test_trainer_ddp_control_counts_the_same_wrapper_breaks_without_tensorquill(tmp_path):
    """The author's terms, Nov 22, 2026: the DDP wrapper's graph breaks are "documented as observed in
    both control and instrumented runs". The control run has no TensorQuill callback and no capture."""
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    results = {}
    for name, extra in (("control", ["--control"]), ("checked", [])):
        out = tmp_path / name
        done = _torchrun(tmp_path, "--backend", "aot_eager", *extra, "--out", str(out))
        assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
        results[name] = json.loads((out / "result.json").read_text())
    control, checked = results["control"], results["checked"]
    assert control["control"] is True and control["passed"] and control["tensorquill"] == "not attached"
    assert not list((tmp_path / "control").glob("*.jsonl"))  # nothing captured
    assert checked["control"] is False and checked["passed"]
    for rank in ("0", "1"):
        without, with_tq = control["ranks"][rank], checked["ranks"][rank]
        assert sum(without["ddp_wrapper_breaks"].values()) >= 1
        assert without["ddp_wrapper_breaks"] == with_tq["ddp_wrapper_breaks"]  # the same breaks, the same counts
        assert without["other_breaks"] == {} == with_tq["other_breaks"]


def test_trainer_ddp_control_cannot_be_eager(tmp_path):
    done = _run([str(EXAMPLES / "trainer_ddp_check.py"), "--toy", "--control", "--eager", "--out", str(tmp_path / "t")], cwd=tmp_path)
    assert done.returncode == 2 and "--control" in done.stderr


def test_trainer_ddp_refuses_one_process(tmp_path):
    done = _run([str(EXAMPLES / "trainer_ddp_check.py"), "--toy", "--out", str(tmp_path / "t")], cwd=tmp_path)
    assert done.returncode == 2 and "torchrun" in done.stderr


def test_skip_agreement_compares_each_step_with_accelerate():
    from trainer_ddp_check import skip_agreement

    records = [{"step": 0, "grad_scale": 65536.0}, {"step": 1, "grad_scale": 32768.0, "step_skipped": True},
               {"step": 2, "grad_scale": 32768.0}]
    assert skip_agreement(records, [False, True, False]) == []
    assert skip_agreement(records, [True, True, False]) == [
        "step 0: recorded step_skipped=False, Accelerate step_was_skipped=True"]
    assert skip_agreement(records[:2], [False, True, False]) == ["step 2: not recorded"]
    assert skip_agreement([{"step": 0}], [False]) == ["step 0: no grad_scale"]
    # An optimizer whose flag could not be read is unknown, never "not skipped".
    assert skip_agreement(records[:1], [None]) == ["step 0: Accelerate's step_was_skipped could not be read"]


def test_split_breaks_keeps_only_ddp_wrapper_breaks_apart():
    from trainer_ddp_check import split_breaks

    wrapper = "Attempted to call function marked as skipped\n  Explanation: ... `DistributedDataParallel._inside_ddp_forward` in"
    logger = "Unsupported method call\n  Explanation: Dynamo does not know how to trace method `set_runtime_stats_and_log` of class `Logger`"
    model = "Data-dependent branching\n  Explanation: in the model's forward"
    ddp, other = split_breaks({wrapper: 2, logger: 1, model: 1})
    assert ddp == {wrapper: 2, logger: 1} and other == {model: 1}


def test_mean_over_ranks():
    from trainer_ddp_check import mean_over_ranks

    assert mean_over_ranks([[1.0, 2.0, None], [3.0, "nan", 1.0]]) == [2.0, pytest.approx(float("nan"), nan_ok=True), None]


def test_lightning_expects_the_watched_wrapper_of_a_wrapped_block():
    # FSDP's per-layer wrap puts each decoder layer inside a wrapper, as `_fsdp_wrapped_module`; the
    # blocks TensorQuill watches are then the wrappers, and the spike is expected there.
    from lightning_check import watched_block
    from torch import nn

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._fsdp_wrapped_module = inner

        def forward(self, x):
            return self._fsdp_wrapped_module(x)

    layers = [nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(4)]
    plain = nn.Module()
    plain.layers = nn.ModuleList(layers)
    assert watched_block(plain, layers[2]) == "layers.2"
    wrapped = nn.Module()
    wrapped.layers = nn.ModuleList(Wrapper(layer) for layer in layers)
    assert watched_block(wrapped, layers[2]) == "layers.2"
    assert watched_block(wrapped, nn.Linear(1, 1)) is None
