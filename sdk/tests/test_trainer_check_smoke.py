"""`examples/trainer_check.py --smoke` (the multi-GPU test plan's T3, the unscored smoke): the run
without the plant, for a machine where the `Trainer` sees two GPUs without torchrun and wraps the
model in `nn.DataParallel`. It records what it saw in `smoke.json`. The default run is unchanged.

Neither a second GPU nor `nn.DataParallel` across devices exists here: the smoke runs on the CPU,
where `n_gpu` is 0 and nothing is wrapped; the check of the wrap itself is tested on a real
`nn.DataParallel` object, which PyTorch builds without CUDA (it then calls the module directly).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SDK = Path(__file__).resolve().parents[1]
EXAMPLES = SDK / "examples"
sys.path.insert(0, str(EXAMPLES))

pytest.importorskip("transformers")
pytest.importorskip("accelerate")

TOY = ["--toy", "--eager", "--steps", "120", "--seed-step", "60", "--log-every", "10", "--probe-every", "40"]

# The default toy run's check lines, as the script printed them before the option was added.
DEFAULT_LINES = [
    "PASS: 120 records, steps 0 to 119, matching Trainer's global_step",
    "PASS: train/loss recorded on every step",
    "PASS: clip_grad_norm recorded on every step",
    "PASS: interval mean of train/loss agrees with Trainer's logged loss on 12 of 12 logged steps (largest difference 0.000%)",
    "PASS: clip_grad_norm agrees with Trainer's logged grad_norm on 12 of 12 logged steps (largest difference 0.000%)",
    "PASS: activation spike at step 60, first in model.layers.2",
    "PASS: 3 probe passes, none failed",
]


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SDK / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    return subprocess.run([sys.executable, str(EXAMPLES / "trainer_check.py"), *TOY, *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=600)


def _checks(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith(("PASS:", "FAIL:", "NOT CHECKED:"))]


def test_the_default_run_is_unchanged(tmp_path):
    out = tmp_path / "default"
    done = _run(tmp_path, "--out", str(out))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert _checks(done.stdout) == DEFAULT_LINES
    assert done.stdout.rstrip().endswith("All checks passed")
    assert not (out / "smoke.json").exists()
    assert sorted(p.name for p in out.iterdir()) == ["trainer", "trainer-check.run.json", "trainer.end.json", "trainer.jsonl"]


def test_the_smoke_leaves_the_plant_out_and_records_what_it_saw(tmp_path):
    out = tmp_path / "smoke"
    done = _run(tmp_path, "--smoke", "--out", str(out))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    smoke = json.loads((out / "smoke.json").read_text())
    assert smoke["scored"] is False and smoke["plant"] is None
    assert smoke["cuda_device_count"] == torch.cuda.device_count()
    assert smoke["n_gpu"] == 0 and smoke["data_parallel"] is False  # the CPU: nothing to wrap
    assert smoke["model_wrapped"] == "LlamaForCausalLM"
    assert smoke["train_error"] is None
    checks = smoke["checks"]
    assert checks["activation spike"] == "not checked: no plant"
    assert {k: v for k, v in checks.items() if k != "activation spike"} == {
        "steps": "PASS", "train/loss": "PASS", "clip_grad_norm": "PASS", "loss agreement": "PASS",
        "grad_norm agreement": "PASS", "probe": "PASS",
    }
    assert "NOT CHECKED: activation spike (no plant: --smoke)" in done.stdout
    # Nothing was planted: no activation spike was recorded.
    from tensorquill import build_evidence, load_records

    evidence = build_evidence(load_records(out / "trainer.jsonl"))
    assert not [i for i in evidence["incidents"] if i["kind"] == "activation_spike"]


def test_the_wrap_is_read_through_compile_and_the_trainers_own_attribute():
    from trainer_check import data_parallel_wrap

    inner = torch.nn.Linear(2, 2)
    wrapped = torch.nn.DataParallel(inner)
    assert data_parallel_wrap(type("T", (), {"model_wrapped": wrapped})()) == (True, "DataParallel")
    compiled = torch.compile(wrapped, backend="eager")
    assert data_parallel_wrap(type("T", (), {"model_wrapped": compiled})()) == (True, "OptimizedModule(DataParallel)")
    assert data_parallel_wrap(type("T", (), {"model_wrapped": inner})()) == (False, "Linear")
