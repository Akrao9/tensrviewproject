"""Lane V8 (wave V): the examples, the experiment scripts and CI.

One or more tests a row of `OPEN_BUGS.md`: X06, X07 (CLI-18, CLI-23), X08 (CLI-16), X09, X10,
X11, CLI-17 and CLI-19. The examples are scripts, so what they judged inline in `main()` is now a
function of its own, and the test calls it with the records that broke it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorquill
from tensorquill import mcp_server

SDK = Path(__file__).resolve().parents[1]
REPO = SDK.parent
EXAMPLES = SDK / "examples"
sys.path.insert(0, str(EXAMPLES))


# ---- X06: trainer_check.py and Trainer's own logged norms ---------------------------------


def history(*entries):
    """Trainer's `log_history` entries: (step, loss, grad_norm)."""
    return [{"step": step, "loss": loss, "grad_norm": norm} for step, loss, norm in entries]


def test_x06_nonfinite_strings_against_finite_logged_norms_are_disagreements():
    from trainer_check import agrees, trainer_agreement

    # Every recorded norm is the string the sink writes for NaN; Trainer logged finite norms.
    # Until lane V8, no pair was compared, the largest difference stayed 0 and the check passed.
    losses = [1.0] * 4
    norms = ["nan"] * 4
    result = trainer_agreement(losses, norms, history((2, 1.0, 0.5), (4, 1.0, 0.7)), log_every=2)
    assert result["grad_norm"]["pairs"] == 2
    assert len(result["grad_norm"]["problems"]) == 2 and "nan" in result["grad_norm"]["problems"][0]
    assert not agrees(result["grad_norm"])
    assert agrees(result["loss"]) and result["loss"]["pairs"] == 2


def test_x06_a_nonfinite_norm_both_sides_logged_is_an_explicit_agreement():
    from trainer_check import agrees, trainer_agreement

    norms = [1.0, "nan", 1.0, "inf"]
    result = trainer_agreement([1.0] * 4, norms, history((2, 1.0, math.nan), (4, 1.0, math.inf)), log_every=2)
    assert result["grad_norm"] == {"pairs": 2, "worst": 0.0, "problems": []}
    assert agrees(result["grad_norm"])
    # NaN on one side and infinity on the other is not the same value.
    result = trainer_agreement([1.0] * 4, [1.0, "nan", 1.0, "inf"], history((2, 1.0, math.inf)), log_every=2)
    assert not agrees(result["grad_norm"])


def test_x06_a_missing_pair_fails_and_no_pair_at_all_fails():
    from trainer_check import agrees, trainer_agreement

    result = trainer_agreement([1.0, 1.0, None, None], [0.5, 0.5, 0.7, None], history((2, 1.0, 0.5), (4, 1.0, 0.7)), 2)
    assert result["grad_norm"]["pairs"] == 1 and len(result["grad_norm"]["problems"]) == 1
    assert "step 3" in result["grad_norm"]["problems"][0]
    assert len(result["loss"]["problems"]) == 1 and not agrees(result["loss"])
    empty = trainer_agreement([1.0] * 4, [0.5] * 4, [], 2)
    assert empty["grad_norm"]["pairs"] == 0 and not agrees(empty["grad_norm"]) and not agrees(empty["loss"])


def test_x06_finite_pairs_are_compared_and_a_nan_loss_is_matched_by_class():
    from trainer_check import agrees, recorded, trainer_agreement

    result = trainer_agreement([1.0, 3.0, "nan", 1.0], [0.5, 0.501, 0.7, 0.7],
                               history((2, 2.0, 0.5), (4, math.nan, 0.7)), log_every=2)
    assert result["loss"] == {"pairs": 2, "worst": 0.0, "problems": []}
    assert result["grad_norm"]["pairs"] == 2 and result["grad_norm"]["worst"] == pytest.approx(0.002)
    assert agrees(result["grad_norm"]) and agrees(result["loss"])
    # "Recorded on every step": a number or the sink's string for a non-finite one, nothing else.
    assert recorded(1.0) and recorded("nan") and recorded("-inf")
    assert not recorded("x") and not recorded(None) and not recorded(True)


# ---- X07 / CLI-18: dense_outlier_check.py prints a failed probe ---------------------------


def probe_record(step, probe):
    return {"step": step, "probe": probe}


def test_cli18_a_failed_probe_is_kept_for_the_summary_and_left_out_of_before_and_after():
    from dense_outlier_check import split_probes

    ok = {"histograms": {}, "attention": {}}
    records = [probe_record(9, ok), probe_record(19, {"error": "RuntimeError('boom')"}), {"step": 20},
               probe_record(29, ok)]
    failed, before, after = split_probes(records, seed_step=20)
    assert failed == [(19, "RuntimeError('boom')")]
    assert [s for s, _ in before] == [9] and [s for s, _ in after] == [29]


def test_x07_dense_attention_text_takes_finite_values_and_says_when_there_are_none():
    from dense_outlier_check import attention_text

    text = attention_text({"max_logit": ["inf", 2.5], "entropy_share": ["nan", 0.5, 0.25], "invalid_rows": 1})
    assert "largest logit 2.50" in text and "entropy share 0.38" in text and "1 invalid rows" in text
    empty = attention_text({"max_logit": ["nan"], "entropy_share": ["nan"], "invalid_rows": 0})
    assert "largest logit n/a" in empty and "entropy share n/a" in empty


# ---- X07 / CLI-23: moe_dogfood.py reads a failed or partial probe -------------------------


def test_cli23_the_last_probe_failed_and_another_is_partly_nonfinite():
    from moe_dogfood import probe_lines

    good = {"histograms": {"a": {}, "b": {}}, "attention": {
        "layers.0": {"max_logit": [3.0, "inf"], "entropy_share": ["nan", 0.5], "masked_rows": 0, "invalid_rows": 2},
        "layers.1": {"max_logit": ["nan"], "entropy_share": ["nan", "nan"], "masked_rows": 1, "invalid_rows": 0},
    }}
    lines = probe_lines([good, {"error": "OutOfMemoryError()"}])
    text = "\n".join(lines)
    assert "1 failed" in text and "OutOfMemoryError()" in text
    assert "2 histograms, attention in 2 layers" in text
    assert "layers.0: largest logit 3.00, entropy 0.50 of maximum" in text
    assert "layers.1: largest logit n/a, entropy n/a of maximum" in text


def test_cli23_every_probe_failed():
    from moe_dogfood import probe_lines

    lines = probe_lines([{"error": "x"}])
    assert any("no probe pass succeeded" in line for line in lines)


# ---- X08 / CLI-16: toy_moe.py names its evidence as the readers pair it -------------------


def child_env():
    env = dict(os.environ)
    env.pop("TENSORQUILL_SERVER", None)
    return env


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    cwd = tmp_path_factory.mktemp("toy")
    done = subprocess.run([sys.executable, str(EXAMPLES / "toy_moe.py")], cwd=cwd, env=child_env(),
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    printed = re.search(r"^evidence -> (.+)$", done.stdout, re.M)
    assert printed, done.stdout
    return cwd, cwd / printed.group(1)


def test_cli16_the_printed_evidence_is_paired_with_its_capture_by_runs(toy_run):
    cwd, evidence = toy_run
    assert evidence.name == "toy-moe.rank0.evidence.json" and evidence.is_file()
    (row,) = tensorquill.list_runs(cwd / "tensorquill-out")
    assert row["run_id"] == "toy-moe"
    assert row["evidence"]["state"] != "missing", row["evidence"]
    assert row["incidents"], row


def test_cli16_the_printed_evidence_reads_its_capture_through_the_mcp_server(toy_run):
    cwd, evidence = toy_run
    series = mcp_server.get_series(mcp_server.Source(evidence), {"name": "router"})
    assert series["records"] == str(cwd / "tensorquill-out" / "toy-moe.rank0.jsonl")


def test_x08_smoke_checks_the_paired_name_through_the_cli_and_the_mcp_server():
    smoke = (SDK / "packaging" / "smoke.sh").read_text(encoding="utf-8")
    assert "toy-moe.rank0.evidence.json" in smoke and "toy-moe.evidence.json" not in smoke
    assert re.search(r"tensorquill\"? runs", smoke) and "tensorquill-mcp" in smoke and "get_series" in smoke
    assert subprocess.run(["bash", "-n", str(SDK / "packaging" / "smoke.sh")]).returncode == 0


# ---- CLI-19: seeded_failures.py writes runs that `tensorquill runs <out>` lists -----------


def test_cli19_runs_lists_a_case_in_the_folder_given_as_out(tmp_path):
    pytest.importorskip("transformers")
    import seeded_failures as sf

    out = tmp_path / "out"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        # The healthy verdict needs its checks made (lane Y5, XC1), so the shortest run that can pass.
        sf.main(["--toy", "--out", str(out), "--case", "healthy", "--steps", str(sf.HEALTHY_MIN_STEPS)])
    rows = tensorquill.list_runs(out)
    assert [row["run_id"] for row in rows] == ["seeded-healthy"]
    assert rows[0]["project"] == sf.PROJECT and rows[0]["evidence"]["state"] != "missing", rows[0]
    assert f"python -m tensorquill runs {out}" in buffer.getvalue()


# ---- X09: the Megatron example records Megatron's own update decision ---------------------


def megatron_example() -> str:
    guide = (SDK / "RUN_ON_YOUR_JOB.md").read_text(encoding="utf-8")
    section = guide.split("### Megatron-LM", 1)[1].split("\n### ", 1)[0]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def test_x09_a_step_megatron_did_not_apply_is_recorded_as_such():
    sink = tensorquill.MemorySink()
    model = torch.nn.Linear(4, 4)
    model(torch.randn(2, 4)).sum().backward()
    shim = SimpleNamespace(init=lambda *a, **k: tensorquill.init(*a, sink=sink, every=1, **k))
    taken = []

    class Optimizer:
        param_groups = [{"lr": 1e-4}]

        def step(self):
            taken.append(True)
            return False, torch.tensor(7.0), None  # Megatron: found Inf or NaN, update skipped

    names = {"tensorquill": shim, "model_chunk": model, "routers": None, "iteration": 12,
             "optimizer": Optimizer(), "loss": torch.tensor(2.0), "mp_group": None,
             "logical_and_across_model_parallel_group": lambda value, group: value,
             "reduce_max_stat_across_model_parallel_group": lambda value, group: value}
    exec(compile(megatron_example(), "RUN_ON_YOUR_JOB.md", "exec"), names)
    names["session"].close()
    (record,) = sink.records
    assert taken and record["step"] == 12 and record["clip_grad_norm"] == 7.0
    assert record["scalars"]["megatron/update_successful"] == 0


# ---- X10: capture_rerun.py takes an explicit step 0 ---------------------------------------


def capture_rerun():
    spec = importlib.util.spec_from_file_location("capture_rerun", REPO / "experiments" / "moe-bakeoff" / "capture_rerun.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # needs no rerun-sdk until main() runs
    return module


def test_x10_an_explicit_step_zero_is_not_the_default():
    evidence = {"samples": [{"step": s} for s in range(0, 60, 10)],
                "incidents": [{"kind": "moe_imbalance", "step": 30}]}
    selected = capture_rerun().selected_steps
    assert selected(evidence, None, None) == (30, 50)
    assert selected(evidence, 0, None) == (0, 50)
    assert selected(evidence, None, 0) == (30, 0)
    assert selected(evidence, 0, 0) == (0, 0)


# ---- X11 and CLI-17: CI covers what the package declares, and lints the cloud stores ------


def ci() -> str:
    return (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_x11_ci_runs_python_310_at_the_torch_floor_and_torch_29():
    text = ci()
    assert re.search(r'- python: "3\.10"\s*\n\s*torch: "2\.6"', text)
    assert re.search(r'- python: "3\.\d+"\s*\n\s*torch: "2\.9"', text)
    pyproject = (SDK / "pyproject.toml").read_text(encoding="utf-8")
    # The jobs test what `requires-python` already declares. No 3.10 classifier until a 3.10 job has passed.
    assert 'requires-python = ">=3.10"' in pyproject


def test_cli17_clippy_runs_with_the_default_features_too():
    lines = [line for line in ci().splitlines() if "cargo clippy" in line]
    assert any("--no-default-features" in line for line in lines)
    assert any("--no-default-features" not in line and "-D warnings" in line for line in lines), lines
