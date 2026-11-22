"""Lane AI-1: the paired BF16 and FP32 run (`examples/seeded_failures.py --case paired_dtype`).

The judge is shown failing on a recorded loss or gradient norm that is not the script's own, and on a
step missing from a capture. Then the toy pair runs on CPU: two captures from the same float32 weights on
the same batches, every step's recorded loss and gradient norm equal to the script's, and the result file
saying what each arm recorded and where they differ. A CPU run checks the harness, not the GPU claim.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import seeded_failures as sf  # noqa: E402


def arm(losses, norms, recorded_loss=None, recorded_norm=None, steps=None):
    steps = range(len(losses)) if steps is None else steps
    records = [{"step": s, "scalars": {"train/loss": (recorded_loss or losses)[s]}, "grad_norm": (recorded_norm or norms)[s]}
               for s in steps]
    return {"dtype": "float32", "records": records, "losses": losses, "norms": norms}


def oks(checks):
    return [c["ok"] for c in checks]


def test_recorded_values_equal_to_the_scripts_own_pass():
    assert all(oks(sf.recorded_against_own(arm([2.0, 1.9], [3.0, 2.5]), 2)))


def test_a_recorded_loss_or_gradient_norm_that_is_not_the_scripts_fails_and_names_the_step():
    checks = sf.recorded_against_own(arm([2.0, 1.9], [3.0, 2.5], recorded_loss=[2.0, 1.8]), 2)
    assert oks(checks) == [True, False, True] and "first step 1" in checks[1]["detail"]
    checks = sf.recorded_against_own(arm([2.0, 1.9], [3.0, 2.5], recorded_norm=[3.01, 2.5]), 2)
    assert oks(checks) == [True, True, False] and "first step 0" in checks[2]["detail"]


def test_a_step_missing_from_the_capture_fails():
    assert oks(sf.recorded_against_own(arm([2.0, 1.9, 1.8], [3.0, 2.5, 2.0], steps=[0, 2]), 3))[0] is False


def test_the_pair_refuses_a_single_dtype(tmp_path):
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        sf.main(["--case", "paired_dtype", "--toy", "--dtype", "float32", "--out", str(tmp_path / "p")])


def test_the_toy_pair_runs_on_cpu_and_says_what_each_recorded_and_where_they_differ(tmp_path):
    pytest.importorskip("transformers")
    out = tmp_path / "paired"
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        sf.main(["--case", "paired_dtype", "--toy", "--steps", "200", "--out", str(out)])
    assert "All cases passed" in printed.getvalue()
    result = json.loads((out / "paired_dtype.json").read_text())
    assert result["status"] == "pass" and all(c["ok"] for c in result["checks"])
    assert len(result["checks"]) == 7
    assert result["initial_float32_weights_sha256_16"] and result["batches_sha256_16"]
    for name in ("float32", "bfloat16"):
        assert (out / f"paired-{name}.jsonl").exists() and result["recorded"][name]["steps"] == 200
    diff = result["differences"]
    # The toy's BF16 RMSNorm weights never move at lr 1e-5 (the script's docstring): a difference, reported.
    assert "stalled_weights" in diff["incident_kinds_only_in_bfloat16"]
    assert all("norm" in name for name in diff["stalled_only_in_bfloat16"])
    assert diff["loss_gap_first"] < sf.PAIRED_START
