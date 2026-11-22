"""Lane AI-1: probe and head coverage counts (`examples/gpu_probe_coverage.py`).

The per-pass judge is shown failing on a block, a head, a row count and head positions that fall short of
the counts taken from the model, and on a model that makes no SDPA call. Then the toy Llama runs on CPU:
every probe pass covers every block, layer, head and position the model itself has. A CPU run checks the
harness, not the GPU claim.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import gpu_probe_coverage as cov  # noqa: E402

LAYERS, HEADS, Q = 2, 4, 8
EXPECTED = {"blocks": ["model.layers.0", "model.layers.1"], "layers": LAYERS, "heads": HEADS, "vocab": 16,
            "positions": 2 * Q, "attention_calls": [{"heads": HEADS, "query_len": Q, "key_len": Q, "batch": 2}] * LAYERS}


def probe():
    entry = {"max_logit": [1.0] * HEADS, "entropy": [0.5] * HEADS, "entropy_share": [0.9] * HEADS,
             "rows": 2 * HEADS * Q, "query_len": Q, "key_len": Q, "masked_rows": 0, "invalid_rows": 0}
    return {"histograms": {b: {} for b in EXPECTED["blocks"]},
            "attention": {b: copy.deepcopy(entry) for b in EXPECTED["blocks"]},
            "head": {"positions": 2 * Q, "nonfinite_positions": 0, "vocab": 16}}


def failing(p, expected=EXPECTED):
    checks, _ = cov.check_probe(49, p, expected)
    return [c["name"] for c in checks if not c["ok"]]


def test_a_pass_that_covers_everything_the_model_has_passes():
    checks, coverage = cov.check_probe(49, probe(), EXPECTED)
    assert all(c["ok"] for c in checks)
    assert coverage["heads_with_finite_max_logit"] == "8/8" and coverage["head_positions_read"] == "16/16"


def test_each_shortfall_is_its_own_failure():
    p = probe(); del p["histograms"]["model.layers.1"]
    assert failing(p) == ["a histogram for every decoder block"]
    p = probe(); p["attention"]["model.layers.0"]["max_logit"].pop()
    assert failing(p) == ["every attention entry has one value per head and the counted rows"]
    p = probe(); p["attention"]["model.layers.1"]["rows"] = HEADS * Q
    assert failing(p) == ["every attention entry has one value per head and the counted rows"]
    p = probe(); p["head"]["positions"] = Q
    assert failing(p) == ["the head's numbers cover every position of the probe batch"]
    p = probe(); p["head"] = {"not_read": "no head"}
    assert "the head's numbers cover every position of the probe batch" in failing(p)


def test_non_finite_and_overflowed_head_positions_still_count_as_covered():
    p = probe(); p["head"].update(positions=10, nonfinite_positions=4, overflow_positions=2)
    assert failing(p) == []


def test_a_model_that_makes_no_sdpa_call_fails_rather_than_passing_on_zero():
    p = probe(); p["attention"] = {}
    assert failing(p, {**EXPECTED, "attention_calls": []}) == ["attention entries for every decoder layer"]


def test_the_toy_run_covers_every_block_head_and_position(tmp_path):
    pytest.importorskip("transformers")
    out = tmp_path / "cov"
    assert cov.main(["--toy", "--steps", "100", "--probe-every", "50", "--out", str(out)]) == 0
    result = json.loads((out / "coverage.json").read_text())
    assert result["status"] == "pass" and result["expected"]["attention_calls"] == 4
    assert sorted(result["coverage_by_probe_step"]) == ["49", "99"]
    for c in result["coverage_by_probe_step"].values():
        assert c["heads_with_finite_max_logit"] == "16/16" and c["head_positions_read"] == "64/64"
