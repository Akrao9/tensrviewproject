"""Lane AO-4 (wave AO): the two detector engines agree on two records the Rust side misread.

- BUG-01: a router's counts are signed 64-bit integers, and Python sums them as integers that do not
  overflow. The Rust detectors summed them in `i64`: past its range a debug build panicked and a
  release build wrapped, so a router that routed tokens read as one that routed none (`moe_routing`,
  degraded) and a window's shares were wrong. They now sum exactly.
- BUG-10: a probe whose `attention` is not an object failed the whole probe's parse in Rust, so its
  `error` and `not_run` were lost (`not_configured`, healthy) where Python reads them (`unavailable`,
  degraded). Rust now reads such an `attention` as none and the rest of the probe as written.

Each record is judged by both engines through `build_evidence` (`detectors="python"` and `"rust"`)
and the parts the findings touch are compared. The binary is `TENSORQUILL_SERVER_BIN` (or one beside
the interpreter), built from this tree; without it, or without pyarrow, every test here is skipped.
The Rust side's own tests are in `server/src/detect/wave_ao_tests.rs`.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os

import pytest

from tensorquill import server_detect
from tensorquill.evidence import build_evidence

BINARY = server_detect.find_binary()

# Marks, not a module-level skip, so that a run of this file on CI's plain jobs (no pyarrow, no
# binary) collects every test and skips each.
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow is not installed"),
    pytest.mark.skipif(BINARY is None or not os.path.isfile(BINARY),
                       reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)"),
]

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def _moe(steps: list[list[int]], top_k: int) -> list[dict]:
    return [{"run_id": "r", "rank": 0, "step": step, "moe": {"router": counts}, "moe_top_k": {"router": top_k}}
            for step, counts in enumerate(steps)]


def _both(records: list[dict], **options) -> tuple[dict, dict]:
    python = build_evidence(records, detectors="python", **options)
    rust = build_evidence(records, detectors="rust", server_binary=BINARY, **options)
    return python, rust


def _verdict(evidence: dict) -> dict:
    """What the findings change: the run's state, the incidents, the MoE and attention kinds' values,
    and the `not_checked` entries."""
    checks = evidence["kind_checks"]
    return {
        "state": evidence["run_state"]["state"],
        "incidents": [(i["kind"], i.get("step"), i.get("overloaded"), i.get("dead"), i.get("experts"))
                      for i in evidence["incidents"]],
        "kinds": {kind: checks[kind]["value"] for kind in
                  ("moe_imbalance", "starved_experts", "attention_collapse", "attention_uniform",
                   "attention_logit_growth")},
        "not_checked": sorted(json.dumps(entry, sort_keys=True) for entry in evidence.get("not_checked", [])),
    }


SAMPLE_KEYS = ("step", "layer", "router", "routed_tokens", "counts", "diversity", "overloaded", "dead",
               "missing", "overload_rule", "starved", "starved_shares")


def _samples(evidence: dict) -> list[dict]:
    return [{key: sample.get(key) for key in SAMPLE_KEYS} for sample in evidence["samples"]]


# Counts whose sums pass the range of i64: a step's (2**63, -2**64, 2**64), the starved window's
# (three steps of 2**62 + 1, with a window of 3), and a top-k router's baseline (ten steps of 2**62).
# Every count is a power of two or 1, so each share is exact in both engines.
MOE_CASES = {
    "a step past i64::MAX": (_moe([[2**62, 2**62]], 1), {}),
    "a step below i64::MIN": (_moe([[INT64_MIN, INT64_MIN]], 1), {}),
    "a step of 2**64": (_moe([[INT64_MAX, INT64_MAX, 2]], 1), {}),
    "the starved window": (_moe([[2**62, 1]] * 3, 1), {"starve_window": 3}),
    "the top-k baseline": (_moe([[2**59] * 8] * 10 + [[2**61, 2**61, 0, 0, 0, 0, 0, 0]], 4), {}),
}


@pytest.mark.parametrize("name", list(MOE_CASES))
def test_moe_counts_summing_past_i64_are_judged_alike(name: str) -> None:
    records, options = MOE_CASES[name]
    python, rust = _both(records, **options)
    assert _verdict(rust) == _verdict(python)
    assert _samples(rust) == _samples(python)


def test_the_sums_are_what_python_sums() -> None:
    # The findings' record: Python says healthy, and so does Rust now.
    records, options = MOE_CASES["a step past i64::MAX"]
    python, rust = _both(records, **options)
    assert (python["run_state"]["state"], rust["run_state"]["state"]) == ("healthy", "healthy")
    assert rust["samples"][0]["routed_tokens"] == 2**63
    _, rust = _both(MOE_CASES["a step below i64::MIN"][0])
    assert rust["samples"][0]["routed_tokens"] == -(2**64) and rust["samples"][0]["missing"] is True
    _, rust = _both(*MOE_CASES["the starved window"][:1], **MOE_CASES["the starved window"][1])
    assert rust["samples"][2]["starved"] == [2]
    _, rust = _both(MOE_CASES["the top-k baseline"][0])
    assert [i["kind"] for i in rust["incidents"]] == ["moe_imbalance"]


def _shares_apart(evidence: dict) -> tuple[dict, list[float]]:
    """The verdict with each starved expert's `window_share` taken out, and the shares."""
    verdict, shares = _verdict(evidence), []
    for incident in evidence["incidents"]:
        for expert in incident.get("experts") or []:
            shares.append(expert["window_share"])
    verdict["incidents"] = [(kind, step, over, dead, experts and [
        {key: value for key, value in expert.items() if key != "window_share"} for expert in experts])
        for kind, step, over, dead, experts in verdict["incidents"]]
    return verdict, shares


def test_counts_within_i64_are_judged_as_before() -> None:
    # Large counts whose every sum fits: the same verdict and sums in both engines; no byte of Rust's
    # changed for such counts (checked against the binary of 453f1e2 when the fix was made). A share
    # of sums above 2**53 can differ in its last bit, as it did before: Python divides the exact
    # integers, Rust each one's nearest float.
    steps = [[2**55 + 3 * s, 2**55 - s, 2**54, 7] for s in range(60)]
    python, rust = _both(_moe(steps, 1))
    (rust_verdict, rust_shares), (python_verdict, python_shares) = _shares_apart(rust), _shares_apart(python)
    assert rust_verdict == python_verdict
    assert len(rust_shares) == len(python_shares) == 1
    assert all(math.isclose(a, b, rel_tol=1e-15) for a, b in zip(rust_shares, python_shares))
    assert [s["routed_tokens"] for s in rust["samples"]] == [sum(c) for c in steps]


PROBES = {
    "failed, attention null": {"error": "CUDA OOM", "attention": None},
    "failed, attention a list": {"error": "CUDA OOM", "attention": [1]},
    "failed, attention a number": {"error": "CUDA OOM", "attention": 3},
    "failed, attention text": {"error": "CUDA OOM", "attention": "x"},
    "failed, attention a bool": {"error": "CUDA OOM", "attention": True},
    "not run, attention null": {"not_run": "sharded model", "attention": None},
    "not failed, attention null": {"attention": None},
    "not failed, attention a list": {"attention": [{"blocks.0.attn": {"entropy_share": [0.6]}}]},
}


@pytest.mark.parametrize("name", list(PROBES))
def test_a_probe_whose_attention_is_not_an_object_is_judged_alike(name: str) -> None:
    records = [{"run_id": "r", "rank": 0, "step": step, "probe": PROBES[name]} for step in range(3)]
    python, rust = _both(records)
    assert _verdict(rust) == _verdict(python)


def test_the_findings_probe_is_degraded_in_both() -> None:
    records = [{"run_id": "r", "rank": 0, "step": 0, "probe": {"error": "CUDA OOM", "attention": None}}]
    python, rust = _both(records)
    for evidence in (python, rust):
        assert evidence["run_state"]["state"] == "degraded"
        assert evidence["kind_checks"]["attention_collapse"]["value"] == "unavailable"


def test_a_probe_with_attention_is_read_as_before() -> None:
    good = {"attention": {"blocks.0.attn": {"entropy_share": [0.6, 0.62], "max_logit": [2.0, 2.1]}}}
    records = [{"run_id": "r", "rank": 0, "step": step, "probe": good} for step in range(5)]
    python, rust = _both(records)
    assert _verdict(rust) == _verdict(python)
    assert rust["kind_checks"]["attention_collapse"]["value"] == "checked"
