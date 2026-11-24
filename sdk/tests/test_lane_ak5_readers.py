"""Lane AK-5: the readers given records with late scalars on them.

A late value goes into a step's own record (the newest step still in memory): no record key is
added and no step is written twice, so neither the Python evidence nor the Rust detectors should
see anything new but the value itself. A gradient spike is made at one step and an evaluation is
logged right after it, under `phase("eval")`: both readers find the spike on that step with the
evaluation's value among its scalars, and neither reports `repeated_steps`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill.sink import JsonlSink

from conftest import built_server_binary

SPIKE = 34


def _run(path: Path, every: int) -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))
    session = tensorquill.init(model, sink=JsonlSink(path), run_id="ak5-readers", every=every, heartbeat_seconds=None,
                               stack_seconds=None)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    for step in range(40):
        loss = model(torch.randn(16, 8)).pow(2).mean() * (1e4 if step == SPIKE else 1.0)
        session.log({"train/loss": loss})
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step in (SPIKE, 20):
            with session.phase("eval"):
                session.log({"eval/loss": 0.5 + step})
    session.log({"eval/final": 7.0})  # pending at close: on step 39
    session.close()


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("every", [5, 7])  # step 34 the step kept back (every=5) or a ring row (every=7)
def test_the_python_evidence_reads_the_late_values_on_the_incidents_step(tmp_path, every):
    path = tmp_path / "run.jsonl"
    _run(path, every)
    records = _records(path)
    assert [r["step"] for r in records] == list(range(40))
    assert records[SPIKE]["scalars"]["eval/loss"] == 0.5 + SPIKE and records[39]["scalars"]["eval/final"] == 7.0
    evidence = tensorquill.build_evidence(path, detectors="python")
    kinds = {i["kind"] for i in evidence["incidents"]}
    assert "repeated_steps" not in kinds
    spike = [i for i in evidence["incidents"] if i.get("step") == SPIKE and "eval/loss" in (i.get("scalars") or {})]
    assert spike, evidence["incidents"]
    assert spike[0]["scalars"]["eval/loss"]["value"] == 0.5 + SPIKE
    assert f"eval/loss {0.5 + SPIKE:g}" in tensorquill.incident_summary(evidence)


@pytest.mark.skipif(built_server_binary() is None, reason="tensorquill-server is not built in this worktree")
def test_the_rust_report_reads_the_late_values_on_the_incidents_step(tmp_path):
    pytest.importorskip("pyarrow.flight")  # the Rust detectors are given Arrow: the `flight` extra
    binary = built_server_binary()
    assert Path(__file__).resolve().parents[2] in binary.parents  # this worktree's own build
    path = tmp_path / "run.jsonl"
    _run(path, 5)
    evidence = tensorquill.build_evidence(path, detectors="rust", server_binary=binary)
    assert evidence.get("detectors") == "rust"  # judged by the Rust detectors, not the Python fallback
    kinds = {i["kind"] for i in evidence["incidents"]}
    assert "repeated_steps" not in kinds
    spike = [i for i in evidence["incidents"] if i.get("step") == SPIKE and "eval/loss" in (i.get("scalars") or {})]
    assert spike, evidence["incidents"]
    assert spike[0]["scalars"]["eval/loss"]["value"] == 0.5 + SPIKE
