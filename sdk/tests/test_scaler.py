import pytest
import torch
from torch import nn

import tensorlens


def _run(scaler_steps, overflow_at, auto_step=True):
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    sink = tensorlens.MemorySink()
    session = tensorlens.init(model, sink=sink, capture="full", every=2, scaler=scaler, auto_step=auto_step)
    for i in range(scaler_steps):
        opt.zero_grad()
        scaler.scale(model(torch.ones(2, 4)).sum()).backward()
        if i in overflow_at:
            model.weight.grad[0, 0] = float("inf")
        scaler.step(opt)
        scaler.update()
        if not auto_step:
            session.step()
    session.close()
    return sink.records, scaler


def test_skipped_steps_are_counted_and_marked():
    records, scaler = _run(4, overflow_at={1})
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    assert [r["step_skipped"] for r in records] == [False, True, False, False]
    assert [r["grad_scale"] for r in records] == [1024.0, 1024.0, 512.0, 512.0]
    assert "update" not in vars(scaler)  # close() removed the wrapper
    evidence = tensorlens.build_evidence(records)
    assert not [i for i in evidence["incidents"] if i["kind"] == "non_finite"]


def test_manual_steps_still_get_the_scaler_fields():
    records, _ = _run(3, overflow_at={0}, auto_step=False)
    assert [r["step_skipped"] for r in records] == [True, False, False]


def test_loss_scale_trouble_detector():
    base = [{"run_id": "s", "step": s, "step_skipped": False, "grad_scale": 512.0} for s in range(30)]
    assert tensorlens.detect_loss_scale_trouble(base) is None
    for r in base[5:8]:  # a short run of skips is normal
        r["step_skipped"] = True
    assert tensorlens.detect_loss_scale_trouble(base) is None
    for r in base[10:22]:
        r["step_skipped"] = True
    hit = tensorlens.detect_loss_scale_trouble(base)
    assert hit["step"] == 19 and hit["reason"] == "skips" and hit["skipped_steps"] == 15
    collapsed = [{"run_id": "s", "step": 0, "step_skipped": False, "grad_scale": 0.5}]
    assert tensorlens.detect_loss_scale_trouble(collapsed)["reason"] == "scale"
    summary = tensorlens.incident_summary(tensorlens.build_evidence(base))
    assert "Loss scaling in trouble at step 19" in summary
