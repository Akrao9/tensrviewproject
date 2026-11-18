"""The size of the optimizer's update (`init(..., updates=True)`, `update_rms`) and `update_spike`.

The capture reads Adam's update to each parameter from the optimizer's own state after the step.
The rule is Altenbernd et al.'s (arXiv 2604.00726, eq. 13): the largest RMS update moved from
the step before by more than `baseline / (0.05 * sqrt(rise))`, where `baseline` is the mean of
that change over the last `update_window` steps that were not flagged and `rise` is how far the
total gradient norm rose. Here: the number recorded is the step PyTorch applied, on every path Adam takes; the
optimizer patterns the capture already handles; an optimizer that cannot be read is said, never
silent; the rule's edges; a seeded fault on a CPU toy; and the same answers from the server.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import evidence as ev

DETECT_RS = Path(__file__).resolve().parents[2] / "server" / "src" / "detect.rs"
NEEDS = "update_rms with grad_norm or clip_grad_norm (updates=True)"


def spikes(evidence: dict) -> list[dict]:
    return [i for i in evidence["incidents"] if i["kind"] == "update_spike"]


def entry(evidence: dict) -> dict | None:
    return next((n for n in evidence.get("not_checked", []) if n["check"] == "update_size"), None)


# --- the number recorded is the step that was applied ------------------------------------------

ADAM, ADAMW = torch.optim.Adam, torch.optim.AdamW
PATHS = {
    "adam, single tensor": lambda p: ADAM(p, lr=1e-2, foreach=False),
    "adam, foreach": lambda p: ADAM(p, lr=1e-2, foreach=True),
    "adam, fused": lambda p: ADAM(p, lr=1e-2, fused=True),
    "adam, amsgrad": lambda p: ADAM(p, lr=1e-2, amsgrad=True, foreach=False),
    "adam, amsgrad foreach": lambda p: ADAM(p, lr=1e-2, amsgrad=True, foreach=True),
    "adamw, single tensor": lambda p: ADAMW(p, lr=1e-2, weight_decay=0.1, foreach=False),
    "adamw, foreach": lambda p: ADAMW(p, lr=1e-2, weight_decay=0.1, foreach=True),
    "adamw, fused": lambda p: ADAMW(p, lr=1e-2, weight_decay=0.1, fused=True),
    "adamw, amsgrad fused": lambda p: ADAMW(p, lr=1e-2, weight_decay=0.1, fused=True, amsgrad=True),
    "adamw, a tensor learning rate": lambda p: ADAMW(p, lr=torch.tensor(1e-2), weight_decay=0.0, foreach=False),
    "adamw, two groups": lambda p: ADAMW([{"params": p[:2], "lr": 3e-3}, {"params": p[2:]}], lr=1e-2, weight_decay=0.0),
}


def small() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("clip", [None, 0.01])
def test_the_recorded_size_is_the_step_adam_applied_with_its_weight_decay_taken_out(path, clip):
    model = small()
    opt = PATHS[path](list(model.parameters()))
    sink = tensorquill.MemorySink()
    applied: dict[tuple[int, str], float] = {}
    group_of = {id(p): g for g in opt.param_groups for p in g["params"]}
    with tensorquill.init(model, run_id="t", every=2, sink=sink, updates=True, weights=False) as session:
        for step in range(5):
            opt.zero_grad()
            model(torch.randn(32, 8)).pow(2).mean().backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            opt.step()
            for name, p in model.named_parameters():
                group = group_of[id(p)]
                # AdamW first shrinks the weight by lr * weight_decay; Adam's decay here is 0.
                decayed = before[name] * (1 - float(group["lr"]) * group["weight_decay"]) if isinstance(opt, ADAMW) else before[name]
                applied[(step, name)] = ((decayed - p.detach()).norm() / math.sqrt(p.numel())).item()
    assert session.updates and [r["step"] for r in sink.records] == [0, 1, 2, 3, 4]
    for record in sink.records:
        assert list(record["update_rms"]) == [name for name, _ in model.named_parameters()]
        for name, size in record["update_rms"].items():
            # The weights are float32, so the step read back from them is good to about 1e-5 of itself.
            assert size == pytest.approx(applied[(record["step"], name)], rel=2e-4), (path, record["step"], name)


def test_half_precision_weights_are_read_in_float32():
    model = small().to(torch.bfloat16)
    opt = ADAMW(model.parameters(), lr=1e-2, weight_decay=0.0)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, updates=True, weights=False, layers=()):
        for _ in range(3):
            opt.zero_grad()
            model(torch.randn(16, 8, dtype=torch.bfloat16)).float().pow(2).mean().backward()
            opt.step()
    sizes = [size for r in sink.records for size in r["update_rms"].values()]
    # About the learning rate on the first steps, where Adam's update is near lr * sign(g).
    assert len(sizes) == 12 and all(math.isfinite(s) and 1e-3 < s < 2e-2 for s in sizes), sizes


# --- off unless asked for, and what it turns on ------------------------------------------------


def test_nothing_is_read_unless_asked_for(toy_moe):
    opt = ADAMW(toy_moe.parameters(), lr=1e-3)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(toy_moe, sink=sink, capture="full")
    assert session.updates is False and session._update.numel() == 0 and session._ring_update.numel() == 0
    with pytest.raises(RuntimeError, match="updates=True"):
        session.record_updates(opt)
    for _ in range(2):
        opt.zero_grad()
        toy_moe(torch.randn(4, 8)).sum().backward()
        opt.step()
    session.close()
    assert all("update_rms" not in r and "update_not_read" not in r for r in sink.records)
    evidence = tensorquill.build_evidence(sink.records)
    assert evidence["kind_checks"]["update_spike"] == {
        "value": "not_configured", "records": 0, "records_needed": 102, "enough_history": False, "needs": NEEDS,
    }
    assert entry(evidence) is None


def test_router_only_capture_gains_the_total_the_rule_needs_and_reads_it_from_clipping_when_the_run_clips(toy_moe):
    def run(clip):
        torch.manual_seed(0)
        model = type(toy_moe)()
        opt = ADAMW(model.parameters(), lr=1e-3)
        sink = tensorquill.MemorySink()
        with tensorquill.init(model, sink=sink, routers={"router": 4}, updates=True):
            for _ in range(3):
                opt.zero_grad()
                model(torch.randn(16, 8)).sum().backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        return sink.records

    # No clipping: the per-parameter norms are read on every step, as a weight check reads them.
    plain = run(clip=False)
    assert all(r["grad_norm"] > 0 and r["grad_norms"] and "clip_grad_norm" not in r and r["update_rms"] for r in plain)
    # Clipping seen: its own norm, before clipping, is the total, and nothing more is read.
    clipped = run(clip=True)
    assert all(r["clip_grad_norm"] > 0 and r["grad_norm"] is None and not r["grad_norms"] and r["update_rms"] for r in clipped)
    for records in (plain, clipped):
        check = tensorquill.build_evidence(records)["kind_checks"]["update_spike"]
        assert (check["value"], check["records"], check["enough_history"]) == ("checked", 3, False)


# --- the optimizer patterns the capture handles ------------------------------------------------


class _TwoPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Linear(4, 4, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.matrix(x))


@pytest.mark.parametrize("adam_leads", [True, False])
def test_two_optimizers_on_one_model_are_each_read_and_the_one_that_is_not_adam_is_named(adam_leads):
    torch.manual_seed(0)
    model = _TwoPart()
    adam = ADAMW([model.head.weight, model.head.bias], lr=1e-2)
    sgd = torch.optim.SGD([model.matrix.weight], lr=0.1)
    order = [adam, sgd] if adam_leads else [sgd, adam]
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=2, updates=True, weights=False, layers=()):
        for _ in range(4):
            for opt in order:
                opt.zero_grad()
            model(torch.randn(8, 4)).sum().backward()
            for opt in order:
                opt.step()
    records = sink.records
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    held = ["head.weight", "head.bias"]
    # Led or not, AdamW's update is on the record of the step it belongs to (SDK-06): one that
    # steps after the lead ended the step is written on that step's record, still in the ring.
    assert all(list(r["update_rms"]) == held for r in records)
    # Said once, on the first record of the block being written when it was known.
    assert [sorted(r.get("update_not_read", {})) for r in records] == [["SGD"], [], [], []]
    said = next(r["update_not_read"]["SGD"] for r in records if "update_not_read" in r)
    assert said.startswith("SGD is not Adam or AdamW")
    evidence = tensorquill.build_evidence(records)
    found = entry(evidence)
    assert found == {"check": "update_size", "kinds": ["update_spike"], "optimizers": ["SGD"],
                     "reason": f"SGD: {said}", "step": 0}
    # Read for some parameters and not for others: asked for and not wholly judged.
    check = evidence["kind_checks"]["update_spike"]
    assert (check["value"], check["check"], check["reason"]) == ("unavailable", "update_size", found["reason"])
    assert evidence["run_state"]["state"] == "degraded"


def test_a_step_with_a_closure_is_read_after_the_closure_ran():
    model = nn.Linear(4, 1)
    opt = ADAMW(model.parameters(), lr=1e-2)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", updates=True)

    def closure():
        opt.zero_grad()
        loss = model(torch.ones(2, 4)).sum()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    session.close()
    assert len(sink.records) == 3
    assert all(r["grad_norm"] > 0 and set(r["update_rms"]) == {"weight", "bias"} for r in sink.records)
    assert all(size > 0 for r in sink.records for size in r["update_rms"].values())


def test_a_step_that_raised_is_not_read_and_the_next_ones_are():
    model = nn.Linear(4, 1)

    class Flaky(torch.optim.AdamW):  # AdamW's own step, called after a first call that fails
        calls = 0

    opt = Flaky(model.parameters(), lr=1e-2)
    original = torch.optim.Adam.step

    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, updates=True, weights=False)
    model(torch.ones(1, 4)).sum().backward()

    def failing(closure=None):
        raise RuntimeError("transient")

    with pytest.raises(RuntimeError, match="transient"):
        opt.step(failing)  # the closure raises inside the step: its post hook never runs
    assert session._update_recorded is False and float(session._update.abs().sum()) == 0.0
    for _ in range(2):
        opt.step()
    session.close()
    assert original is torch.optim.Adam.step
    assert [r["step"] for r in sink.records] == [0, 1]
    assert all(set(r["update_rms"]) == {"weight", "bias"} for r in sink.records)
    assert all("update_not_read" not in r for r in sink.records)


def _scaled(make, overflow_at, steps=4):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = make(model.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, capture="full", every=2, scaler=scaler, updates=True, weights=False):
        for i in range(steps):
            opt.zero_grad()
            scaler.scale(model(torch.randn(2, 4)).sum()).backward()
            if i in overflow_at:
                model.weight.grad[0, 0] = float("inf")
            scaler.step(opt)
            scaler.update()
    return sink.records


def test_a_step_the_loss_scaler_skipped_has_no_update_and_is_not_called_unread():
    records = _scaled(lambda p: ADAMW(p, lr=1e-2), overflow_at={1})
    assert [r["step_skipped"] for r in records] == [False, True, False, False]
    # The scaler never called the optimizer on step 1, so there was no update to read.
    assert ["update_rms" in r for r in records] == [True, False, True, True]
    assert all("update_not_read" not in r for r in records)
    evidence = tensorquill.build_evidence(records)
    assert entry(evidence) is None and evidence["kind_checks"]["update_spike"]["value"] == "checked"


def test_a_fused_step_the_scaler_skipped_reads_an_unchanged_state_and_the_rule_leaves_it_out():
    try:
        records = _scaled(lambda p: ADAMW(p, lr=1e-2, fused=True), overflow_at={0, 2})
    except RuntimeError as error:  # a PyTorch whose fused CPU step does not take the scaler
        pytest.skip(str(error))
    assert [r["step_skipped"] for r in records] == [True, False, True, False]
    # The fused step ran and took its count back: what is read is finite, and step 2 repeats step 1.
    read = [r.get("update_rms") for r in records]
    assert all(sizes is None or all(math.isfinite(v) for v in sizes.values()) for sizes in read)
    if read[2] is not None:
        assert read[2] == pytest.approx(read[1])
    # Skipped steps are never judged and never the step before: the rule reads the run as if
    # they were not there, and a wild size on one changes nothing.
    kept = [r for r in records if not r["step_skipped"]]
    wild = [{**r, "update_rms": {name: 9.0 for name in r.get("update_rms") or {"weight": 0}}} if r["step_skipped"] else r
            for r in records]
    assert ev.detect_update_spikes(wild, window=1) == ev.detect_update_spikes(kept, window=1)


def test_an_optimizer_that_holds_none_of_the_models_parameters_is_said_once_the_user_steps_by_hand():
    # DeepSpeed ZeRO and Megatron step copies of the weights. With the hooks no step is counted
    # (the capture warns); with `auto_step=False` the steps are written, and the records say
    # why they carry no update size.
    model = nn.Linear(4, 1)
    copy = nn.Parameter(torch.zeros(5))
    optimizer = ADAMW([copy], lr=0.1)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, capture="full", auto_step=False, updates=True, every=2) as session:
        for _ in range(4):
            model(torch.randn(2, 4)).sum().backward()
            copy.grad = torch.ones(5)
            optimizer.step()
            session.record_updates(optimizer)  # holds none of them: nothing to read, nothing raised
            session.step()
    records = sink.records
    assert len(records) == 4 and all("update_rms" not in r for r in records)
    assert [list(r.get("update_not_read", {})) for r in records] == [["auto_step=False"], [], [], []]
    evidence = tensorquill.build_evidence(records)
    assert entry(evidence)["reason"].startswith("auto_step=False: the optimizer hooks are off, so call session.record_updates")
    assert evidence["kind_checks"]["update_spike"]["value"] == "unavailable"


def test_with_the_hooks_off_the_optimizer_is_read_when_it_is_handed_over():
    model = nn.Linear(4, 1)
    opt = ADAMW(model.parameters(), lr=1e-2)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, capture="full", auto_step=False, updates=True) as session:
        for _ in range(3):
            opt.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            session.record_updates(opt)
            session.step()
    assert all(set(r["update_rms"]) == {"weight", "bias"} and "update_not_read" not in r for r in sink.records)


def test_a_scaler_run_whose_optimizer_holds_none_of_the_parameters_says_so():
    model = nn.Linear(4, 1)
    copy = nn.Parameter(torch.zeros(5))
    optimizer = ADAMW([copy], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, capture="full", scaler=scaler, updates=True, every=2):
        for _ in range(2):
            scaler.scale(model(torch.randn(2, 4)).sum()).backward()
            copy.grad = torch.ones(5)
            scaler.step(optimizer)
            scaler.update()  # ends the step: no optimizer of this model was seen
    assert list(sink.records[0]["update_not_read"]) == ["no optimizer"]
    reason = entry(tensorquill.build_evidence(sink.records))["reason"]
    assert "no optimizer that has stepped holds this model's parameters" in reason and "no Adam state to read" in reason


class _Lookahead(torch.optim.AdamW):
    def step(self, closure=None):  # its own step: what it applies is not known to be Adam's
        return super().step(closure)


@pytest.mark.parametrize("make, name, starts", [
    (lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9), "SGD", "SGD is not Adam or AdamW"),
    (lambda p: torch.optim.Adafactor(p, lr=0.01), "Adafactor", "Adafactor is not Adam or AdamW"),
    (lambda p: torch.optim.RMSprop(p, lr=0.01), "RMSprop", "RMSprop is not Adam or AdamW"),
    (lambda p: torch.optim.NAdam(p, lr=0.01), "NAdam", "NAdam is not Adam or AdamW"),
    (lambda p: _Lookahead(p, lr=0.01), "_Lookahead", "_Lookahead changes Adam's step"),
])
def test_any_other_optimizer_is_not_checked_with_its_reason_never_silence(make, name, starts):
    model = nn.Linear(4, 1)
    opt = make(model.parameters())
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, capture="full", updates=True, every=2):
        for _ in range(4):
            opt.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
    records = sink.records
    assert len(records) == 4 and all("update_rms" not in r for r in records)
    # Said once, on the first record written.
    assert [list(r.get("update_not_read", {})) for r in records] == [[name], [], [], []]
    evidence = tensorquill.build_evidence(records)
    found = entry(evidence)
    assert found["kinds"] == ["update_spike"] and found["optimizers"] == [name] and found["step"] == 0
    assert found["reason"].startswith(f"{name}: {starts}")
    check = evidence["kind_checks"]["update_spike"]
    assert check == {"value": "unavailable", "records": 0, "records_needed": 102, "enough_history": False,
                     "needs": NEEDS, "check": "update_size", "reason": found["reason"]}
    assert evidence["run_state"]["state"] == "degraded"
    assert "update_spike" in ev.checked_line(evidence) and "update_spike" in ev.kinds_by_value(evidence)["unavailable"]


def test_a_subclass_that_leaves_the_step_alone_is_read():
    class Mine(torch.optim.AdamW):
        pass

    model = nn.Linear(4, 1)
    opt = Mine(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, updates=True):
        opt.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
    assert set(sink.records[0]["update_rms"]) == {"weight", "bias"}


def test_a_parameter_adam_left_alone_this_step_is_left_out():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    opt = ADAMW(model.parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, updates=True, weights=False, layers=()):
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            if step == 1:  # only the head gets a gradient: Adam does not touch the first layer
                model[1](torch.randn(2, 4)).sum().backward()
            else:
                model(torch.randn(2, 4)).sum().backward()
            opt.step()
    assert [sorted(r["update_rms"]) for r in sink.records] == [
        ["0.bias", "0.weight", "1.bias", "1.weight"], ["1.bias", "1.weight"], ["0.bias", "0.weight", "1.bias", "1.weight"],
    ]


def test_a_read_that_fails_is_said_and_training_goes_on(monkeypatch):
    model = nn.Linear(4, 1)
    opt = ADAMW(model.parameters(), lr=0.01, foreach=True)  # this path never calls `torch.sqrt` itself
    sink = tensorquill.MemorySink()
    real, fail = torch.sqrt, {"now": False}

    def sqrt(*args, **kwargs):
        if fail["now"]:
            raise RuntimeError("out of memory, say")
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "sqrt", sqrt)
    with tensorquill.init(model, sink=sink, updates=True, weights=False, every=2):
        for step in range(4):
            opt.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
            fail["now"] = step == 1
            opt.step()  # nothing is raised into the step
    records = sink.records
    assert [r["step"] for r in records] == [0, 1, 2, 3]  # every step trained and was written
    # Read on the first step, failed on the second, and that optimizer is not read again.
    assert ["update_rms" in r for r in records] == [True, False, False, False]
    said = records[0]["update_not_read"]["AdamW"]
    assert said.startswith("reading its state failed (RuntimeError") and "out of memory, say" in said
    assert tensorquill.build_evidence(records)["kind_checks"]["update_spike"]["value"] == "unavailable"


# --- the rule ----------------------------------------------------------------------------------


def step(at: int, big: float = 0.010, small: float = 0.002, total: float = 1.0, attempt: int = 1, run: str = "updates", **more) -> dict:
    return {"schema_version": 1, "run_id": run, "rank": 0, "attempt": attempt, "step": at, "ts": float(at),
            "grad_norm": total, "update_rms": {"big": big, "small": small}, **more}


def quiet(count: int, start: int = 0, **more) -> list[dict]:
    """The largest update moves 0.001 on every step."""
    return [step(at, big=0.010 + 0.001 * (at % 2), **more) for at in range(start, start + count)]


def test_a_jump_past_the_baseline_is_flagged_only_when_the_gradient_norm_rose():
    records = quiet(6)  # five changes: four make the first baseline, the fifth is judged with no rise
    assert ev.detect_update_spikes(records, window=4) is None
    # The baseline is 0.001, so the limit for a rise of 4 is 0.001 / (0.05 * 2) = 0.01.
    assert ev.detect_update_spikes(records + [step(6, small=0.3, total=0.5)], window=4) is None  # the norm fell
    hit = ev.detect_update_spikes(records + [step(6, small=0.3, total=5.0)], window=4)
    assert (hit["kind"], hit["step"], hit["spike_count"], hit["spike_steps"]) == ("update_spike", 6, 1, [6])
    assert hit["baseline"] == pytest.approx(0.001) and hit["threshold"] == pytest.approx(0.01)
    assert (hit["grad_norm"], hit["grad_norm_before"], hit["grad_rise"]) == (5.0, 1.0, 4.0)
    assert (hit["update_rms"], hit["previous"]) == (0.3, 0.011) and hit["change"] == pytest.approx(0.289)
    # The parameter whose own update moved most is first.
    assert [p["name"] for p in hit["parameters"]] == ["small", "big"]
    assert hit["parameters"][0] == {"name": "small", "update_rms": 0.3, "previous": 0.002, "change": pytest.approx(0.298)}
    # Under the limit: a change of 0.009 with the same rise.
    assert ev.detect_update_spikes(records + [step(6, big=0.020, total=5.0)], window=4) is None
    # A smaller rise raises the limit: 0.001 / (0.05 * sqrt(0.0001)) = 2.
    assert ev.detect_update_spikes(records + [step(6, small=0.3, total=1.0001)], window=4) is None


def test_the_baseline_is_the_last_changes_that_were_not_flagged():
    jump = dict(small=0.3, total=5.0)
    # A jump that lasts: its first step is flagged and stays out of the baseline, so the steps
    # after it are judged against what came before. They do not move, and are not flagged.
    lasting = quiet(6) + [step(at, big=0.011, small=0.3, total=5.0 + at) for at in range(6, 12)]
    hit = ev.detect_update_spikes(lasting, window=4)
    assert (hit["step"], hit["spike_steps"]) == (6, [6])
    # A second jump right after the first is judged by the same baseline: flagged too.
    twice = quiet(6) + [step(6, **jump), step(7, small=0.9, total=9.0)]
    assert ev.detect_update_spikes(twice, window=4)["spike_steps"] == [6, 7]
    # The fall back, with no rise in the norm, is not flagged and joins the baseline. It raises
    # it for the next `window` steps: the same jump two steps later is under the limit then.
    back = quiet(6) + [step(6, **jump), step(7), step(8, **jump)]
    assert ev.detect_update_spikes(back, window=4)["spike_steps"] == [6]
    # Once the window has moved past it, the same jump is flagged again.
    later = quiet(6) + [step(6, **jump), step(7)] + quiet(5, start=8) + [step(13, **jump)]
    assert ev.detect_update_spikes(later, window=4)["spike_steps"] == [6, 13]
    # A baseline that follows the run: changes ten times larger for a window, and a jump that
    # was far over the old limit is under the new one.
    grown = quiet(6) + [step(at, big=0.010 + 0.010 * (at % 2)) for at in range(6, 12)] + [step(12, big=0.010, small=0.08, total=5.0)]
    assert ev.detect_update_spikes(grown, window=4) is None
    assert ev.detect_update_spikes(quiet(12) + [step(12, big=0.010, small=0.08, total=5.0)], window=4)["step"] == 12


def test_what_cannot_be_judged_is_left_out_and_is_never_the_step_before():
    nan = float("nan")
    base = quiet(6)
    jump = dict(small=0.3, total=4.5)
    # A skipped step, a record without sizes, one without a finite total, one whose sizes are
    # not finite: none is judged, and the next record is compared with the last that was.
    for odd in (
        step(6, small=5.0, total=9.0, step_skipped=True),
        {k: v for k, v in step(6, total=9.0).items() if k != "update_rms"},
        step(6, small=5.0, total=nan), step(6, small=5.0, total="inf"),
        step(6, big=nan, small="nan", total=9.0),
    ):
        hit = ev.detect_update_spikes(base + [odd, step(7, **jump)], window=4)
        assert (hit["step"], hit["previous"], hit["grad_norm_before"]) == (7, 0.011, 1.0), odd
        assert ev.detect_update_spikes(base + [odd], window=4) is None, odd
    # One parameter that is not finite is left out; the others are judged.
    hit = ev.detect_update_spikes(base + [step(6, big=nan, **jump)], window=4)
    assert hit["step"] == 6 and [p["name"] for p in hit["parameters"]] == ["small"]
    # A non-finite total is `non_finite`'s to report.
    evidence = tensorquill.build_evidence(base + [step(6, small=5.0, total=nan)], update_window=4)
    assert [i["kind"] for i in evidence["incidents"]] == ["non_finite"]


def test_a_new_attempt_is_not_compared_with_the_one_it_replaced():
    resumed = quiet(6) + [step(6, small=0.3, total=4.5, attempt=2), step(7, small=0.3, total=4.5, attempt=2)]
    assert ev.detect_update_spikes(resumed, window=4) is None
    later = resumed + [step(8, small=0.9, total=8.5, attempt=2)]
    assert ev.detect_update_spikes(later, window=4)["step"] == 8


def test_a_run_that_clips_is_judged_by_the_norm_before_clipping():
    # `grad_norm` is read after clipping and stays flat; the jump is in `clip_grad_norm`.
    clipped = [{**r, "clip_grad_norm": 2.0} for r in quiet(6)] + [step(6, small=0.3, total=1.0, clip_grad_norm=6.0)]
    hit = ev.detect_update_spikes(clipped, window=4)
    assert (hit["step"], hit["grad_norm"], hit["grad_rise"]) == (6, 6.0, 4.0)
    # Once any record has it, a record without it has no total: the steps before clipping began are not judged.
    late = quiet(6) + [{**r, "clip_grad_norm": 2.0} for r in quiet(3, start=6)] + [step(9, small=0.3, total=1.0, clip_grad_norm=6.0)]
    assert ev.detect_update_spikes(late, window=4) is None  # two changes since clipping began: no baseline yet
    assert ev.detect_update_spikes(late, window=2)["step"] == 9


def test_the_baseline_is_the_option_and_the_kind_needs_two_records_more():
    records = quiet(12) + [step(12, small=0.3, total=4.5)]
    for window, found in ((4, True), (11, True), (12, False)):
        evidence = tensorquill.build_evidence(records, update_window=window)
        check = evidence["kind_checks"]["update_spike"]
        assert bool(spikes(evidence)) is found, window
        assert (check["records"], check["records_needed"], check["enough_history"]) == (13, window + 2, found)
        # Off its default, the option is listed with the others and named in the summary.
        assert evidence["thresholds"]["update_window"] == window
        assert f"update_window {window} (default: 100)" in ev.thresholds_line(evidence)
    usual = tensorquill.build_evidence(records)
    assert "thresholds" not in usual and not spikes(usual) and ev.thresholds_line(usual) is None
    assert "update_spike 13 of 102" in ev.checked_line(usual)
    for bad in (0, -1, 2.5, True, None):
        with pytest.raises(ValueError, match="update_window"):
            tensorquill.build_evidence(records, update_window=bad)


def test_the_summary_names_the_step_the_sizes_and_the_parameter():
    evidence = tensorquill.build_evidence(quiet(6) + [step(6, small=0.3, total=4.5)], update_window=4)
    (found,) = spikes(evidence)
    assert found["class"] == "anomaly" and evidence["run_state"]["state"] == "unhealthy"
    line = next(text for text in ev.incident_summary(evidence).splitlines() if "Update-size jump" in text)
    assert line.endswith(
        "Update-size jump at step 6: the largest RMS update moved 0.289 from the step before (0.011 to 0.3), against a usual"
        " 0.001 and a limit of 0.0107 for a gradient norm that rose 3.5, most in small (0.002 to 0.3); 1 step flagged"
    ), line


def test_a_record_counts_when_it_carries_sizes_and_a_total():
    records = [step(0), {**step(1), "grad_norm": None}, {**step(2), "update_rms": {}}, {**step(3), "grad_norm": None, "clip_grad_norm": 2.0},
               {k: v for k, v in step(4).items() if k != "update_rms"}]
    counts = ev.input_counts(records)
    assert (counts["update"], counts["grad_total"], counts["steps"]) == (2, 4, 5)
    assert ev.KIND_INPUTS["update_spike"] == "update" and ev.INPUT_NEEDS["update"] == NEEDS


# --- class and trigger -------------------------------------------------------------------------


def test_it_is_an_anomaly_and_does_not_fire_the_trigger_unless_anomalies_are_asked_for():
    from tensorquill.trigger import Trigger

    assert ev.INCIDENT_CLASSES["update_spike"] == "anomaly" and "update_spike" in ev.SERVER_KINDS
    incident = {"kind": "update_spike", "class": "anomaly", "step": 8}
    assert Trigger("true")._fires(incident) is False
    assert Trigger("true", anomalies=True)._fires(incident) is True
    assert Trigger("true", kinds=["update_spike"])._fires(incident) is True


# --- S3: what equal replicas do not rule out ---------------------------------------------------


def test_a_checked_replica_drift_says_that_equal_replicas_do_not_rule_out_corruption():
    spread = [{"run_id": "r", "step": s, "ts": float(s), "replica_spread": {"w": [0.0, 0.0, 0.0]}} for s in range(3)]
    check = tensorquill.build_evidence(spread)["kind_checks"]["replica_drift"]
    assert check == {"value": "checked", "records": 3, "records_needed": 1, "enough_history": True,
                     "needs": "replica_spread (replica_group=)", "note": ev.KIND_NOTES["replica_drift"]}
    note = check["note"]
    assert note.startswith("equal replicas do not rule out silent data corruption")
    assert "the same wrong result goes back to every rank" in note and "SCOUT, arXiv 2608.11034, pp. 1 and 3" in note
    # Only a checked kind carries it: a run without a replica group says nothing about replicas.
    plain = tensorquill.build_evidence([{"run_id": "r", "step": 0, "ts": 0.0, "grad_norm": 1.0}])["kind_checks"]
    assert plain["replica_drift"]["value"] == "not_configured" and all("note" not in c for c in plain.values())
    unavailable = ev.kind_checks({**dict.fromkeys(ev.INPUT_NEEDS, 0), "replica_spread": 3},
                                 [{"check": "x", "kinds": ["replica_drift"], "reason": "why"}])
    assert "note" not in unavailable["replica_drift"]
    assert list(ev.KIND_NOTES) == ["replica_drift"]


def test_the_servers_note_is_this_note():
    source = DETECT_RS.read_text(encoding="utf-8")
    found = re.search(r'pub const KIND_NOTES: \[\(&str, &str\); 1\] = \[\(\s*"replica_drift",\s*"(.*?)",\s*\)\];', source, re.S)
    assert found, "KIND_NOTES is not in server/src/detect.rs in the shape this test reads"
    assert re.sub(r"\\\n\s*", "", found.group(1)) == ev.KIND_NOTES["replica_drift"]


# --- a seeded fault on a CPU toy ---------------------------------------------------------------


class Toy(nn.Module):
    """Two transformer blocks over 16 tokens, 32 wide."""

    def __init__(self, width: int = 32, vocab: int = 64, length: int = 16) -> None:
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, width), nn.Embedding(length, width)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.0, batch_first=True, norm_first=True) for _ in range(2)
        )
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.tok(tokens) + self.pos(torch.arange(tokens.shape[1]))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


SEEDED = "blocks.1.linear1.weight"
WINDOW, FAULT_STEP, TOY_STEPS = ev.UPDATE_WINDOW, 180, 260


def toy_run(seed: int, *, clip: float | None = None, fault: float | None = None, make=None) -> list[dict]:
    """Train the toy with AdamW on a small fixed set of sequences. `fault` multiplies the seeded
    parameter's gradient on one step, before clipping, as a corrupted gradient would arrive."""
    torch.manual_seed(seed)
    model = Toy()
    opt = (make or (lambda p: ADAMW(p, lr=1e-3)))(model.parameters())
    draw = torch.Generator().manual_seed(seed + 100)
    data = torch.randint(0, 64, (64, 17), generator=draw)
    seeded = dict(model.named_parameters())[SEEDED]
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, run_id=f"toy-{seed}", every=50, sink=sink, updates=True, weights=False, layers=()):
        for at in range(TOY_STEPS):
            batch = data[torch.randint(0, 64, (16,), generator=draw)]
            opt.zero_grad()
            nn.functional.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten()).backward()
            if fault is not None and at == FAULT_STEP:
                seeded.grad.mul_(fault)
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    return sink.records


def margin(records: list[dict], window: int = WINDOW) -> float:
    """The largest change over its limit on any judged step, by the rule as `detect_update_spikes`
    applies it: a step is flagged above 1, and a flagged change stays out of the baseline."""
    key = "clip_grad_norm" if any("clip_grad_norm" in r for r in records) else "grad_norm"
    worst, recent, before = 0.0, [], None
    for record in records:
        largest, total = max(record["update_rms"].values()), record[key]
        if before is not None:
            change, rise = abs(largest - before[0]), max(0.0, total - before[1])
            if len(recent) < window:
                recent.append(change)
            else:
                over = change * ev.UPDATE_ALPHA * math.sqrt(rise) / (math.fsum(recent[-window:]) / window)
                worst = max(worst, over)
                if over <= 1:
                    recent.append(change)
        before = (largest, total)
    return worst


@pytest.mark.parametrize("clip", [None, 1.0], ids=["no clipping", "clipping at 1"])
def test_clean_toy_runs_stay_silent_over_several_seeds(clip):
    for seed in range(4):
        records = toy_run(seed, clip=clip)
        evidence = tensorquill.build_evidence(records, update_window=WINDOW)
        check = evidence["kind_checks"]["update_spike"]
        assert (check["value"], check["records"], check["enough_history"]) == ("checked", TOY_STEPS, True)
        assert spikes(evidence) == [] and entry(evidence) is None, (seed, clip)
        # Not a near miss: the worst healthy step is well under the limit.
        assert margin(records) < 0.5, (seed, clip, margin(records))


@pytest.mark.parametrize("clip", [None, 1.0], ids=["no clipping", "clipping at 1"])
def test_a_seeded_gradient_fault_fires_at_its_step_and_names_the_parameter(clip):
    records = toy_run(0, clip=clip, fault=1e4)
    evidence = tensorquill.build_evidence(records, update_window=WINDOW)
    (found,) = spikes(evidence)
    assert found["step"] == FAULT_STEP and found["spike_steps"][0] == FAULT_STEP
    assert found["parameters"][0]["name"] == SEEDED
    # Adam bounds what one bad gradient can do: the seeded parameter moved a few learning rates.
    top = found["parameters"][0]
    assert top["update_rms"] > 1.5 * top["previous"] and top["update_rms"] < 10 * 1e-3
    assert f"most in {SEEDED}" in ev.incident_summary(evidence)
    # The same run without the fault is silent (above), and the fault is the only step flagged
    # before it: nothing earlier was a near miss either.
    assert margin([r for r in records if r["step"] < FAULT_STEP]) < 0.5


def test_the_toy_with_sgd_reads_not_checked_with_its_reason():
    records = toy_run(0, fault=1e4, make=lambda p: torch.optim.SGD(p, lr=0.05))[:60]
    evidence = tensorquill.build_evidence(records, update_window=WINDOW)
    assert spikes(evidence) == [] and evidence["kind_checks"]["update_spike"]["value"] == "unavailable"
    assert entry(evidence)["reason"].startswith("SGD: SGD is not Adam or AdamW: the update size is worked out from the two moments")


# --- record sets for the parity tests and the golden files -------------------------------------


def _wobble(at: int, salt: int = 0) -> float:
    return 0.9 + ((at * 37 + salt * 11) % 21) / 100


def _sizes(at: int, **over) -> dict:
    sizes = {"embed.weight": 0.0010 * _wobble(at), "blocks.0.weight": 0.0020 * _wobble(at, 1), "blocks.0.bias": 0.0030 * _wobble(at, 2)}
    return {**sizes, **over}


def _run(name: str, count: int, change=lambda at, record: None, attempt: int = 1, start: int = 0) -> list[dict]:
    out = []
    for at in range(start, start + count):
        record = {"schema_version": 1, "run_id": name, "rank": 0, "attempt": attempt, "step": at, "ts": float(at),
                  "grad_norm": _wobble(at, 3), "grad_norms": {"embed.weight": 0.5, "blocks.0.weight": 0.6, "blocks.0.bias": 0.4},
                  "update_rms": _sizes(at), "scalars": {"train/loss": 2.0 - at / 2000}}
        change(at, record)
        out.append(record)
    return out


def update_cases() -> dict[str, list[dict]]:
    """Record sets that reach each branch of the rule at the default baseline of the last 100 changes."""
    nan = float("nan")

    def spiked(at, record):
        if at in (105, 120):  # flagged: the gradient norm rose and one parameter's update jumped
            record.update(grad_norm=40.0, update_rms=_sizes(at, **{"blocks.0.weight": 0.05}))
        if at == 110:  # the same jump with a gradient norm that fell is not flagged. The step after it is: the
            # rule reads the change either way, the size fell back and the norm rose to its usual value
            record.update(grad_norm=0.5, update_rms=_sizes(at, **{"blocks.0.weight": 0.05}))

    def late_clipping(at, record):
        if at >= 20:
            record["clip_grad_norm"] = 2.0 * _wobble(at, 4)
        if at == 130:
            record.update(clip_grad_norm=30.0, update_rms=_sizes(at, **{"embed.weight": 0.04}))

    def odd(at, record):
        if at in (30, 31, 104):  # the scaler skipped: sizes repeat or are wild, and are not read
            record.update(step_skipped=True, grad_scale=1024.0, grad_norm=float("inf"), update_rms=_sizes(at, **{"blocks.0.bias": 9.0}))
        else:
            record.update(step_skipped=False, grad_scale=1024.0)
        if at == 40:
            del record["update_rms"]
        if at == 41:
            record["update_rms"] = {}
        if at == 42:
            record["update_rms"] = _sizes(at, **{"embed.weight": nan})
        if at == 43:
            record["update_rms"] = {name: nan for name in _sizes(at)}
        if at == 44:
            record["grad_norm"] = None
        if at == 112:
            record.update(grad_norm=25.0, update_rms=_sizes(at, **{"blocks.0.bias": 0.06, "embed.weight": 0.03}))

    def resumed(at, record):  # the second attempt: its first record is not compared, the next jump is
        if at == 115:
            record.update(grad_norm=30.0, update_rms=_sizes(at, **{"blocks.0.weight": 0.07}))

    def unread(at, record):
        del record["update_rms"]
        if at == 0:
            record["update_not_read"] = {"SGD": "SGD is not Adam or AdamW: the update size is worked out from the two moments Adam "
                                                "keeps (exp_avg and exp_avg_sq), which this optimizer does not have"}

    def partly(at, record):
        if at == 2:
            record["update_not_read"] = {"SGD": "SGD is not Adam or AdamW", "Adafactor": "Adafactor is not Adam or AdamW"}
        if at == 5:
            record["update_not_read"] = {"SGD": "said again, and the first stands", "auto_step=False": "the optimizer hooks are off"}

    def no_total(at, record):
        record["grad_norm"] = None
        del record["grad_norms"]

    first = _run("update-resumed", 108)
    first[106].update(grad_norm=30.0, update_rms=_sizes(106, **{"blocks.0.weight": 0.07}))  # rolled back by the resume
    return {
        "update size, a jump": _run("update-jump", 130, spiked),
        "update size, clipping starts late": _run("update-clip", 140, late_clipping),
        "update size, what is left out": _run("update-odd", 120, odd),
        "update size, a resume": first + _run("update-resumed", 20, resumed, attempt=2, start=105),
        "update size, not read": _run("update-unread", 12, unread),
        "update size, read for some": _run("update-partly", 12, partly),
        "update size, no gradient norm": _run("update-no-total", 12, no_total),
        "update size, still warming up": _run("update-short", 40),
    }


UPDATE_CASES = update_cases()


def test_the_record_sets_reach_each_branch_of_the_rule():
    found = {name: tensorquill.build_evidence(records) for name, records in UPDATE_CASES.items()}
    def of(name):
        return [(i["step"], i["spike_steps"], [p["name"] for p in i["parameters"]][:1]) for i in spikes(found[name])]
    assert of("update size, a jump") == [(105, [105, 111, 120], ["blocks.0.weight"])]
    assert of("update size, clipping starts late") == [(130, [130], ["embed.weight"])]
    assert of("update size, what is left out") == [(112, [112], ["blocks.0.bias"])]
    assert of("update size, a resume") == [(115, [115], ["blocks.0.weight"])]
    values = {name: evidence["kind_checks"]["update_spike"] for name, evidence in found.items()}
    assert (values["update size, not read"]["value"], values["update size, not read"]["records"]) == ("unavailable", 0)
    assert (values["update size, read for some"]["value"], values["update size, read for some"]["records"]) == ("unavailable", 12)
    assert entry(found["update size, read for some"]) == {
        "check": "update_size", "kinds": ["update_spike"], "optimizers": ["Adafactor", "SGD", "auto_step=False"],
        "reason": "Adafactor: Adafactor is not Adam or AdamW; SGD: SGD is not Adam or AdamW; auto_step=False: the optimizer hooks are off",
        "step": 2,
    }
    assert values["update size, no gradient norm"] == {
        "value": "not_configured", "records": 0, "records_needed": 102, "enough_history": False, "needs": NEEDS}
    assert (values["update size, still warming up"]["value"], values["update size, still warming up"]["enough_history"]) == ("checked", False)
    # The scalars logged on the step come with the incident, as for every kind.
    assert spikes(found["update size, a jump"])[0]["scalars"]["train/loss"]["value"] == 2.0 - 105 / 2000


def test_adam_and_adamw_are_both_read_whichever_way_pytorch_relates_them():
    # AdamW is a subclass of Adam only from PyTorch 2.7; in 2.6 it is a class of its own, and a
    # check for Adam alone refused every AdamW there. Classes are matched, never names.
    from tensorquill.capture import _not_adam

    class Plain(torch.optim.Adam):
        pass

    class PlainW(torch.optim.AdamW):
        pass

    class OwnStep(torch.optim.Adam):
        def step(self, closure=None):
            return super().step(closure)

    class Adam(torch.optim.SGD):  # another optimizer that only shares the name
        pass

    params = lambda: nn.Linear(2, 1).parameters()  # noqa: E731
    for cls in (torch.optim.Adam, torch.optim.AdamW, Plain, PlainW):
        assert _not_adam(cls(params(), lr=0.01)) is None, cls
    assert _not_adam(OwnStep(params(), lr=0.01)).startswith("OwnStep changes Adam's step")
    assert _not_adam(_Lookahead(params(), lr=0.01)).startswith("_Lookahead changes Adam's step")
    assert _not_adam(Adam(params(), lr=0.01)).startswith("Adam is not Adam or AdamW")
