"""Lane AE5: the capture's side of `WAVE_AE_CONTRACT.md`, sections 1, 3 and 4.

Section 1: `clip_calls` on a step's record when more than one `clip_grad_norm_` call covering the
session's parameters ran in the step (lane AD3's `l20_two_clips.py`).
Section 3: `init(..., ends_step=optimizer)`, the optimizer whose step closes each iteration (lane
AD3's `item5_a.py`, case 1). The default's stated limit stays pinned in `test_step_accounting.py`.
Section 4: `step_skipped_part` when the scaler skipped some of its optimizers and applied the others
(lane AD3's `item5_b.py`), read at the flush with no device sync added at the step."""

from __future__ import annotations

import math
import warnings

import pytest
import torch
from torch import nn
import torch.optim.optimizer as _optimizer_module
from torch.utils._python_dispatch import TorchDispatchMode

import tensorquill
from tensorquill.flight import record_to_row, row_to_record

# Keys that differ between two runs of one loop whatever is captured.
_RUN_KEYS = {"ts", "attempt", "step_time", "data_wait", "run_id"}


def _same(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _RUN_KEYS}


# --- Section 1: clip_calls -------------------------------------------------------------------


class _TwoGroups(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False)
        self.b = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.b(self.a(x)).sum()


def _groups() -> _TwoGroups:
    torch.manual_seed(0)
    m = _TwoGroups()
    with torch.no_grad():
        m.a.weight.fill_(1.0)
        m.b.weight.fill_(0.05)  # gradient norms: a 0.40, b 16.0 (lane AD3's l20_two_clips.py)
    return m


def _clip(m: _TwoGroups, how: str) -> None:
    if how == "two":
        torch.nn.utils.clip_grad_norm_(m.a.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(m.b.parameters(), 1.0)
    elif how == "one":
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)


def _clip_run(plan: list[str], **init) -> list[dict]:
    m = _groups()
    sink = tensorquill.MemorySink()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)  # weights stay, so every step clips the same norms
    with tensorquill.init(m, sink=sink, every=8, weights=False, **init):
        for how in plan:
            opt.zero_grad()
            m(torch.ones(2, 4)).backward()
            _clip(m, how)
            opt.step()
    return sink.records


def test_two_clip_calls_on_separate_groups_write_clip_calls():
    (record,) = _clip_run(["two"])
    assert record["clip_calls"] == 2
    assert record["clip_grad_norm"] == pytest.approx(math.sqrt(0.4**2 + 16.0**2), rel=1e-6)  # 16.0050, as on the base
    assert record["clip_grad_norm"] == pytest.approx(16.0050, abs=1e-4)


def test_one_clip_call_over_both_groups_writes_no_key():
    two, one = _clip_run(["two", "one"])
    assert two["clip_calls"] == 2  # the same run's step that clipped twice
    assert "clip_calls" not in one
    assert one["clip_grad_norm"] == pytest.approx(two["clip_grad_norm"], rel=1e-6)
    # With one call, the record is the one a run without the two-call step writes.
    (alone,) = _clip_run(["one"])
    assert _same(one) | {"step": 0} == _same(alone)


def test_a_step_that_does_not_clip_after_one_that_clipped_twice_has_no_key():
    records = _clip_run(["two", "none", "two", "two"])
    assert [r.get("clip_calls") for r in records] == [2, None, 2, 2]  # each step's own count
    assert "clip_grad_norm" not in records[1]


def test_clipping_inside_a_compiled_step_writes_no_clip_calls():
    torch._dynamo.reset()
    m = _groups()
    sink = tensorquill.MemorySink()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)

    @torch.compile(backend="eager")
    def clip_twice():
        torch.nn.utils.clip_grad_norm_(m.a.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(m.b.parameters(), 1.0)

    with tensorquill.init(m, sink=sink, every=8, weights=False):
        for compiled in (True, True, False):
            opt.zero_grad()
            m(torch.ones(2, 4)).backward()
            clip_twice() if compiled else _clip(m, "two")
            opt.step()
    compiled_a, compiled_b, eager = sink.records
    # The count is not known in a compiled step: no key, whatever the number of calls.
    assert "clip_calls" not in compiled_a and "clip_calls" not in compiled_b
    assert compiled_a["clip_grad_norm"] == pytest.approx(16.0050, abs=1e-4)
    assert eager["clip_calls"] == 2


def test_clip_calls_and_step_skipped_part_travel_through_extra_json():
    record = {"schema_version": 1, "run_id": "r", "rank": 0, "step": 3, "clip_grad_norm": 16.005,
              "clip_calls": 2, "step_skipped": True, "step_skipped_by": "scaler",
              "step_skipped_part": {"skipped": 1, "of": 2}}
    row = record_to_row(record)
    assert "clip_calls" in row["extra_json"] and "step_skipped_part" in row["extra_json"]
    back = row_to_record(row)
    assert back["clip_calls"] == 2
    assert back["step_skipped_part"] == {"skipped": 1, "of": 2}
    assert back["step_skipped"] is True and back["step_skipped_by"] == "scaler"


# --- Section 3: ends_step= -------------------------------------------------------------------


def _deep() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))


def _late_run(*, ends: bool, skip_other: tuple[int, ...] = (), steps: int = 5, compile_backend: str | None = None,
              updates: bool = True, probe=None, **init):
    """Lane AD3's item5_a.py, case 1: `other` starts at iteration 2 and sits before `first`."""
    model = _deep()
    first = torch.optim.Adam(list(model[0].parameters()) + list(model[2].parameters()), lr=1e-2)
    other = torch.optim.Adam(model[4].parameters(), lr=1e-2)
    sink = tensorquill.MemorySink()
    moved, after = {}, {}
    if compile_backend is not None:
        torch._dynamo.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tensorquill.init(model, sink=sink, every=8, updates=updates, weights=1, layers=(),
                              probe=probe, **({"ends_step": first} if ends else {}), **init):
            run = torch.compile(model, backend=compile_backend) if compile_backend else model
            torch.manual_seed(1)
            for step in range(steps):
                for opt in (first, other):
                    opt.zero_grad()
                run(torch.randn(8, 16)).pow(2).mean().backward()
                before = model[4].weight.detach().clone()
                if step >= 2 and step not in skip_other:
                    other.step()
                first.step()
                moved[step] = not torch.equal(before, model[4].weight.detach())
                after[step] = model[4].weight.detach().clone()
    return sink.records, moved, after


_LAST = {"4.weight", "4.bias"}


def test_ends_step_puts_an_earlier_optimizers_update_on_the_step_it_ends():
    records, moved, _ = _late_run(ends=True, probe=lambda m: m(torch.ones(1, 16)), probe_every=1)
    assert [r["step"] for r in records] == [0, 1, 2, 3, 4]
    assert [s for s, m in moved.items() if m] == [2, 3, 4]
    # Its update is on the records of the steps it moved at: 2, 3 and 4 (on the base: 1, 2 and 3).
    both = sorted(_LAST)
    assert [sorted(_LAST & set(r.get("update_rms", {}))) for r in records] == [[], [], both, both, both]
    # The record of step 4 does not list it unchanged (on the base it does, though it moved).
    assert "4.weight" not in records[4]["weights_unchanged"]
    assert "4.weight" in records[1]["weights_unchanged"]  # it did not move at step 1
    assert all("probe" in r for r in records)


def test_ends_step_an_iteration_in_which_the_other_does_not_step_has_nothing_of_it():
    # The author's control: `other` does not step at iteration 3, so nothing of it is on that record.
    records, moved, _ = _late_run(ends=True, skip_other=(3,), steps=6)
    full, _, _ = _late_run(ends=True, steps=6)
    assert [s for s, m in moved.items() if m] == [2, 4, 5]
    assert not _LAST & set(records[3].get("update_rms", {}))  # its previous update is not carried forward
    assert _LAST <= set(records[3]["weights_unchanged"])
    for step in (2, 4, 5):
        assert _LAST <= set(records[step]["update_rms"]), step
        assert not _LAST & set(records[step]["weights_unchanged"]), step
    # The records before it are those of the run in which it stepped.
    for a, b in zip(records[:3], full[:3]):
        assert _same(a) == _same(b)


def _follower_run(*, ends: bool, steps: int = 7, every: int = 3):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    first = torch.optim.Adam(model[0].parameters(), lr=0.1)
    other = torch.optim.Adam(model[2].parameters(), lr=0.01)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=every, weights=1, updates=True,
                          probe=lambda m: m(torch.ones(1, 4)), probe_every=2,
                          **({"ends_step": other} if ends else {})):
        torch.manual_seed(1)
        for _ in range(steps):
            first.zero_grad()
            other.zero_grad()
            model(torch.randn(3, 4)).pow(2).mean().backward()
            first.step()
            other.step()
    return sink.records


@pytest.mark.parametrize("every", [3, 1, 50])
def test_ends_step_on_a_true_follower_gives_the_records_the_default_gives(every):
    default, named = _follower_run(ends=False, every=every), _follower_run(ends=True, every=every)
    assert [r["step"] for r in named] == [r["step"] for r in default] == list(range(7))
    for a, b in zip(default, named):
        # Every key the same but the learning rate, which is the named optimizer's: the default
        # records the first optimizer's, as its lead. This page's reading, said in the report.
        assert {k: v for k, v in _same(b).items() if k != "lr"} == {k: v for k, v in _same(a).items() if k != "lr"}
        assert a["lr"] == [0.1] and b["lr"] == [0.01]


class _Players(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.g = nn.Linear(4, 4)
        self.d = nn.Linear(4, 1)

    def forward(self, z):  # not used by the loop; both players are called directly
        return self.d(self.g(z))


@pytest.mark.parametrize("every", [4, 1])
def test_ends_step_two_players_each_with_its_own_backward(every):
    torch.manual_seed(0)
    model = _Players()
    d_opt = torch.optim.Adam(model.d.parameters(), lr=1e-2)
    g_opt = torch.optim.Adam(model.g.parameters(), lr=1e-2)
    sink = tensorquill.MemorySink()
    change: list[dict[str, float]] = []
    with tensorquill.init(model, sink=sink, every=every, updates=True, weights=1, layers=(), ends_step=g_opt):
        torch.manual_seed(1)
        for _ in range(6):
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            real, z = torch.randn(8, 4), torch.randn(8, 4)
            d_opt.zero_grad()
            (model.d(model.g(z).detach()).mean() - model.d(real).mean()).backward()
            d_opt.step()
            g_opt.zero_grad()
            (-model.d(model.g(z)).mean()).backward()
            g_opt.step()
            change.append({n: float((p.detach() - before[n]).norm() / math.sqrt(p.numel()))
                           for n, p in model.named_parameters()})
    assert [r["step"] for r in sink.records] == list(range(6))
    for record, moved in zip(sink.records, change):
        # Each player's update is on the record of the iteration it stepped in: its RMS is the
        # change the parameter made in that iteration (Adam, no weight decay).
        assert set(record["update_rms"]) == set(moved)
        for name, rms in moved.items():
            assert record["update_rms"][name] == pytest.approx(rms, rel=1e-4), (record["step"], name)
        assert record["lr"] == [1e-2]


def test_without_ends_step_the_late_optimizer_is_read_as_before():
    # The default, as `test_step_accounting.py`'s pinned test states it: one step early.
    records, _, _ = _late_run(ends=False)
    assert [sorted(_LAST & set(r.get("update_rms", {}))) for r in records] == [
        [], sorted(_LAST), sorted(_LAST), sorted(_LAST), []]
    assert "4.weight" in records[4]["weights_unchanged"]


def test_ends_step_with_the_weight_check_and_the_probe_reads_the_model_after_every_optimizer():
    seen = []
    records, _, after = _late_run(ends=True, probe=lambda m: seen.append(m[4].weight.detach().clone()),
                                  probe_every=1)
    assert len(seen) == 5
    for step, weight in enumerate(seen):
        assert torch.equal(weight, after[step]), step  # the probe ran once both optimizers had stepped
    norms = [r["weight_norms"]["4.weight"] for r in records]
    assert norms == pytest.approx([float(after[s].double().norm()) for s in range(5)], rel=1e-6)


def test_ends_step_with_updates_off_ends_the_same_steps():
    records, moved, _ = _late_run(ends=True, updates=False)
    assert [r["step"] for r in records] == [0, 1, 2, 3, 4]
    assert all("update_rms" not in r for r in records)
    for step in range(1, 5):
        assert ("4.weight" in records[step]["weights_unchanged"]) == (not moved[step]), step


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_ends_step_under_torch_compile_gives_the_eager_records(backend):
    eager, _, _ = _late_run(ends=True)
    compiled, _, _ = _late_run(ends=True, compile_backend=backend)
    assert [r["step"] for r in compiled] == [0, 1, 2, 3, 4]
    for a, b in zip(eager, compiled):
        assert sorted(a.get("update_rms", {})) == sorted(b.get("update_rms", {}))
        assert b.get("update_rms") == pytest.approx(a.get("update_rms"), rel=1e-5)
        assert a.get("weights_unchanged") == b.get("weights_unchanged")


@pytest.mark.parametrize("bad, why", [
    ("not_optimizer", "must be a torch.optim.Optimizer"),
    ("foreign", "holds at least one of the model's parameters"),
    ("scaler", "cannot be combined with scaler="),
    ("manual", "cannot be combined with auto_step=False"),
])
def test_ends_step_refused_at_init(bad, why):
    model = _deep()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = {
        "not_optimizer": {"ends_step": object()},
        "foreign": {"ends_step": torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)},
        "scaler": {"ends_step": opt, "scaler": torch.amp.GradScaler("cpu")},
        "manual": {"ends_step": opt, "auto_step": False},
    }[bad]
    sink = tensorquill.MemorySink()
    hooks = len(_optimizer_module._global_optimizer_post_hooks)
    with pytest.raises(ValueError, match=why):
        tensorquill.init(model, sink=sink, **kwargs)
    assert len(_optimizer_module._global_optimizer_post_hooks) == hooks  # nothing was installed
    # Training goes on untouched after the refusal.
    opt.zero_grad()
    model(torch.randn(8, 16)).pow(2).mean().backward()
    opt.step()
    assert sink.records == []


def test_ends_step_with_a_disabled_scaler_is_accepted():
    # This page's reading as built: a disabled scaler does not end steps (`plan.scaler` is None), so
    # nothing conflicts.
    model = _deep()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=4, ends_step=opt,
                          scaler=torch.amp.GradScaler("cpu", enabled=False)):
        for _ in range(2):
            opt.zero_grad()
            model(torch.randn(8, 16)).pow(2).mean().backward()
            opt.step()
    assert [r["step"] for r in sink.records] == [0, 1]


def test_ends_step_that_never_steps_warns_once_and_records_nothing():
    model = _deep()
    first = torch.optim.SGD(model[0].parameters(), lr=0.1)
    never = torch.optim.SGD(model[4].parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tensorquill.init(model, sink=sink, every=4, ends_step=never):
            for _ in range(25):
                first.zero_grad()
                model(torch.randn(8, 16)).pow(2).mean().backward()
                first.step()
    said = [str(w.message) for w in caught if "ends_step" in str(w.message)]
    assert len(said) == 1 and "10 steps" in said[0]
    assert sink.records == []


# --- Section 4: step_skipped_part ------------------------------------------------------------


def _mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def _scaled_run(optimizers: int, inf_at=(), both=False, every=8, steps=5, *, mode=None, **init):
    """Lane AD3's item5_b.py, case 8: an Inf only in the second group's gradients at `inf_at`."""
    model = _mlp()
    groups = [list(model[0].parameters()), list(model[2].parameters())]
    opts = ([torch.optim.SGD(model.parameters(), lr=0.1)] if optimizers == 1
            else [torch.optim.SGD(g, lr=0.1) for g in groups])
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=every, scaler=scaler, **{"weights": 1, **init})
    torch.manual_seed(1)
    data = [torch.randn(3, 4) for _ in range(steps)]
    if mode is not None:
        mode.__enter__()
    try:
        for step in range(steps):
            scaler.scale(model(data[step]).pow(2).mean()).backward()
            if step in inf_at:
                model[2].weight.grad[0, 0] = float("inf")
                if both:
                    model[0].weight.grad[0, 0] = float("inf")
            for opt in opts:
                scaler.step(opt)
            scaler.update()
            for opt in opts:
                opt.zero_grad()
    finally:
        if mode is not None:
            mode.__exit__(None, None, None)
    return sink, session


def test_a_partial_skip_says_how_many_optimizers_were_skipped():
    sink, session = _scaled_run(2, inf_at=(2,))
    session.close()
    records = sink.records
    assert [r["step"] for r in records] == [0, 1, 2, 3, 4]
    partial = records[2]
    assert partial["step_skipped"] is True and partial["step_skipped_by"] == "scaler"
    assert partial["step_skipped_part"] == {"skipped": 1, "of": 2}
    # The weight check on it as on the base: the second's parameters unchanged, the first's changed.
    assert partial["weights_unchanged"] == ["2.weight", "2.bias"]
    assert all("step_skipped_part" not in r for i, r in enumerate(records) if i != 2)


def test_every_optimizer_skipped_has_no_part_key():
    sink, session = _scaled_run(2, inf_at=(2,), both=True)
    session.close()
    record = sink.records[2]
    assert record["step_skipped"] is True and record["step_skipped_by"] == "scaler"
    assert "step_skipped_part" not in record
    # The same run's other key: a partial step in it says so.
    sink, session = _scaled_run(2, inf_at=(1,))
    session.close()
    assert sink.records[1]["step_skipped_part"] == {"skipped": 1, "of": 2}


def test_no_optimizer_skipped_has_no_part_key():
    sink, session = _scaled_run(2, inf_at=(3,))
    session.close()
    for record in sink.records[:3] + sink.records[4:]:
        assert record["step_skipped"] is False and "step_skipped_part" not in record
    assert sink.records[3]["step_skipped_part"] == {"skipped": 1, "of": 2}


@pytest.mark.parametrize("inf_at", [(), (2,)])
def test_one_optimizer_through_the_scaler_has_the_base_records(inf_at):
    sink, session = _scaled_run(1, inf_at=inf_at)
    session.close()
    base_keys = {"schema_version", "rank", "step", "layers", "grad_norm", "grad_norms", "weight_norms",
                 "optimizer_ownership", "weight_dtypes", "lr", "step_skipped", "grad_scale"}
    for record in sink.records:
        assert "step_skipped_part" not in record
        extra = set(record) - base_keys - _RUN_KEYS - {"weights_unchanged", "step_skipped_by", "launch"}
        assert not extra, extra
        assert record["step_skipped"] is (record["step"] in inf_at)
    # A two-optimizer partial step in the same process says so: the key is the condition's.
    sink, session = _scaled_run(2, inf_at=(2,))
    session.close()
    assert sink.records[2]["step_skipped_part"] == {"skipped": 1, "of": 2}


def test_a_caller_who_also_reports_the_partial_step_keeps_the_part_key():
    model = _mlp()
    opts = [torch.optim.SGD(model[0].parameters(), lr=0.1), torch.optim.SGD(model[2].parameters(), lr=0.1)]
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8)
    sink = tensorquill.MemorySink()
    with tensorquill.init(model, sink=sink, every=8, scaler=scaler, auto_step=False) as session:
        for step in range(3):
            scaler.scale(model(torch.randn(3, 4)).pow(2).mean()).backward()
            if step == 1:
                model[2].weight.grad[0, 0] = float("inf")
            for opt in opts:
                scaler.step(opt)
            scaler.update()
            session.step(skipped=step == 1)
            for opt in opts:
                opt.zero_grad()
    record = sink.records[1]
    assert record["step_skipped_by"] == "both"
    assert record["step_skipped_part"] == {"skipped": 1, "of": 2}


class _Syncs(TorchDispatchMode):
    """Counts the ops that copy a device value to the host (`.item()`, `bool()`, `float()` of a
    tensor), the device sync the step must not gain. On the CPU, a copy from the device cannot
    happen, so these are what a sync would be."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("capture", [{}, {"weights": False, "grads": False, "layers": ()}])
def test_a_partial_skip_adds_no_device_sync_at_the_step(capture):
    # The same loop without TensorQuill is the control: PyTorch's own `scaler.step` reads its
    # verdict with `.item()`. The ring holds more steps than the loop, so nothing is flushed in it.
    def bare(mode):
        model = _mlp()
        opts = [torch.optim.SGD(model[0].parameters(), lr=0.1), torch.optim.SGD(model[2].parameters(), lr=0.1)]
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0**8)
        torch.manual_seed(1)
        data = [torch.randn(3, 4) for _ in range(5)]
        with mode:
            for step in range(5):
                scaler.scale(model(data[step]).pow(2).mean()).backward()
                if step == 2:
                    model[2].weight.grad[0, 0] = float("inf")
                for opt in opts:
                    scaler.step(opt)
                scaler.update()
                for opt in opts:
                    opt.zero_grad()

    control = _Syncs()
    bare(control)
    watched = _Syncs()
    sink, session = _scaled_run(2, inf_at=(2,), every=16, mode=watched, **capture)
    assert watched.count == control.count > 0
    # The verdicts are still on the device, in the ring, until the flush.
    pending = [v for row in session._ring_amp for v in row.values() if isinstance(v, torch.Tensor)]
    assert len(pending) == 3 * 5  # the verdict, the scale and the count of optimizers skipped, a step each
    assert sink.records == []
    session.close()
    assert sink.records[2]["step_skipped_part"] == {"skipped": 1, "of": 2}


# --- The author's condition on section 3's warning: it cannot raise into training under -W error ---


def _never_steps_loop(model, first, steps=25):
    for _ in range(steps):
        first.zero_grad()
        model(torch.randn(8, 16)).pow(2).mean().backward()
        first.step()


def test_the_ends_step_warning_does_not_raise_under_an_error_filter(caplog):
    model = _deep()
    first = torch.optim.SGD(model[0].parameters(), lr=0.1)
    never = torch.optim.SGD(model[4].parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    before = model[0].weight.detach().clone()
    with caplog.at_level("WARNING", logger="tensorquill"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # every warning an error, as `python -W error` makes it
            with tensorquill.init(model, sink=sink, every=4, ends_step=never):
                _never_steps_loop(model, first)  # raises nothing
    assert not torch.equal(before, model[0].weight)  # training went on
    said = [r.getMessage() for r in caplog.records if "ends_step" in r.getMessage()]
    assert len(said) == 1 and "10 steps" in said[0]  # said once, on the logger instead
    assert sink.records == []


def test_the_base_ten_step_warning_does_not_raise_under_an_error_filter(caplog):
    # A control, as the base has it: an optimizer holding none of the model's parameters.
    model = _deep()
    elsewhere = nn.Linear(16, 1)
    foreign = torch.optim.SGD(elsewhere.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    with caplog.at_level("WARNING", logger="tensorquill"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with tensorquill.init(model, sink=sink, every=4):
                for _ in range(12):
                    foreign.zero_grad()
                    elsewhere(torch.randn(8, 16)).pow(2).mean().backward()
                    foreign.step()
    assert any("has recorded no steps" in r.getMessage() for r in caplog.records)


_W_ERROR_SCRIPT = """
import torch, tensorquill
from torch import nn
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
first = torch.optim.SGD(model[0].parameters(), lr=0.1)
never = torch.optim.SGD(model[2].parameters(), lr=0.1)
sink = tensorquill.MemorySink()
with tensorquill.init(model, sink=sink, every=4, ends_step=never):
    for _ in range(25):
        first.zero_grad()
        model(torch.randn(8, 16)).pow(2).mean().backward()
        first.step()
print("trained", len(sink.records))
"""


def test_the_ends_step_warning_under_python_w_error():
    import os
    import subprocess
    import sys

    src = os.path.join(os.path.dirname(tensorquill.__file__), os.pardir)
    env = {**os.environ, "PYTHONPATH": os.path.abspath(src)}
    # Without NumPy, importing PyTorch itself warns ("Failed to initialize NumPy"), which `-W error` would turn
    # into a failed import before TensorQuill runs. Only that message is let through; every other warning is an error.
    done = subprocess.run([sys.executable, "-W", "error", "-W", "ignore:Failed to initialize NumPy", "-c", _W_ERROR_SCRIPT],
                          capture_output=True, text=True,
                          env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "trained 0" in done.stdout
    assert done.stderr.count("ends_step") == 1  # the one warning, on the logger's stderr
