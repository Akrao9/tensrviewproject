"""Lane AK-5: where a scalar logged outside a step is recorded, and `step=` refused.

The owner's decision of Nov 24 (planning/OPEN_BUGS.md, section 20): a scalar logged while the
session's phase is `"eval"`, or still pending when `close()` runs, is recorded on the newest step,
if that step is still in memory (the ring's last row, or the step kept back when the ring last
filled). A record already handed to the writer is never changed. With no step in memory the value
is left out, said once per session through the warning that never raises, until protocol 5's
scalar-only record gives it a place. `session.log(..., step=...)` raises `TypeError`; a dictionary
key `"step"` stays an ordinary scalar.

The two reproduction scripts of section 20 (`planning/repro/eval_scalar_step_repro.py` and
`eval_scalar_edge_cases.py`) are the first tests here, row by row, with the new results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture
from tensorquill.sink import JsonlSink, MemorySink

SDK = Path(__file__).resolve().parents[1]
REPO = SDK.parent
BASE = "e9b159888975a99c767b0c648279ec72eeef5fd7"  # the lane's base: lane AK-1's mirror, before AK-5
LEFT_OUT = "protocol 5"  # the warning names the scalar-only record that will give such a value a place


# ---- helpers ---------------------------------------------------------------------------------


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))


def _open(model: nn.Module, every: int = 2, sink=None, **kwargs):
    sink = MemorySink() if sink is None else sink
    options = dict(sink=sink, every=every, run_id="ak5", heartbeat_seconds=None, stack_seconds=None)
    options.update(kwargs)
    session = tensorquill.init(model, **options)
    return session, sink, torch.optim.SGD(model.parameters(), lr=0.01)


def _step(session, model, opt, value=None) -> float:
    loss = model(torch.randn(8, 4)).pow(2).mean()
    session.log({"train/loss": loss if value is None else value})  # before optimizer.step(): this step's
    loss.backward()
    opt.step()  # the TensorQuill step ends here (auto_step)
    opt.zero_grad()
    return loss.item()


def _by_step(records) -> dict[int, dict]:
    return {r["step"]: r.get("scalars", {}) for r in records}


def _on_disk(path: Path) -> list[int]:
    try:
        return [json.loads(line)["step"] for line in path.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def _wait_for(path: Path, steps: list[int], seconds: float = 10.0) -> list[int]:
    """The steps in the records file once the writer has written `steps` (or the deadline)."""
    deadline = time.monotonic() + seconds
    while _on_disk(path) != steps and time.monotonic() < deadline:
        time.sleep(0.02)
    return _on_disk(path)


def _left_out_warnings(caught) -> list[str]:
    return [str(w.message) for w in caught if LEFT_OUT in str(w.message)]


# ---- section 20's first table (eval_scalar_step_repro.py) --------------------------------------


def test_the_first_reproduction_script_row_by_row():
    model = _model()
    session, sink, opt = _open(model, every=2)
    for i in range(5):
        _step(session, model, opt)
        if i == 2:
            with session.phase("eval"):
                session.log({"eval/loss": 1.8})  # an evaluation after step 2
    session.log({"eval/final": 0.9})  # after the last step, before close()
    with pytest.raises(TypeError, match="protocol 5"):
        session.log({"x": 1.0}, step=500)
    session.close()
    scalars = _by_step(sink.records)
    assert sorted(scalars) == [0, 1, 2, 3, 4]
    assert all("train/loss" in s for s in scalars.values())
    assert scalars[2]["eval/loss"] == 1.8 and "eval/loss" not in scalars[3]  # was step 3
    assert scalars[4]["eval/final"] == 0.9  # was in no record
    assert all("x" not in s and "step" not in s for s in scalars.values())


# ---- section 20's second table (eval_scalar_edge_cases.py) -------------------------------------


@pytest.mark.parametrize("then", ["step", "close"])
def test_a_slow_evaluation_at_a_ring_boundary(tmp_path, then):
    path = tmp_path / "slow.jsonl"
    model = _model()
    session, _, opt = _open(model, every=2, sink=JsonlSink(path))
    for i in range(4):
        _step(session, model, opt)  # every=2: the ring filled at step 1 and at step 3, which is kept back
    assert _wait_for(path, [0, 1, 2]) == [0, 1, 2]
    with session.phase("eval"):
        session.log({"eval/loss": 1.5})
        time.sleep(0.3)  # a slow validation: no timer moves step 3 out of memory
        assert _on_disk(path) == [0, 1, 2]
    assert _on_disk(path) == [0, 1, 2]
    if then == "step":
        _step(session, model, opt)
    session.close()
    scalars = _by_step(json.loads(line) for line in path.read_text().splitlines())
    assert scalars[3]["eval/loss"] == 1.5
    assert all("eval/loss" not in s for step, s in scalars.items() if step != 3)
    assert sorted(scalars) == ([0, 1, 2, 3, 4] if then == "step" else [0, 1, 2, 3])


def test_a_slow_evaluation_off_a_ring_boundary(tmp_path):
    path = tmp_path / "slow.jsonl"
    model = _model()
    session, _, opt = _open(model, every=2, sink=JsonlSink(path))
    for i in range(5):
        _step(session, model, opt)  # step 3 written when step 4 ended; step 4 is the ring's only row
    assert _wait_for(path, [0, 1, 2, 3]) == [0, 1, 2, 3]
    with session.phase("eval"):
        time.sleep(0.3)
        session.log({"eval/loss": 1.25})
        assert _on_disk(path) == [0, 1, 2, 3]
    _step(session, model, opt)
    session.close()
    scalars = _by_step(json.loads(line) for line in path.read_text().splitlines())
    assert scalars[4]["eval/loss"] == 1.25 and "eval/loss" not in scalars[5]


def test_flush_then_an_evaluation_is_left_out_with_one_warning(tmp_path):
    path = tmp_path / "flushed.jsonl"
    model = _model()
    session, _, opt = _open(model, every=50, sink=JsonlSink(path))
    for i in range(3):
        _step(session, model, opt)
    session.flush()
    assert _wait_for(path, [0, 1, 2]) == [0, 1, 2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with session.phase("eval"):
            session.log({"eval/loss": 1.8})  # step 2 is already written: no step in memory
            session.log({"eval/acc": 0.5})
    assert len(_left_out_warnings(caught)) == 1
    _step(session, model, opt)
    session.close()
    scalars = _by_step(json.loads(line) for line in path.read_text().splitlines())
    assert sorted(scalars) == [0, 1, 2, 3]
    assert all("eval/loss" not in s and "eval/acc" not in s for s in scalars.values())  # was on step 3


@pytest.mark.parametrize("steps", [2, 0])
def test_a_baseline_evaluation_before_the_first_step_is_left_out(steps):
    model = _model()
    session, sink, opt = _open(model, every=50)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with session.phase("eval"):
            session.log({"eval/baseline": 2.5})
        session.log({"eval/baseline2": 2.0})  # pending at close too, when no step follows
        for _ in range(steps):
            _step(session, model, opt)
        session.close()
    said = _left_out_warnings(caught)
    assert len(said) == 1 and "eval/baseline" in said[0]
    scalars = _by_step(sink.records)
    assert sorted(scalars) == list(range(steps))
    assert all("eval/baseline" not in s for s in scalars.values())  # was on step 0
    if steps:
        assert scalars[0]["eval/baseline2"] == 2.0  # logged in "train" before step 0: step 0's, as before


def test_values_left_out_never_raise_under_python_w_error(tmp_path):
    script = textwrap.dedent(
        f"""
        import torch, tensorquill
        model = torch.nn.Linear(4, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        s = tensorquill.init(model, sink=tensorquill.JsonlSink({str(tmp_path / 'r.jsonl')!r}), every=50,
                             heartbeat_seconds=None, stack_seconds=None)
        with s.phase("eval"):
            s.log({{"eval/baseline": 2.5}})      # before the first step
        for _ in range(2):
            model(torch.randn(2, 4)).sum().backward(); opt.step(); opt.zero_grad()
        s.flush()
        with s.phase("eval"):
            s.log({{"eval/after_flush": 1.0}})   # no step in memory
        s.log({{"pending": 1.0}})                # pending at close after a flush
        s.close()
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(SDK / "src")}
    done = subprocess.run([sys.executable, "-W", "error", "-c", script], capture_output=True, text=True, env=env,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
    assert done.stderr.count(LEFT_OUT) == 1  # said once per session, on the tensorquill logger


def test_step_as_a_keyword_is_refused_and_a_step_key_is_a_scalar():
    model = _model()
    session, sink, opt = _open(model, every=50)
    for call in (lambda: session.log(eval_loss=1.8, step=500), lambda: session.log({"a": 1.0}, step=None),
                 lambda: session.log({"a": 1.0}, step=3)):
        with pytest.raises(TypeError, match="explicit step") as raised:
            call()
        assert "protocol 5" in str(raised.value)
    session.log({"step": 5})  # W&B users log a metric named step as a custom x-axis
    _step(session, model, opt)
    session.close()
    assert sink.records[0]["scalars"]["step"] == 5
    assert "eval_loss" not in sink.records[0]["scalars"] and "a" not in sink.records[0]["scalars"]


def test_the_repository_passes_no_step_keyword_to_session_log():
    """Searched before refusing `step=`: no example, test or page passes it to `session.log`."""
    import re

    # A call with a value for `step` (the refusal's own words, `step=...)`, are not a call).
    call = re.compile(r"session\.log\([^)]*\bstep\s*=\s*(?!\.\.\.)")
    found = []
    for path in list((SDK / "examples").rglob("*.py")) + list((SDK / "src").rglob("*.py")) + [
        SDK / "README.md", SDK / "RUN_ON_YOUR_JOB.md", SDK / "AGENT_INSTRUCTIONS.md",
    ]:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if call.search(line):
                found.append(f"{path.name}:{number}")
    assert found == []


# ---- what else the rule covers ----------------------------------------------------------------


def test_every_1_each_evaluation_lands_on_the_step_it_followed():
    model = _model()
    session, sink, opt = _open(model, every=1)
    for i in range(3):
        _step(session, model, opt)
        with session.phase("eval"):
            session.log({"eval/loss": float(i)})
    session.close()
    scalars = _by_step(sink.records)
    assert [scalars[i]["eval/loss"] for i in range(3)] == [0.0, 1.0, 2.0]


def test_set_phase_eval_as_the_callbacks_set_it():
    model = _model()
    session, sink, opt = _open(model, every=4)
    _step(session, model, opt)
    session.set_phase("eval")
    session.log({"eval/loss": 3.0})
    session.set_phase("train")
    _step(session, model, opt)
    session.close()
    assert _by_step(sink.records)[0]["eval/loss"] == 3.0 and "eval/loss" not in _by_step(sink.records)[1]


def test_an_evaluation_name_logged_twice_keeps_the_last_value():
    model = _model()
    session, sink, opt = _open(model, every=2)
    for i in range(2):
        _step(session, model, opt)  # step 1 kept back (held)
    with session.phase("eval"):
        session.log({"eval/loss": 1.0})
        session.log(**{"eval/loss": 0.5})
    _step(session, model, opt)
    with session.phase("eval"):
        session.log({"eval/loss": 2.0})
        session.log({"eval/loss": 1.5})  # step 2 in the ring, not held
    session.close()
    scalars = _by_step(sink.records)
    assert scalars[1]["eval/loss"] == 0.5 and scalars[2]["eval/loss"] == 1.5


def test_a_value_pending_at_close_does_not_replace_a_name_the_newest_step_holds():
    """A loop that logs its loss after `optimizer.step()` has every loss one step late and the last
    one pending at close. That pending value is not put over the newest step's own; a name the
    newest step does not hold is added."""
    model = _model()
    session, sink, opt = _open(model, every=2)
    losses = []
    for i in range(3):
        loss = model(torch.randn(8, 4)).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        session.log({"loss": float(i)})  # after the step: recorded on the next one
        losses.append(float(i))
    session.log({"eval/final": 0.25})
    session.close()
    scalars = _by_step(sink.records)
    assert "loss" not in scalars[0]
    assert scalars[1]["loss"] == 0.0 and scalars[2]["loss"] == 1.0  # not 2.0, the value pending at close
    assert scalars[2]["eval/final"] == 0.25


def test_a_stopped_session_writes_nothing_more_and_raises_nothing():
    class Failing(MemorySink):
        def write(self, records):
            raise OSError("disk gone")

    model = _model()
    session, sink, opt = _open(model, every=1, sink=Failing())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            _step(session, model, opt)
            if session._stopped:
                break
        assert session._stopped
        warnings.simplefilter("error")
        with session.phase("eval"):
            session.log({"eval/loss": 1.0})
        session.log({"pending": 1.0})
        session.close()
    assert sink.records == []


def test_a_followers_loss_a_save_and_an_evaluation_share_the_held_step(tmp_path):
    """A follower optimizer's loss (`_add_to_newest`, the Lightning callback's path) and a save after
    the step (`_record_manifest`) still land on the step kept back, beside the evaluation."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    session, sink, _ = _open(model, every=2)
    lead = torch.optim.SGD(model[0].parameters(), lr=0.01)
    follower = torch.optim.SGD(model[1].parameters(), lr=0.01)
    expected = {}
    for i in range(2):
        first = model(torch.randn(8, 4)).pow(2).mean()
        session.log({"train/loss": first})
        first.backward()
        lead.step()  # the lead ends the TensorQuill step
        second = model(torch.randn(8, 4)).pow(2).mean()
        second.backward()
        follower.step()
        assert session._follows(follower)
        session._add_to_newest("train/loss", second.detach())
        lead.zero_grad()
        follower.zero_grad()
        expected[i] = first.item() + second.item()
    assert session._held is not None  # step 1 kept back
    session.checkpoint({"state_dict": model.state_dict()}, event="save")
    with session.phase("eval"):
        session.log({"eval/loss": 0.75})
        session._add_to_newest("eval/extra", torch.tensor(1.0))  # adds under eval too
    session.close()
    records = {r["step"]: r for r in sink.records}
    assert records[0]["scalars"]["train/loss"] == pytest.approx(expected[0], rel=1e-5)
    assert records[1]["scalars"]["train/loss"] == pytest.approx(expected[1], rel=1e-5)
    assert records[1]["scalars"]["eval/loss"] == 0.75 and records[1]["scalars"]["eval/extra"] == 1.0
    assert records[1]["checkpoint"]["save"]["after_step"] == 1
    assert "eval/loss" not in records[0]["scalars"] and "checkpoint" not in records[0]


def test_a_late_cpu_tensor_is_recorded_and_adds_no_transfer_outside_the_held_steps_write(monkeypatch):
    real = capture._scalars_to_cpu
    calls = []

    def counted(rows):
        calls.append(len(rows))
        return real(rows)

    monkeypatch.setattr(capture, "_scalars_to_cpu", counted)

    def run(late: bool, every: int, n: int) -> tuple[list[int], dict]:
        calls.clear()
        model = _model()
        session, sink, opt = _open(model, every=every)
        for i in range(n):
            _step(session, model, opt, value=torch.tensor(float(i)))
        if late:
            with session.phase("eval"):
                session.log({"eval/loss": torch.tensor(0.5)})
        session.close()
        return list(calls), _by_step(sink.records)

    # Ring not full (every=4, three steps): the late tensor goes with the ring's own one transfer.
    plain, _ = run(False, 4, 3)
    late, scalars = run(True, 4, 3)
    assert late == plain and scalars[2]["eval/loss"] == 0.5
    # The newest step kept back (every=2): read again once when it is written, as a follower's loss is.
    plain, _ = run(False, 2, 2)
    late, scalars = run(True, 2, 2)
    assert late == plain + [1] and scalars[1]["eval/loss"] == 0.5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_a_late_mps_tensor_is_not_read_when_it_is_logged():
    model = _model().to("mps")
    session = tensorquill.init(model, sink=MemorySink(), every=4, run_id="ak5-mps", heartbeat_seconds=None,
                               stack_seconds=None)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        loss = model(torch.randn(8, 4, device="mps")).pow(2).mean()
        session.log({"train/loss": loss})
        loss.backward()
        opt.step()
        opt.zero_grad()
    with session.phase("eval"):
        session.log({"eval/loss": torch.tensor(0.5, device="mps")})
    kept = session._ring_scalars[-1]["eval/loss"]
    assert isinstance(kept, torch.Tensor) and kept.device.type == "mps"  # a copy on the device, not read
    session.close()
    assert _by_step(session._sink.records)[1]["eval/loss"] == pytest.approx(0.5)


def test_a_compiled_step_that_logs_is_not_recompiled_by_an_evaluation():
    """Today the evaluation's value sits in the step-in-progress dictionary, whose emptiness the
    compiled step guards on, and the next compiled call recompiles."""
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    model = nn.Linear(4, 1)
    session, sink, opt = _open(model, every=2)
    counter = CompileCounter()

    @torch.compile(backend=counter, fullgraph=True)
    def forward(x):
        loss = model(x).pow(2).mean()
        session.log({"train/loss": loss})
        return loss

    for i in range(6):
        forward(torch.randn(8, 4)).backward()
        opt.step()
        opt.zero_grad()
        if i in (1, 2):
            with session.phase("eval"):
                session.log({"eval/x": float(i)})
    session.close()
    torch._dynamo.reset()
    assert counter.frame_count == 1
    scalars = _by_step(sink.records)
    assert scalars[1]["eval/x"] == 1.0 and scalars[2]["eval/x"] == 2.0


def test_a_phase_left_behind_sends_the_next_steps_values_to_the_step_before():
    """Pinned as it is, for the owner: `set_phase("eval")` never set back. The next step's values,
    logged before its `optimizer.step()` ends the phase, are taken as late values of the newest step.
    The callbacks set the phase back; the Hugging Face one also does at each step's beginning."""
    model = _model()
    session, sink, opt = _open(model, every=4)
    _step(session, model, opt, value=1.0)
    session.set_phase("eval")
    _step(session, model, opt, value=2.0)  # ends the phase, after its value was logged
    _step(session, model, opt, value=3.0)
    session.close()
    scalars = _by_step(sink.records)
    assert scalars[0]["train/loss"] == 2.0 and "train/loss" not in scalars[1] and scalars[2]["train/loss"] == 3.0


# ---- a run that logs only before optimizer.step(): the same bytes as the base ----------------------


_FROZEN_RUN = textwrap.dedent(
    """
    import sys, time as real_time, types, torch
    from torch import nn
    import tensorquill
    from tensorquill import capture
    clock = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("__")})
    clock.time = lambda: 1_790_000_000.0
    capture.time = clock
    capture._next_attempt = lambda: 1_790_000_000_000
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    session = tensorquill.init(model, sink=tensorquill.JsonlSink(sys.argv[1]), run_id="ak5", heartbeat_seconds=None,
                               stack_seconds=None, every=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(7):
        loss = model(torch.full((2, 4), float(i))).pow(2).mean()
        loss.backward()
        session.log({"loss": loss.detach(), "i": i})
        opt.step()
        opt.zero_grad()
    session.close()
    """
)


def test_a_run_that_logs_only_before_the_step_writes_the_same_bytes_as_the_base(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", BASE, "sdk/src"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(base)], input=archive.stdout, check=True)
    out = {}
    for name, src in (("base", base / "sdk" / "src"), ("lane", SDK / "src")):
        env = {**os.environ, "PYTHONPATH": str(src)}
        path = tmp_path / f"{name}.jsonl"
        done = subprocess.run([sys.executable, "-c", _FROZEN_RUN, str(path)], capture_output=True, text=True,
                              env=env, timeout=180)
        assert done.returncode == 0, done.stderr
        out[name] = path.read_bytes()
    assert out["lane"].count(b"\n") == 7 and b'"loss"' in out["lane"]
    assert out["lane"] == out["base"]


# ---- the README ----------------------------------------------------------------------------------


def test_the_readme_says_where_a_late_value_is_recorded():
    text = " ".join((SDK / "README.md").read_text(encoding="utf-8").split())
    assert 'session.phase("eval")' in text
    assert "newest step" in text and "still in memory" in text
    assert "step=" in text and "TypeError" in text and "protocol 5" in text
