"""Lane AL-5, change 2: `Session.log(values, step=N)` writes the scalar-only record of
`WAVE_AL_CONTRACT.md` section 4.2.

Cases `scalar_record_keys` (the capture's path), `scalar_log_refusals` and
`scalar_baseline_and_step_0_share_a_point` (what the capture writes). In this lane's tree the
readers still refuse a record of version 2 (lane AL-4' widens `identity.py` in parallel), so what
the capture writes is read here as `.jsonl` lines, never through `build_evidence`.

Section 7's invariant is here too: a run that logs no `step=` writes the bytes it wrote before.
Shown without git history, as an A/B on the same code path: one frozen run with the feature
unused, one with `step=` calls added, whose step lines must be byte-equal.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture
from tensorquill.sink import JsonlSink, MemorySink

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_al" / "cases.json").read_text())["cases"]
KEYS = ["schema_version", "record", "run_id", "rank", "attempt", "step", "seq", "ts", "scalars"]


def _case(name: str) -> dict:
    return next(c for c in CASES if c["name"] == name)


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))


def _open(model, path: Path, every: int = 4, **kwargs):
    options = dict(sink=JsonlSink(path), every=every, run_id="al5_rec", heartbeat_seconds=None, stack_seconds=None)
    options.update(kwargs)
    session = tensorquill.init(model, **options)
    return session, torch.optim.SGD(model.parameters(), lr=0.01)


def _step(session, model, opt, value=None) -> None:
    loss = model(torch.randn(8, 4)).pow(2).mean()
    session.log({"train/loss": loss if value is None else value})
    loss.backward()
    opt.step()  # the TensorQuill step ends here (auto_step)
    opt.zero_grad()


def _lines(path: Path) -> list[dict]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _scalar_records(path: Path) -> list[dict]:
    return [r for r in _lines(path) if r.get("record") == "scalars"]


def _step_records(path: Path) -> list[dict]:
    return [r for r in _lines(path) if r.get("record") != "scalars"]


def _wait_for(path: Path, count: int, seconds: float = 10.0) -> list[dict]:
    deadline = time.monotonic() + seconds
    while len(_lines(path)) < count and time.monotonic() < deadline:
        time.sleep(0.02)
    return _lines(path)


# ---- scalar_record_keys: the nine keys --------------------------------------------------------


def test_the_record_has_exactly_the_nine_keys(tmp_path):
    """`scalar_record_keys`, the capture's path: `session.log(eval_loss=1.8, step=500)` at step 3 of
    six writes one line with these keys, in this order, and no others."""
    case = _case("scalar_record_keys")
    path = tmp_path / "keys.jsonl"
    model = _model()
    session, opt = _open(model, path, every=4)
    for i in range(6):
        _step(session, model, opt, value=2.0 - i / 10)
        if i == 3:
            before = time.time()
            session.log(eval_loss=1.8, step=500)
            after = time.time()
    session.close()
    records = _scalar_records(path)
    assert len(records) == 1
    record = records[0]
    assert list(record) == case["expect"]["keys"] == KEYS
    assert record["schema_version"] == 2 and record["record"] == "scalars"
    assert record["run_id"] == "al5_rec" and record["rank"] == 0 and record["attempt"] == session.attempt
    assert record["step"] == 500 and record["seq"] == 0
    assert before <= record["ts"] <= after
    assert record["scalars"] == {"eval_loss": 1.8}
    steps = _step_records(path)
    assert [r["step"] for r in steps] == [0, 1, 2, 3, 4, 5]
    assert all(r["schema_version"] == 1 and "seq" not in r and "record" not in r for r in steps)
    assert all("eval_loss" not in r.get("scalars", {}) for r in steps)  # never on a training step


def test_seq_counts_per_session_from_0_in_the_order_of_the_calls(tmp_path):
    model = _model()
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    session, opt = _open(model, first)
    session.log({"a": 1.0}, step=7)
    _step(session, model, opt)
    session.log({"b": 2.0}, step=3)  # an earlier step: seq, not step, orders the records
    session.log({"a": 3.0}, step=7)
    session.close()
    assert [(r["step"], r["seq"], r["scalars"]) for r in _scalar_records(first)] == [
        (7, 0, {"a": 1.0}), (3, 1, {"b": 2.0}), (7, 2, {"a": 3.0})]
    again, opt = _open(_model(), second)
    again.log({"c": 1.0}, step=0)
    again.close()
    assert [r["seq"] for r in _scalar_records(second)] == [0]  # a new session counts from 0 again


def test_several_values_in_one_call_are_one_record(tmp_path):
    path = tmp_path / "many.jsonl"
    session, opt = _open(_model(), path)
    session.log({"eval/loss": 1.5, "eval/acc": 0.25}, step=10, **{"eval/n": 64})
    session.close()
    (record,) = _scalar_records(path)
    assert record["scalars"] == {"eval/loss": 1.5, "eval/acc": 0.25, "eval/n": 64}
    assert isinstance(record["scalars"]["eval/n"], int)


def test_a_non_finite_value_is_written_as_the_sinks_string(tmp_path):
    path = tmp_path / "nan.jsonl"
    session, opt = _open(_model(), path)
    session.log({"x": float("nan"), "y": float("inf"), "z": torch.tensor(-math.inf)}, step=1)
    session.close()
    (record,) = _scalar_records(path)
    assert record["scalars"] == {"x": "nan", "y": "inf", "z": "-inf"}


# ---- scalar_log_refusals ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", _case("scalar_log_refusals")["expect"]["refused_steps"] + [None, 2.0, [3]],
                         ids=repr)
def test_step_refuses_before_anything_is_recorded(tmp_path, bad):
    """`scalar_log_refusals`: a bool, a float, a negative integer and anything not an integer, as
    `TypeError`, or `ValueError` for a negative. Nothing is recorded and no `seq` is taken."""
    path = tmp_path / "refused.jsonl"
    model = _model()
    session, opt = _open(model, path)
    expected = ValueError if isinstance(bad, int) and not isinstance(bad, bool) else TypeError
    with pytest.raises(expected, match="step"):
        session.log({"eval/loss": 1.0}, step=bad)
    assert session._scalar_records == []
    session.log({"eval/ok": 2.0}, step=4)
    _step(session, model, opt)
    session.close()
    records = _scalar_records(path)
    assert [(r["seq"], r["scalars"]) for r in records] == [(0, {"eval/ok": 2.0})]
    assert all("eval/loss" not in r.get("scalars", {}) for r in _lines(path))
    assert _case("scalar_log_refusals")["expect"]["written"] == 0


def test_a_bad_value_with_a_good_step_is_refused_before_anything_is_recorded(tmp_path):
    path = tmp_path / "value.jsonl"
    session, opt = _open(_model(), path)
    with pytest.raises(TypeError):
        session.log({"ok": 1.0, "text": "not a number"}, step=3)
    with pytest.raises(ValueError):
        session.log({"ok": 1.0, "vector": torch.ones(3)}, step=3)
    session.log({"after": 1.0}, step=3)
    session.close()
    assert [(r["seq"], r["scalars"]) for r in _scalar_records(path)] == [(0, {"after": 1.0})]


def test_a_numpy_integer_step_is_taken(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "numpy.jsonl"
    session, opt = _open(_model(), path)
    session.log({"a": 1.0}, step=np.int64(12))
    session.close()
    (record,) = _scalar_records(path)
    assert record["step"] == 12


def test_a_key_step_in_the_dictionary_stays_an_ordinary_scalar(tmp_path):
    path = tmp_path / "key.jsonl"
    model = _model()
    session, opt = _open(model, path)
    session.log({"step": 5})
    _step(session, model, opt)
    session.close()
    assert _scalar_records(path) == []
    assert _step_records(path)[0]["scalars"]["step"] == 5


def test_an_empty_call_with_step_writes_nothing(tmp_path):
    path = tmp_path / "empty.jsonl"
    session, opt = _open(_model(), path)
    session.log({}, step=3)
    session.log(step=4)
    session.log({"a": 1.0}, step=5)
    session.close()
    assert [(r["step"], r["seq"]) for r in _scalar_records(path)] == [(5, 0)]


# ---- where it goes, and when it is written ------------------------------------------------------


def test_a_value_without_step_that_a_step_in_memory_takes_is_never_a_scalar_only_record(tmp_path):
    path = tmp_path / "taken.jsonl"
    model = _model()
    session, opt = _open(model, path, every=2)
    for i in range(3):
        _step(session, model, opt)
        with session.phase("eval"):
            session.log({"eval/loss": float(i)})  # the step it followed, still in memory
    session.log({"pending": 1.0})  # pending at close: on the newest step
    session.close()
    assert _scalar_records(path) == []
    steps = _step_records(path)
    assert [r["scalars"]["eval/loss"] for r in steps] == [0.0, 1.0, 2.0]
    assert steps[-1]["scalars"]["pending"] == 1.0


def test_an_explicit_step_is_a_scalar_only_record_even_naming_the_step_in_memory(tmp_path):
    """The capture writes one for every explicit `step=`, also one naming a step still in memory,
    under "eval" or not: the step's own record is unchanged."""
    path = tmp_path / "explicit.jsonl"
    model = _model()
    session, opt = _open(model, path, every=4)
    _step(session, model, opt, value=1.0)
    with session.phase("eval"):
        session.log({"eval/loss": 0.5}, step=0)
    session.log({"other": 2.0}, step=1)
    _step(session, model, opt, value=3.0)
    session.close()
    assert [(r["step"], r["scalars"]) for r in _scalar_records(path)] == [(0, {"eval/loss": 0.5}),
                                                                          (1, {"other": 2.0})]
    steps = _step_records(path)
    assert [r["scalars"] for r in steps] == [{"train/loss": 1.0}, {"train/loss": 3.0}]


def test_it_is_written_with_the_next_block_of_step_records(tmp_path):
    path = tmp_path / "block.jsonl"
    model = _model()
    session, opt = _open(model, path, every=4)
    _step(session, model, opt)
    _step(session, model, opt)
    session.log({"eval/loss": 1.0}, step=1)
    time.sleep(0.2)
    assert _lines(path) == []  # pending until the next block
    _step(session, model, opt)
    _step(session, model, opt)  # the ring fills: steps 0 to 2 written, step 3 kept back
    lines = _wait_for(path, 4)
    assert [r.get("record", r["step"]) for r in lines] == [0, 1, 2, "scalars"]
    session.close()
    lines = _lines(path)
    assert [r.get("record", r["step"]) for r in lines] == [0, 1, 2, "scalars", 3]


@pytest.mark.parametrize("every", [1, 3])
def test_with_a_held_step_it_waits_for_the_block_that_is_put(tmp_path, every):
    """At `every=1` a filled ring puts nothing (the step is kept back): the record goes with the held
    step's block when the next step ends it. Nothing is lost and nothing is written twice."""
    path = tmp_path / f"held{every}.jsonl"
    model = _model()
    session, opt = _open(model, path, every=every)
    for i in range(5):
        _step(session, model, opt)
        session.log({"eval/i": float(i)}, step=100 + i)
    session.close()
    records = _scalar_records(path)
    assert [(r["step"], r["seq"]) for r in records] == [(100 + i, i) for i in range(5)]
    assert [r["step"] for r in _step_records(path)] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("trained", [0, 3])
def test_at_close_with_no_block_left_it_is_written_alone(tmp_path, trained):
    """Never trained, or after `flush()` with no step since: no block of step records follows, so
    `close()` writes the record on its own."""
    path = tmp_path / f"alone{trained}.jsonl"
    model = _model()
    session, opt = _open(model, path, every=4)
    for _ in range(trained):
        _step(session, model, opt)
    if trained:
        session.flush()
    session.log({"eval/loss": 2.5}, step=trained)
    session.close()
    assert [r["step"] for r in _step_records(path)] == list(range(trained))
    assert [(r["step"], r["scalars"]) for r in _scalar_records(path)] == [(trained, {"eval/loss": 2.5})]
    assert path.with_name("alone%d.end.json" % trained).exists()  # a clean close


def test_a_session_with_routers_writes_a_record_alone_at_close(tmp_path, toy_moe):
    """The block that carries the record alone has no steps: decoding it reads no router counts."""
    path = tmp_path / "routers.jsonl"
    session = tensorquill.init(toy_moe, sink=JsonlSink(path), every=4, run_id="al5_moe", heartbeat_seconds=None,
                               stack_seconds=None, routers={"router": 4})
    session.log({"eval/loss": 1.0}, step=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.close()
    assert not session._stopped, [str(w.message) for w in caught]
    assert [r["scalars"] for r in _scalar_records(path)] == [{"eval/loss": 1.0}]


def test_a_stop_of_capture_loses_a_pending_one(tmp_path):
    """As it loses the steps not yet handed to the writer (section 4.2)."""
    model = _model()
    sink = MemorySink()
    session = tensorquill.init(model, sink=sink, every=4, run_id="al5_stop", heartbeat_seconds=None,
                               stack_seconds=None)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    _step(session, model, opt)
    session.log({"eval/loss": 1.0}, step=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session._stop(RuntimeError("stopped by the test"), reason="flush_error")
        session.log({"eval/later": 1.0}, step=1)  # a stopped session takes nothing
        session.close()
    assert sink.records == []
    assert session._scalar_records == []


# ---- a tensor value ---------------------------------------------------------------------------


class _NoRead:
    """Makes any read of a tensor's value raise inside the block: a sync on a device."""

    NAMES = ("item", "tolist", "cpu", "numpy", "__float__", "__int__", "__bool__")

    def __enter__(self):
        self.saved = {name: getattr(torch.Tensor, name) for name in self.NAMES}

        def refuse(*args, **kwargs):
            raise AssertionError("a tensor was read")

        for name in self.NAMES:
            setattr(torch.Tensor, name, refuse)
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(torch.Tensor, name, value)


def test_a_tensor_is_copied_on_its_device_and_read_with_the_next_flush(tmp_path, monkeypatch):
    path = tmp_path / "tensor.jsonl"
    model = _model()
    session, opt = _open(model, path, every=2)
    calls = []
    real = capture._scalars_to_cpu
    monkeypatch.setattr(capture, "_scalars_to_cpu", lambda rows: calls.append(len(rows)) or real(rows))
    _step(session, model, opt, value=1.0)
    value = torch.tensor(0.75)
    with _NoRead():
        session.log({"eval/loss": value}, step=0)  # no read of the value: no sync
    assert calls == []
    pending = session._scalar_records[-1]["scalars"]["eval/loss"]
    assert isinstance(pending, torch.Tensor) and pending is not value  # a copy, on the value's device
    value.fill_(9.0)  # changed after the call: the record keeps what was logged
    _step(session, model, opt, value=2.0)  # the ring fills: the block is put, and the copy read with it
    assert calls  # read at that flush, with that flush's records
    session.close()
    (record,) = _scalar_records(path)
    assert record["scalars"]["eval/loss"] == 0.75


def test_a_record_with_no_tensor_adds_no_transfer(tmp_path, monkeypatch):
    """A record of Python numbers is not moved at all: the block's own transfers are the same as
    without it."""
    real = capture._scalars_to_cpu

    def run(with_record: bool) -> list[int]:
        calls = []
        monkeypatch.setattr(capture, "_scalars_to_cpu", lambda rows: calls.append(len(rows)) or real(rows))
        model = _model()
        session, opt = _open(model, tmp_path / f"t{with_record}.jsonl", every=2)
        for i in range(3):
            _step(session, model, opt, value=float(i))
            if with_record:
                session.log({"eval/loss": 0.5}, step=i)
        session.close()
        return calls

    assert run(True) == run(False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_an_mps_tensor_is_not_read_when_it_is_logged(tmp_path):
    path = tmp_path / "mps.jsonl"
    model = _model().to("mps")
    session = tensorquill.init(model, sink=JsonlSink(path), every=4, run_id="al5_mps", heartbeat_seconds=None,
                               stack_seconds=None)
    with _NoRead():
        session.log({"eval/loss": torch.tensor(0.5, device="mps")}, step=3)
    kept = session._scalar_records[-1]["scalars"]["eval/loss"]
    assert isinstance(kept, torch.Tensor) and kept.device.type == "mps"
    session.close()
    (record,) = _scalar_records(path)
    assert record["scalars"]["eval/loss"] == pytest.approx(0.5)


# ---- scalar_baseline_and_step_0_share_a_point: what the capture writes --------------------------


def test_a_baseline_at_step_0_and_step_0s_own_evaluation_are_both_written(tmp_path):
    """`scalar_baseline_and_step_0_share_a_point`: a baseline logged with `step=0` before training,
    then step 0, an evaluation after it under the same name (on step 0's own record), then step 1.
    The capture writes the case's records, but for what the case leaves out (the stamps, the attempt,
    the step records' other keys)."""
    case = _case("scalar_baseline_and_step_0_share_a_point")
    path = tmp_path / "share.jsonl"
    model = _model()
    session, opt = _open(model, path, every=4, run_id="al_share")
    session.log({"eval/loss": 3.0}, step=0)
    _step(session, model, opt, value=2.9)
    with session.phase("eval"):
        session.log({"eval/loss": 2.8})
    _step(session, model, opt, value=2.7)
    session.close()
    got = _lines(path)
    want = case["records"]
    assert len(got) == len(want)
    scalar_only = [r for r in got if r.get("record") == "scalars"]
    assert [{k: r[k] for k in KEYS if k not in ("ts", "attempt")} for r in scalar_only] == [
        {k: r[k] for k in KEYS if k not in ("ts", "attempt")} for r in want if r.get("record") == "scalars"]
    steps = {r["step"]: r["scalars"] for r in got if r.get("record") != "scalars"}
    assert steps == {r["step"]: r["scalars"] for r in want if r.get("record") != "scalars"}
    assert steps[0]["eval/loss"] == case["expect"]["step_0_record_eval_loss"]


# ---- section 7: a run that logs no step= writes the same bytes ---------------------------------


def _frozen_run(path: Path, monkeypatch, with_records: bool) -> None:
    clock = type("Clock", (), {k: staticmethod(getattr(time, k)) for k in ("monotonic", "perf_counter", "sleep")})
    clock.time = staticmethod(lambda: 1_790_000_000.0)
    monkeypatch.setattr(capture, "time", clock)
    monkeypatch.setattr(capture, "_next_attempt", lambda: 1_790_000_000_000)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    session = tensorquill.init(model, sink=JsonlSink(path), run_id="al5_bytes", heartbeat_seconds=None,
                               stack_seconds=None, every=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(7):
        loss = model(torch.full((2, 4), float(i))).pow(2).mean()
        loss.backward()
        session.log({"loss": loss.detach(), "i": i})
        if with_records:
            session.log({"eval/loss": loss.detach() * 2, "eval/i": i}, step=i)
        opt.step()
        opt.zero_grad()
        if i == 3:
            with session.phase("eval"):
                session.log({"eval/late": 0.5})
            if with_records:
                session.log({"eval/explicit": 1.5}, step=3)
    session.close()


def test_a_run_that_logs_no_step_writes_the_same_step_lines(tmp_path, monkeypatch):
    """Section 7's invariant, with no git history: the same frozen run, once with the feature unused
    and once with `step=` calls, gives byte-identical step lines; the second only adds lines."""
    plain, mixed = tmp_path / "plain.jsonl", tmp_path / "mixed.jsonl"
    _frozen_run(plain, monkeypatch, with_records=False)
    _frozen_run(mixed, monkeypatch, with_records=True)
    plain_lines = plain.read_bytes().splitlines(keepends=True)
    mixed_lines = mixed.read_bytes().splitlines(keepends=True)
    kept = [line for line in mixed_lines if b'"record":"scalars"' not in line]
    assert len(plain_lines) == 7 and b'"eval/late"' in plain.read_bytes()
    assert kept == plain_lines
    assert len(mixed_lines) - len(kept) == 8
    assert b'"record"' not in plain.read_bytes() and b'"seq"' not in plain.read_bytes()
