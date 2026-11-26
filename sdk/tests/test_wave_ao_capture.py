"""Wave AO, lane AO-1: the capture's findings on `453f1e2`.

Each finding's tests were committed failing before its fix. They run on the bare interpreters as
CI does (PyTorch 2.6 and the newest release): nothing here needs a package beyond PyTorch, and the
tests that need Apple's MPS skip where it is not there.
"""

from __future__ import annotations

import json
import numbers
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture, identity, manifest
from tensorquill.evidence import load_records
from tensorquill.manifest import find_manifest
from tensorquill.sink import MemorySink

# ---- SIG-26: the version counter's calling form ---------------------------------------------------
#
# PyTorch 2.6 takes `_unsafe_set_version_counter(tensor, version)`, one tensor at a time; later
# releases take sequences. A call in the wrong form raises a `TypeError` whose text prints each
# tensor (`Tensor.__repr__` runs inside the C call), and an exception raised while it prints, such as
# a job's SIGTERM handler raising `SystemExit`, is lost there. So no step may make a call that fails.


def _counting_repr(monkeypatch):
    calls = []
    original = torch.Tensor.__repr__

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "__repr__", spy)
    return calls


def _recording_setter(monkeypatch):
    """Wrap the private setter: each call's arguments' kind and whether it raised."""
    original = torch._C._autograd._unsafe_set_version_counter
    calls = []

    def setter(*args):
        try:
            original(*args)
        except BaseException as error:
            calls.append((type(args[0]).__name__, type(error).__name__))
            raise
        calls.append((type(args[0]).__name__, None))

    monkeypatch.setattr(torch._C._autograd, "_unsafe_set_version_counter", setter)
    return calls


def test_restoring_a_version_prints_no_tensor(monkeypatch):
    tensor = torch.zeros(3)
    tensor.add_(1)
    assert tensor._version == 1
    calls = _counting_repr(monkeypatch)
    capture._restore_versions([tensor], [0])
    assert calls == []
    assert tensor._version == 0


def test_restoring_versions_makes_no_call_that_fails(monkeypatch):
    tensors = [torch.zeros(2), torch.ones(4)]
    for tensor in tensors:
        tensor.add_(1)
        tensor.add_(1)
    capture._restore_versions(tensors, [0, 1])  # the first call may choose the form; never by failing
    calls = _recording_setter(monkeypatch)
    for _ in range(3):
        for tensor in tensors:
            tensor.add_(1)
        capture._restore_versions(tensors, [0, 1])
    assert [t._version for t in tensors] == [0, 1]
    assert calls and all(error is None for _, error in calls)


def test_an_exception_raised_while_a_step_restores_versions_comes_out(monkeypatch):
    """The SIGTERM case made deterministic: whatever runs Python code during the restore (here
    `Tensor.__repr__`, as on PyTorch 2.6 before the fix) raising `SystemExit` must not be lost. With
    nothing printed, the restore runs no Python code at all, so the training's own handler is the
    only place such an exit can come from, and it comes out of the step."""
    tensor = torch.zeros(3)
    tensor.add_(1)
    raised = []

    def raising(self, *args, **kwargs):
        raised.append(1)
        raise SystemExit(143)

    monkeypatch.setattr(torch.Tensor, "__repr__", raising)
    try:
        capture._restore_versions([tensor], [0])
        came_out = False
    except SystemExit:
        came_out = True
    assert came_out or not raised, "a SystemExit raised during the restore was lost"
    assert raised == [] and tensor._version == 0


# ---- BUG-05: `start_step` and `every` are integers, as `log`'s `step=` is ------------------------
#
# A bool is an `int` in Python, so `start_step=True` was taken and step 0 was written `"step": true`,
# which every reader refuses. `init` refuses them by `log`'s rule for `step=` (`_explicit_step`): a
# bool, a float or anything not an integer raises `TypeError`, a number below the least `ValueError`,
# each naming the argument, before anything is opened.


def _init_in(tmp_path, monkeypatch, **options):
    monkeypatch.chdir(tmp_path)
    return tensorquill.init(nn.Linear(2, 2), run_id="r", heartbeat_seconds=None, **options)


@pytest.mark.parametrize("value", [True, False, 1.0, 0.0, "1", None])
def test_init_refuses_a_start_step_that_is_not_an_integer(tmp_path, monkeypatch, value):
    with pytest.raises(TypeError, match=r"start_step must be an integer of 0 or more"):
        _init_in(tmp_path, monkeypatch, start_step=value)
    assert list(tmp_path.iterdir()) == []  # nothing opened, nothing written


def test_init_refuses_a_negative_start_step(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"start_step must be an integer of 0 or more, not -1"):
        _init_in(tmp_path, monkeypatch, start_step=-1)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [True, False, 2.0, "2", None])
def test_init_refuses_an_every_that_is_not_an_integer(tmp_path, monkeypatch, value):
    with pytest.raises(TypeError, match=r"every must be an integer of 1 or more"):
        _init_in(tmp_path, monkeypatch, every=value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [0, -3])
def test_init_refuses_an_every_below_one(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match=r"every must be at least 1"):
        _init_in(tmp_path, monkeypatch, every=value)
    assert list(tmp_path.iterdir()) == []


class _Index:
    """An integer that is not an `int`, as a NumPy integer is: `numbers.Integral`."""

    def __init__(self, value):
        self.value = value

    def __int__(self):
        return self.value

    __index__ = __int__


numbers.Integral.register(_Index)


@pytest.mark.parametrize("start_step", [0, 3, _Index(3)])
def test_a_start_step_that_is_an_integer_is_taken_and_written_as_an_int(start_step):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = MemorySink()
    with tensorquill.init(model, sink=sink, every=_Index(2), start_step=start_step, heartbeat_seconds=None):
        for _ in range(3):
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
    first = int(start_step)
    assert [r["step"] for r in sink.records] == [first, first + 1, first + 2]
    assert all(type(r["step"]) is int for r in sink.records)
    for record in sink.records:
        identity.check(record)  # every reader takes the record: raises if it does not
    load_records(sink.records)


def test_logs_step_rule_and_its_message_are_unchanged():
    session = tensorquill.init(nn.Linear(2, 2), sink=MemorySink(), heartbeat_seconds=None)
    try:
        with pytest.raises(TypeError, match=r"^session\.log\(\.\.\., step=\.\.\.\): step must be an integer"):
            session.log({"a": 1.0}, step=True)
        with pytest.raises(ValueError, match=r"^session\.log\(\.\.\., step=\.\.\.\): step must be an integer"):
            session.log({"a": 1.0}, step=-1)
    finally:
        session.close()


# ---- BUG-26, the clipping wrapper: a parameter unfrozen before `clip_grad_norm_` ------------------
#
# A parameter frozen at `init` joins the session's trainable ones at the optimizer's step
# (`_take_unfrozen`), after `clip_grad_norm_` ran, so the wrapper did not match it on the step it was
# unfrozen and `clip_grad_norm` was missing there. It counts once it is trainable, as at that step.


def _one_step(model, optimizer, clip, *, inputs=None):
    model(torch.ones(2, 4) if inputs is None else inputs).sum().backward()
    norm = clip()
    optimizer.step()
    optimizer.zero_grad()
    return norm


def _quiet_init(model, sink, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # every parameter frozen at `init` is warned of
        return tensorquill.init(model, sink=sink, every=1, heartbeat_seconds=None, **options)


def test_a_parameter_unfrozen_just_before_clipping_has_its_norm_recorded_on_that_step():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = MemorySink()
    with _quiet_init(model, sink):
        for p in model.parameters():
            p.requires_grad_(True)
        norm = _one_step(model, optimizer, lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
        _one_step(model, optimizer, lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
    assert [r["step"] for r in sink.records] == [0, 1]
    assert sink.records[0].get("clip_grad_norm") == pytest.approx(float(norm), rel=1e-6)
    assert "clip_grad_norm" in sink.records[1]


def test_a_lora_like_model_clipping_only_its_newly_unfrozen_part_is_recorded():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for p in model[1].parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sink = MemorySink()
    with _quiet_init(model, sink):
        for p in model[1].parameters():
            p.requires_grad_(True)
        norm = _one_step(model, optimizer, lambda: torch.nn.utils.clip_grad_norm_(model[1].parameters(), 0.5))
    assert sink.records[0].get("clip_grad_norm") == pytest.approx(float(norm), rel=1e-6)


def test_a_clip_over_parameters_still_frozen_or_of_another_model_records_nothing():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for p in model[1].parameters():
        p.requires_grad_(False)
    other = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model[0].parameters(), lr=0.01)
    sink = MemorySink()
    with _quiet_init(model, sink):
        def clip():
            torch.nn.utils.clip_grad_norm_(model[1].parameters(), 0.5)  # frozen still
            other(torch.ones(1, 4)).sum().backward()
            torch.nn.utils.clip_grad_norm_(other.parameters(), 0.5)  # not this model's
        _one_step(model, optimizer, clip)
    assert "clip_grad_norm" not in sink.records[0]


# ---- BUG-03: a logged value that cannot be read never reaches `close()` or `flush()` --------------
#
# A tensor on `meta` holds no values. `log` took one, and the transfer of the scalar-only records at
# `close()` and `flush()`, made outside the guard of the ring's own transfer, raised: inside `with
# session:` that replaced the training's own exception, no end marker was written and the writer thread
# stayed alive. `log` now refuses such a value by the rule the session judges any tensor by (its device
# and layout, no data read: `_grad_unread`, `_output_unread`), and a transfer that fails for another
# reason (a stand-in here) leaves that value out on its own.


def _jsonl_session(tmp_path, name="r"):
    path = tmp_path / f"{name}.rank0.jsonl"
    session = tensorquill.init(nn.Linear(2, 2), run_id=name, sink=tensorquill.JsonlSink(path),
                               heartbeat_seconds=None)
    return session, path


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("explicit_step", [True, False])
def test_log_refuses_a_tensor_on_meta_before_anything_is_recorded(tmp_path, explicit_step):
    session, path = _jsonl_session(tmp_path)
    value = torch.tensor(1.5, device="meta")
    with pytest.raises(ValueError, match=r"loss: the tensor is on meta, which holds no values"):
        if explicit_step:
            session.log({"loss": value, "ok": 1.0}, step=0)
        else:
            session.log({"loss": value, "ok": 1.0})
    assert session._scalar_records == [] and session._pending_scalars == {}
    session.close()
    assert session.writer_error is None and not session._worker.is_alive()
    assert path.with_name("r.rank0.end.json").exists()


def test_with_a_meta_value_logged_the_trainings_own_exception_comes_out(tmp_path):
    session, path = _jsonl_session(tmp_path)
    with pytest.raises(ValueError, match="the training's own failure"):
        with session:
            try:
                session.log({"loss": torch.tensor(1.5, device="meta")}, step=0)
            except ValueError:
                pass  # refused at the call
            raise ValueError("the training's own failure")
    assert not session._worker.is_alive()
    assert path.with_name("r.rank0.end.json").exists()


class _Unreadable(torch.Tensor):
    """A tensor whose transfer to the CPU fails, for a reason other than `meta` (a stand-in: no device
    here fails that way on purpose). Put into a pending record by hand, past `log`'s checks."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.stack:
            raise RuntimeError("stand-in: the copy off the device failed")
        return super().__torch_function__(func, types, args, kwargs or {})


def _log_with_an_unreadable_value(session, step=0):
    session.log({"eval/loss": torch.tensor(0.25, dtype=torch.float64), "eval/n": 4}, step=step)
    bad = torch.tensor(2.0, dtype=torch.float64).as_subclass(_Unreadable)
    session._scalar_records[-1]["scalars"]["eval/bad"] = bad


def test_a_value_whose_transfer_fails_at_flush_is_left_out_and_flush_does_not_raise(tmp_path):
    session, path = _jsonl_session(tmp_path)
    _log_with_an_unreadable_value(session)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        session.flush()
    session.close()
    told = [str(w.message) for w in said if "eval/bad" in str(w.message)]
    assert len(told) == 1 and "stand-in: the copy off the device failed" in told[0]
    (record,) = [r for r in _lines(path) if r.get("record") == "scalars"]
    assert record["scalars"] == {"eval/loss": 0.25, "eval/n": 4}
    assert session.writer_error is None and not session._worker.is_alive()
    assert path.with_name("r.rank0.end.json").exists()


def test_a_value_whose_transfer_fails_at_close_leaves_the_end_marker_and_stops_the_writer(tmp_path):
    session, path = _jsonl_session(tmp_path)
    _log_with_an_unreadable_value(session, step=3)
    session.log({"only/bad": 1.0}, step=4)
    session._scalar_records[-1]["scalars"]["only/bad"] = torch.tensor(1.0).as_subclass(_Unreadable)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.close()  # never raises
    records = [r for r in _lines(path) if r.get("record") == "scalars"]
    assert [(r["step"], r["scalars"]) for r in records] == [(3, {"eval/loss": 0.25, "eval/n": 4})]
    assert session.writer_error is None and not session._worker.is_alive()
    assert path.with_name("r.rank0.end.json").exists()


def test_with_a_value_whose_transfer_fails_the_trainings_own_exception_comes_out(tmp_path):
    session, path = _jsonl_session(tmp_path)
    with pytest.raises(KeyError, match="the training's own failure"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with session:
                _log_with_an_unreadable_value(session)
                raise KeyError("the training's own failure")
    assert not session._worker.is_alive()
    assert path.with_name("r.rank0.end.json").exists()


# ---- BUG-06: a run ID never leads a file out of the output folder --------------------------------
#
# The run ID names the files: `tensorquill-out/<run_id>.rank<N>.jsonl` and `<run_id>.run.json` beside
# the records file. One holding a path separator wrote outside the folder (`../../escaped`) or into a
# subfolder whose manifest landed in a doubled one (`sweep1/trial2`). An explicit `run_id` that could
# do that is refused at `init`, before anything is opened; one from the environment or a tracker is
# made to fit, with a warning, as a project name from there is (`manifest.resolve_project`).

_LEAVING = ["../../escaped", "sweep1/trial2", "/abs/path", "a\\b", "C:x", "a\nb", "a\x00b", "tab\there",
            "del\x7f", ".", ".."]


def _files_under(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _work_folder(tmp_path, monkeypatch):
    work = tmp_path / "work" / "inner"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    for name in ("MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "TENSORQUILL_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    return work


@pytest.mark.parametrize("run_id", _LEAVING)
def test_an_explicit_run_id_that_could_leave_the_folder_is_refused_before_anything_is_opened(
        tmp_path, monkeypatch, run_id):
    _work_folder(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"^run_id .* cannot be used: "):
        tensorquill.init(nn.Linear(2, 2), run_id=run_id, project="p", heartbeat_seconds=None)
    assert _files_under(tmp_path) == []


@pytest.mark.parametrize("run_id", _LEAVING)
def test_an_explicit_run_id_that_could_leave_the_folder_is_refused_with_a_sink_given_too(tmp_path, run_id):
    path = tmp_path / "given.rank0.jsonl"
    with pytest.raises(ValueError, match=r"^run_id .* cannot be used: "):
        tensorquill.init(nn.Linear(2, 2), run_id=run_id, sink=tensorquill.JsonlSink(path), heartbeat_seconds=None)
    assert [p for p in tmp_path.rglob("*") if p.is_file() and p.stat().st_size] == []


_FITTED = [("../../escaped", "..-..-escaped"), ("sweep1/trial2", "sweep1-trial2"), ("a\\b:c", "a-b-c"),
           ("a\nb", "a-b"), ("..", "-")]


# MLflow's ID is taken as it is; torchrun's rendezvous ID becomes `run-<id>` (`tracker._launcher_run_id`),
# so `..` there is `run-..`, which stays in the folder.
@pytest.mark.parametrize("variable, value, found, fitted", [
    *(("MLFLOW_RUN_ID", value, value, fitted) for value, fitted in _FITTED),
    *(("TORCHELASTIC_RUN_ID", value, f"run-{value}", f"run-{fitted}") for value, fitted in _FITTED if value != ".."),
])
def test_a_run_id_from_the_environment_is_made_to_fit_with_a_warning(tmp_path, monkeypatch, variable, value, found,
                                                                       fitted):
    work = _work_folder(tmp_path, monkeypatch)
    monkeypatch.setenv(variable, value)
    with pytest.warns(RuntimeWarning, match=r"is recorded under") as said:
        with tensorquill.init(nn.Linear(2, 2), project="p", heartbeat_seconds=None) as session:
            records_file = session._sink.path
    assert session.run.run_id == fitted
    assert any(repr(found) in str(w.message) and repr(fitted) in str(w.message) for w in said)
    out = work / "tensorquill-out"
    assert records_file == Path("tensorquill-out") / f"{fitted}.rank0.jsonl"
    assert _files_under(tmp_path) == sorted(
        str((out / name).relative_to(tmp_path)) for name in
        (f"{fitted}.rank0.jsonl", f"{fitted}.rank0.end.json", f"{fitted}.run.json"))
    found_manifest = find_manifest(out / f"{fitted}.rank0.jsonl")
    assert found_manifest is not None and found_manifest["run_id"] == fitted


@pytest.mark.parametrize("run_id", ["run-slurm1234.0", "exp 1", "résumé_7", "a..b", "...x", "UPPER-lower_09"])
def test_a_run_id_that_stays_in_the_folder_is_taken_as_given(tmp_path, monkeypatch, run_id):
    work = _work_folder(tmp_path, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with tensorquill.init(nn.Linear(2, 2), run_id=run_id, project="p", heartbeat_seconds=None) as session:
            pass
    assert session.run.run_id == run_id
    assert (work / "tensorquill-out" / f"{run_id}.rank0.jsonl").exists()
    assert (work / "tensorquill-out" / f"{run_id}.run.json").exists()


def test_the_run_id_rule_is_one_pure_function_beside_the_project_rule():
    assert manifest.run_id_problem("ok-1.2_3") is None
    for bad in _LEAVING:
        assert manifest.run_id_problem(bad)
        fitted = manifest.conform_run_id(bad)
        assert manifest.run_id_problem(fitted) is None
        assert manifest.conform_run_id(bad) == fitted  # every rank makes the same ID of the same one
