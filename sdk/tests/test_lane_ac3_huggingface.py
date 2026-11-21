"""Lane AC3: a `Trainer.train()` that raises closes its TensorQuill session as raised.

`Trainer` has no event for a failed run, so the callback's session stayed open after a raise:
the newest step was never written, the file had no end marker, and `callback.close()` or the
next `train()` closed it as a clean end. Now the end of the process closes it (an `atexit`
hook held only while a session is open), and every close of a `train()` that did not reach
`on_train_end` writes `outcome: "raised"`, with the exception's type when it came out of this
`train()`, else `TrainEndNotReached`.
"""

import atexit
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

import tensorquill  # noqa: E402
from tensorquill import capture  # noqa: E402
from tensorquill import huggingface as hf  # noqa: E402
from accelerate import Accelerator  # noqa: E402

SRC = str(Path(tensorquill.__file__).resolve().parents[1])


class Model(nn.Module):
    """Raises (`error`) on its `raise_at`-th forward, before that step's backward."""

    def __init__(self, raise_at=None, error=RuntimeError("boom")):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.calls, self.raise_at, self.error = 0, raise_at, error

    def forward(self, x, labels=None):
        self.calls += 1
        if self.calls == self.raise_at:
            raise self.error
        return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}


def _trainer(tmp_path, model, callback, max_steps=8):
    torch.manual_seed(0)
    data = [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(128)]
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "out"), max_steps=max_steps, per_device_train_batch_size=8, report_to="none",
        save_strategy="no", disable_tqdm=True, use_cpu=True, logging_steps=100,
    )
    return transformers.Trainer(model=model, args=args, train_dataset=data, callbacks=[callback])


def _marker(tmp_path, run_id):
    path = tmp_path / "tensorquill-out" / f"{run_id}.rank0.end.json"
    return json.loads(path.read_text()) if path.exists() else None


def _steps(tmp_path, run_id):
    lines = (tmp_path / "tensorquill-out" / f"{run_id}.rank0.jsonl").read_text().splitlines()
    return [json.loads(line)["step"] for line in lines]


_SCRIPT = """
import os, signal, sys, torch
from torch import nn
import transformers
from tensorquill.huggingface import TensorQuillCallback

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.calls = 0
    def forward(self, x, labels=None):
        self.calls += 1
        if self.calls == 6:  # steps 0 to 4 have ended
            if sys.argv[1] == "error":
                raise RuntimeError("boom in forward")
            os.kill(os.getpid(), signal.SIGINT)  # a real Ctrl-C, as the terminal sends it
        return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}

torch.manual_seed(0)
data = [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(128)]
args = transformers.TrainingArguments(output_dir="out", max_steps=20, per_device_train_batch_size=8,
                                      report_to="none", save_strategy="no", disable_tqdm=True, use_cpu=True)
trainer = transformers.Trainer(model=Model(), args=args, train_dataset=data,
                               callbacks=[TensorQuillCallback(run_id="run-" + sys.argv[1], every=1)])
trainer.train()  # raises, uncaught: the process ends with it
"""


@pytest.mark.parametrize("how", ["error", "sigint"])
def test_a_process_that_ends_with_train_s_raise_writes_every_step_and_says_it_raised(tmp_path, how):
    script = tmp_path / "train.py"
    script.write_text(textwrap.dedent(_SCRIPT))
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("TENSORQUILL_ENDPOINT", None)
    done = subprocess.run([sys.executable, str(script), how], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=300)
    # What the raise does to the program is unchanged: its traceback, and its exit status
    # (1 for the error; for Ctrl-C, CPython ends by SIGINT, as a shell's 130 shows).
    if how == "error":
        assert done.returncode == 1, done.stderr
        assert done.stderr.rstrip().endswith("RuntimeError: boom in forward")
    else:
        assert done.returncode == -signal.SIGINT, done.stderr
        assert done.stderr.rstrip().endswith("KeyboardInterrupt")
    run = f"run-{how}"
    # Before: steps 0 to 3 (step 4, the newest, was still held), and no end marker.
    assert _steps(tmp_path, run) == [0, 1, 2, 3, 4]
    marker = _marker(tmp_path, run)
    assert marker is not None and marker["outcome"] == "raised"
    assert marker["exception"] == ("RuntimeError" if how == "error" else "KeyboardInterrupt")
    from tensorquill.runs import _end_text, _local_end

    files = {0: tmp_path / "tensorquill-out" / f"{run}.rank0.jsonl"}
    assert _end_text(_local_end(files)) == "closed, raised"


def test_close_after_a_raise_says_raised_and_leaves_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    registered, unregistered = [], []
    register, unregister = atexit.register, atexit.unregister

    def note(into, call):
        def noted(f, *args, **kwargs):
            if isinstance(getattr(f, "__self__", None), hf.TensorQuillCallback):
                into.append(f)
            return call(f, *args, **kwargs)
        return noted

    monkeypatch.setattr(atexit, "register", note(registered, register))
    monkeypatch.setattr(atexit, "unregister", note(unregistered, unregister))
    originals = (Accelerator.clip_grad_norm_, Accelerator.backward, Accelerator.save_state, Accelerator.load_state)

    # Closed where the exception is handled: the marker names it.
    named = hf.TensorQuillCallback(run_id="named", every=1)
    try:
        _trainer(tmp_path, Model(raise_at=4), named).train()
    except RuntimeError:
        assert named.session is not None  # Trainer has no event for the raise
        named.close()
    assert named.session is None
    assert _marker(tmp_path, "named")["outcome"] == "raised"
    assert _marker(tmp_path, "named")["exception"] == "RuntimeError"
    assert _steps(tmp_path, "named") == [0, 1, 2]

    # Closed after the exception was let go (a caught Ctrl-C, a notebook cell): its type is not
    # known here, and the marker says raised all the same, never a clean end.
    unknown = hf.TensorQuillCallback(run_id="unknown", every=1)
    try:
        _trainer(tmp_path, Model(raise_at=3, error=KeyboardInterrupt()), unknown).train()
    except KeyboardInterrupt:
        pass
    unknown.close()
    unknown.close()
    assert _marker(tmp_path, "unknown")["outcome"] == "raised"
    assert _marker(tmp_path, "unknown")["exception"] == "TrainEndNotReached"

    # Nothing is left patched or registered once the callbacks are closed.
    assert (Accelerator.clip_grad_norm_, Accelerator.backward, Accelerator.save_state,
            Accelerator.load_state) == originals
    assert len(registered) == 2 and unregistered == registered  # bound methods: equal, as atexit compares them
    assert [f.__self__ for f in registered] == [named, unknown]


def test_the_next_train_closes_the_raised_one_as_raised_and_names_only_its_own_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    ends = []
    real = capture.write_end_marker

    def spy(path, **kwargs):
        ends.append(None if kwargs.get("exception") is None else type(kwargs["exception"]).__name__)
        return real(path, **kwargs)

    monkeypatch.setattr(capture, "write_end_marker", spy)
    callback = hf.TensorQuillCallback(run_id="again", every=1)
    model = Model(raise_at=3)
    trainer = _trainer(tmp_path, model, callback, max_steps=4)
    try:
        trainer.train()
    except RuntimeError as error:
        first = error
        # train() again inside the handler: the exception being handled came out of the first
        # train(), so the first session is closed naming it.
        model.raise_at, model.error = model.calls + 2, ValueError("second")
        trainer.state.global_step = 0
        with pytest.raises(ValueError):
            trainer.train()
    assert ends == ["RuntimeError"]
    # The second train() raised too. An exception from the first, handled again, is not its own.
    try:
        raise first
    except RuntimeError:
        callback.close()
    assert ends == ["RuntimeError", "TrainEndNotReached"]
    # A third train() that finishes closes clean.
    model.raise_at = None
    trainer.state.global_step = 0
    trainer.train()
    assert ends == ["RuntimeError", "TrainEndNotReached", None] and callback.session is None


def test_the_exit_hook_closes_only_in_the_process_that_registered_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    callback = hf.TensorQuillCallback(run_id="fork", every=1)
    try:
        _trainer(tmp_path, Model(raise_at=3), callback).train()
    except RuntimeError:
        pass
    pid = callback._pid
    callback._pid = pid + 1  # as a forked DataLoader worker sees it: another process's hook
    callback._at_exit()
    assert callback.session is not None and _marker(tmp_path, "fork") is None
    callback._pid = pid
    callback._at_exit()  # the process's own exit, with the exception gone: raised, type unknown
    assert callback.session is None
    assert (_marker(tmp_path, "fork")["outcome"], _marker(tmp_path, "fork")["exception"]) == (
        "raised", "TrainEndNotReached")


def test_a_train_whose_frame_cannot_be_read_is_still_closed_as_raised(tmp_path, monkeypatch):
    # `inspect.currentframe()` may return None (a Python without frame support): the exception
    # can then not be tied to this train(), and the close still says raised, never a clean end.
    import inspect

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    monkeypatch.setattr(inspect, "currentframe", lambda: None)
    callback = hf.TensorQuillCallback(run_id="frameless", every=1)
    try:
        _trainer(tmp_path, Model(raise_at=3), callback).train()
    except RuntimeError:
        callback.close()
    assert (_marker(tmp_path, "frameless")["outcome"], _marker(tmp_path, "frameless")["exception"]) == (
        "raised", "TrainEndNotReached")
