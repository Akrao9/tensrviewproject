"""Lane AD6: the Hugging Face callback, a `close()` made on purpose during training, and an
interrupted `Trainer` on two ranks.

Lane AC3 made every close of a `train()` that had not reached `on_train_end` say it raised. That
included a `callback.close()` the user makes on purpose while training runs (from another
callback, or another thread), which then read as a crash. Now such a close is a clean close:
the callback tells it by the `train()` that opened the session still running on one of the
process's threads. A close after that `train()` stopped is still a raise (lane AC3's tests).

The last test runs a real `Trainer` on two CPU ranks under torchrun (gloo), interrupted as lane
AC3's test interrupts one process, and shows what each rank's records and end marker say.
"""

from __future__ import annotations

import gc
import json
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import threading
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
    def __init__(self, raise_at=None):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.calls, self.raise_at = 0, raise_at

    def forward(self, x, labels=None):
        self.calls += 1
        if self.calls == self.raise_at:
            raise RuntimeError("boom")
        return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}


class CloseAt(transformers.TrainerCallback):
    """The user's own callback: closes TensorQuill's at the end of the step `at` (global_step),
    on this thread or, with `thread=True`, from another thread while `train()` waits for it."""

    def __init__(self, target, at, thread=False):
        self.target, self.at, self.thread, self.closed_at = target, at, thread, None

    def on_step_end(self, args, state, control, **_):
        if state.global_step == self.at and self.target.session is not None:
            if self.thread:
                worker = threading.Thread(target=self.target.close)
                worker.start()
                worker.join()
            else:
                self.target.close()
            self.closed_at = state.global_step


def _trainer(tmp_path, model, callbacks, max_steps=8):
    torch.manual_seed(0)
    data = [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(128)]
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "out"), max_steps=max_steps, per_device_train_batch_size=8, report_to="none",
        save_strategy="no", disable_tqdm=True, use_cpu=True, logging_steps=100,
    )
    return transformers.Trainer(model=model, args=args, train_dataset=data, callbacks=callbacks)


def _marker(tmp_path, run_id, rank=0):
    path = tmp_path / "tensorquill-out" / f"{run_id}.rank{rank}.end.json"
    return json.loads(path.read_text()) if path.exists() else None


def _steps(tmp_path, run_id, rank=0):
    path = tmp_path / "tensorquill-out" / f"{run_id}.rank{rank}.jsonl"
    return [json.loads(line)["step"] for line in path.read_text().splitlines()] if path.exists() else None


def _spy_ends(monkeypatch):
    ends = []
    real = capture.write_end_marker

    def spy(path, **kwargs):
        ends.append(None if kwargs.get("exception") is None else type(kwargs["exception"]).__name__)
        return real(path, **kwargs)

    monkeypatch.setattr(capture, "write_end_marker", spy)
    return ends


def test_a_close_made_on_purpose_during_training_is_a_close_and_a_later_raise_is_still_a_raise(
        tmp_path, monkeypatch):
    """Fail before: the end marker said `raised`, `TrainEndNotReached`, for a close the user made."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    originals = (Accelerator.clip_grad_norm_, Accelerator.backward, Accelerator.save_state, Accelerator.load_state)
    callback = hf.TensorQuillCallback(run_id="on-purpose", every=1)
    closer = CloseAt(callback, at=3)
    trainer = _trainer(tmp_path, Model(), [callback, closer])
    trainer.train()
    assert closer.closed_at == 3 and trainer.state.global_step == 8  # training went on to its end
    marker = _marker(tmp_path, "on-purpose")
    assert (marker["outcome"], marker.get("exception")) == ("closed", None), marker
    assert _steps(tmp_path, "on-purpose") == [0, 1, 2]  # what ran while the session was open
    from tensorquill.runs import _end_text, _local_end

    assert _end_text(_local_end({0: tmp_path / "tensorquill-out" / "on-purpose.rank0.jsonl"})) == "closed"
    assert callback.session is None and callback._trainer is None
    assert (Accelerator.clip_grad_norm_, Accelerator.backward, Accelerator.save_state,
            Accelerator.load_state) == originals

    # The same callback later: a train() that raises, the exception let go, and the next train().
    # The frame of the train() that raised is gone, and the next one may be given its id: the
    # session it left is closed as raised, never as a close made on purpose.
    ends = _spy_ends(monkeypatch)
    again = hf.TensorQuillCallback(run_id="raised-then-again", every=1)
    model = Model(raise_at=3)
    trainer = _trainer(tmp_path, model, [again], max_steps=4)
    try:
        trainer.train()
    except RuntimeError:
        pass
    gc.collect()
    assert again.session is not None and ends == []
    model.raise_at = None
    trainer.state.global_step = 0
    trainer.train()
    assert ends == ["TrainEndNotReached", None]


def test_a_close_from_another_thread_while_train_runs_is_a_close(tmp_path, monkeypatch):
    """Fail before: `raised`, `TrainEndNotReached`. The `train()` runs on the main thread while
    another thread closes the session; the callback reads every thread's stack."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_ENDPOINT", raising=False)
    callback = hf.TensorQuillCallback(run_id="from-a-thread", every=1)
    closer = CloseAt(callback, at=2, thread=True)
    _trainer(tmp_path, Model(), [callback, closer]).train()
    marker = _marker(tmp_path, "from-a-thread")
    assert closer.closed_at == 2 and (marker["outcome"], marker.get("exception")) == ("closed", None), marker
    assert _steps(tmp_path, "from-a-thread") == [0, 1]

    # Where the training frame cannot be read, a close during training cannot be told from one
    # after a raise: it is closed as raised, as lane AC3 closes it (a limit, not a clean close).
    import inspect

    monkeypatch.setattr(inspect, "currentframe", lambda: None)
    blind = hf.TensorQuillCallback(run_id="frameless", every=1)
    _trainer(tmp_path, Model(), [blind, CloseAt(blind, at=2)]).train()
    marker = _marker(tmp_path, "frameless")
    assert (marker["outcome"], marker["exception"]) == ("raised", "TrainEndNotReached"), marker


# --- Two ranks under torchrun ------------------------------------------------------------------

_SCRIPT = """
import os, signal, sys, torch
from torch import nn
import transformers
from tensorquill.huggingface import TensorQuillCallback

how, who = sys.argv[1], sys.argv[2]  # error | sigint | close; "all" or one rank
rank = int(os.environ["RANK"])
mine = who == "all" or int(who) == rank

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        self.calls = 0
    def forward(self, x, labels=None):
        self.calls += 1
        if self.calls == 6 and mine and how != "close":  # steps 0 to 4 have ended
            if how == "error":
                raise RuntimeError(f"boom in forward on rank {rank}")
            os.kill(os.getpid(), signal.SIGINT)  # a real Ctrl-C, as the terminal sends it
        return {"loss": (self.net(x).squeeze(-1) - labels).pow(2).mean()}

class CloseAt(transformers.TrainerCallback):
    def __init__(self, target):
        self.target = target
    def on_step_end(self, args, state, control, **_):
        if how == "close" and mine and state.global_step == 3:
            self.target.close()

torch.manual_seed(0)
data = [{"x": torch.randn(4), "labels": torch.randn(())} for _ in range(256)]
args = transformers.TrainingArguments(output_dir="out", max_steps=20, per_device_train_batch_size=8,
                                      report_to="none", save_strategy="no", disable_tqdm=True, use_cpu=True,
                                      ddp_backend="gloo")
callback = TensorQuillCallback(run_id="run-" + how, every=1)
trainer = transformers.Trainer(model=Model(), args=args, train_dataset=data, callbacks=[callback, CloseAt(callback)])
assert trainer.args.world_size == 2, trainer.args.world_size
trainer.train()
print(f"rank {rank} trained to step {trainer.state.global_step}", flush=True)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _exit_codes(stderr: str) -> dict[int, int]:
    """Each failed rank's exit code, from torchrun's summary ("rank : R ..." then "exitcode : C").
    Empty when torchrun printed none: on this Mac it raised `PermissionError` from `os.killpg` in its
    own shutdown in 2 of 24 runs, after every rank had ended."""
    return {int(r): int(c) for r, c in re.findall(r"rank\s*:\s*(\d+)[^\n]*\n\s*exitcode\s*:\s*(-?\d+)", stderr)}


@pytest.mark.parametrize("how,who", [("error", "all"), ("sigint", "all"), ("close", "all"), ("error", "1")])
def test_an_interrupted_trainer_on_two_ranks_under_torchrun(tmp_path, how, who):
    """Measured on this Mac before the test was written (lane AD6's report): 16 runs with both
    ranks raising, 16 with both interrupted by SIGINT, 7 with rank 1 alone raising, 3 with rank 1
    alone interrupted, 3 with rank 0 alone raising; the records and markers were the same in every
    run of a case. Fail before: the `close` case only (each rank's marker said `raised`).

    - Both ranks interrupted at the same step, by an error or by SIGINT: each rank writes steps 0
      to 4 and an end marker `raised` with the exception's type. A rank still exiting when the
      other has ended may be stopped by torchrun after its exit hook ran (its status -15).
    - Both ranks closing on purpose at step 3: each writes steps 0 to 2 and `closed`; training
      goes on to its end and the job exits 0.
    - Rank 1 alone raising: rank 1 as above. Rank 0 is then waiting in the gradient all-reduce,
      and torchrun stops it (SIGTERM on this Mac) before its exit hook can run: its newest step
      is lost and it leaves no end marker, so it reads as not closed. A limit (lane AD6's report):
      asserted here only as never a clean close, with its steps a prefix of rank 1's."""
    script = tmp_path / "train.py"
    script.write_text(textwrap.dedent(_SCRIPT))
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("TENSORQUILL_ENDPOINT", None)
    done = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2", "--master-addr", "127.0.0.1",
         "--master-port", str(_free_port()), str(script), how, who],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    said = done.stdout + done.stderr
    run = f"run-{how}"
    steps = {rank: _steps(tmp_path, run, rank) for rank in (0, 1)}
    markers = {rank: _marker(tmp_path, run, rank) for rank in (0, 1)}
    ends = {rank: m and (m["outcome"], m.get("exception")) for rank, m in markers.items()}
    seen = f"steps {steps}, markers {ends}, exit {done.returncode}\n{said}"
    assert {m["run_id"] for m in markers.values() if m} == {run}, seen

    if how == "close":
        assert done.returncode == 0, seen
        assert steps == {0: [0, 1, 2], 1: [0, 1, 2]}, seen
        assert ends == {0: ("closed", None), 1: ("closed", None)}, seen
        assert "rank 0 trained to step 20" in done.stdout and "rank 1 trained to step 20" in done.stdout, seen
        return
    assert done.returncode != 0, seen
    kind = "RuntimeError" if how == "error" else "KeyboardInterrupt"
    if who == "all":
        assert steps == {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3, 4]}, seen
        assert ends == {0: ("raised", kind), 1: ("raised", kind)}, seen
        # Each rank ends as one process would, or, once the other has ended, is stopped by torchrun
        # (SIGTERM) after its exit hook: 4 of 24 runs here, each with its records complete.
        expected, codes = 1 if how == "error" else -signal.SIGINT, _exit_codes(done.stderr)
        assert set(codes.values()) <= {expected, -signal.SIGTERM}, seen
        assert not codes or expected in codes.values(), seen
    else:
        assert steps[1] == [0, 1, 2, 3, 4] and ends[1] == ("raised", kind), seen
        assert steps[0] == steps[1][: len(steps[0])], seen
        assert ends[0] is None or ends[0][0] == "raised", seen
        assert _exit_codes(done.stderr).get(1, 1) == 1, seen
