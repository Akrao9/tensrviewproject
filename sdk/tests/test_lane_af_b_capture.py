"""Lane AF-B (wave AF): the capture's integrity.

- Capture on and capture off give the same training: final parameters, each step's update, the
  modules' training flags, BatchNorm statistics, the batches, the random state and completion.
- SDK-07: `close()` in a forked child leaves the parent's session and its file as they were.
- SDK-01: in the router-only mode a parameter frozen at `init` and unfrozen later is not
  recorded, also by a hand `record_grads()`: a stated limit, pinned here.
- The per-router device tensor of expert ids that nothing read is gone, and router counts are
  unchanged eager and compiled.
- SIGTERM: what an interrupted session leaves, pinned as a stated limit.
"""

from __future__ import annotations

import json
import os
import signal
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
from tensorquill.sink import JsonlSink, end_marker_path, read_end_marker

SRC = str(Path(tensorquill.__file__).resolve().parents[1])


class _Moe(nn.Module):
    """A small model with a router, a BatchNorm and a Dropout: modes, statistics and the RNG."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(6, 8)
        self.norm = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(0.3)
        self.router = nn.Linear(8, 4)
        self.experts = nn.ModuleList(nn.Linear(8, 8) for _ in range(4))
        self.out = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch.relu(self.norm(self.inp(x))))
        choice = self.router(h).argmax(-1)
        mixed = torch.stack([e(h) for e in self.experts], 1)
        h = mixed[torch.arange(h.shape[0]), choice]
        return self.out(h)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _train(tmp_path: Path, mode: str | None, steps: int = 13) -> dict:
    torch.manual_seed(0)
    model = _Moe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = torch.utils.data.TensorDataset(torch.randn(64, 6), torch.randn(64, 1))
    loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True,
                                         generator=torch.Generator().manual_seed(1))
    scaler = None
    session = None
    if mode is not None:
        kwargs: dict = {"run_id": f"parity-{mode}", "every": 4,
                        "sink": JsonlSink(tmp_path / f"{mode}.jsonl"), "heartbeat_seconds": None}
        if mode == "routers":
            kwargs.update(routers={"router": tensorquill.Router(4, top_k=1)})
        elif mode == "full":
            kwargs.update(capture="full", routers={"router": tensorquill.Router(4, top_k=1)})
        elif mode == "full_updates_probe":
            kwargs.update(capture="full", updates=True, probe=torch.randn(8, 6, generator=torch.Generator().manual_seed(2)), probe_every=3)
        elif mode == "data_wait":
            kwargs.update(capture="full", data_wait=True)
        elif mode == "scaler":
            scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 4)
            kwargs.update(capture="full", scaler=scaler)
        session = tensorquill.init(model, **kwargs)
    seen, updates, modes = [], [], []
    done = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        it = iter(loader)
        while done < steps:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            seen.append(x.clone())
            before = [p.detach().clone() for p in model.parameters()]
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            updates.append([p.detach() - b for p, b in zip(model.parameters(), before)])
            modes.append([m.training for m in model.modules()])
            done += 1
        if session is not None:
            session.close()
    return {
        "params": [p.detach().clone() for p in model.parameters()],
        "buffers": [b.clone() for b in model.buffers()],
        "updates": updates, "modes": modes, "batches": seen, "done": done,
        "rng": torch.get_rng_state(), "opt": opt.state_dict(), "session": session,
    }


@pytest.mark.parametrize("mode", ["routers", "full", "full_updates_probe", "data_wait", "scaler"])
def test_capture_on_and_off_train_the_same(tmp_path, mode):
    # The scaler is part of the loop, not of capture: that mode is compared with the scaler's loop, capture off.
    off = _train_scaler_off(tmp_path) if mode == "scaler" else _train(tmp_path, None)
    on = _train(tmp_path, mode)
    assert on["done"] == off["done"] == 13
    assert all(torch.equal(a, b) for a, b in zip(on["params"], off["params"]))
    assert all(torch.equal(a, b) for a, b in zip(on["buffers"], off["buffers"]))
    assert all(torch.equal(a, b) for sa, sb in zip(on["updates"], off["updates"]) for a, b in zip(sa, sb))
    assert on["modes"] == off["modes"]
    assert all(torch.equal(a, b) for a, b in zip(on["batches"], off["batches"]))
    assert torch.equal(on["rng"], off["rng"])
    for key, state in on["opt"]["state"].items():
        other = off["opt"]["state"][key]
        assert all(torch.equal(state[k], other[k]) for k in state)
    session = on["session"]
    assert session._closed and session.writer_error is None
    records = _records(tmp_path / f"{mode}.jsonl")
    assert [r["step"] for r in records] == list(range(13))
    assert read_end_marker(tmp_path / f"{mode}.jsonl")["state"] == "closed"


def _train_scaler_off(tmp_path: Path) -> dict:
    """The scaler loop with capture off: the same seed, data and scaler."""
    torch.manual_seed(0)
    model = _Moe()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = torch.utils.data.TensorDataset(torch.randn(64, 6), torch.randn(64, 1))
    loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True,
                                         generator=torch.Generator().manual_seed(1))
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 4)
    seen, updates, modes = [], [], []
    it = iter(loader)
    for _ in range(13):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        seen.append(x.clone())
        before = [p.detach().clone() for p in model.parameters()]
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        updates.append([p.detach() - b for p, b in zip(model.parameters(), before)])
        modes.append([m.training for m in model.modules()])
    return {
        "params": [p.detach().clone() for p in model.parameters()],
        "buffers": [b.clone() for b in model.buffers()],
        "updates": updates, "modes": modes, "batches": seen, "done": 13,
        "rng": torch.get_rng_state(), "opt": opt.state_dict(),
    }


# ---- SDK-07: close() in a forked child ------------------------------------------------------


def _wait_idle(session, path: Path) -> bytes:
    """The writer has taken and written everything queued: the queue is empty and the file stops growing."""
    deadline = time.monotonic() + 10
    last = None
    while time.monotonic() < deadline:
        now = path.read_bytes()
        if session._queue.empty() and now == last:
            return now
        last = now
        time.sleep(0.1)
    raise AssertionError("the writer did not go idle")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_sdk07_close_in_a_forked_child_leaves_the_parents_session_and_file(tmp_path):
    path = tmp_path / "fork.jsonl"
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tensorquill.init(model, run_id="fork", every=2, sink=JsonlSink(path), heartbeat_seconds=None,
                               capture="full")

    def step() -> None:
        opt.zero_grad()
        model(torch.randn(3, 4)).sum().backward()
        opt.step()

    for _ in range(5):
        step()
    before = _wait_idle(session, path)
    assert before and not end_marker_path(path).exists()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork with threads running
        pid = os.fork()
    if pid == 0:  # the child: close the inherited session and leave without running anything else
        code = 1
        try:
            session.close()
            code = 0 if session._closed else 2
        finally:
            os._exit(code)
    deadline = time.monotonic() + 30
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise AssertionError("the child did not end")
        time.sleep(0.05)
    assert os.waitstatus_to_exitcode(status) == 0
    assert path.read_bytes() == before, "the child's close wrote to the parent's file"
    assert not end_marker_path(path).exists(), "the child's close wrote the parent's end marker"
    assert not session._closed and not session._stopped and session.writer_error is None
    for _ in range(3):
        step()
    session.close()
    assert [r["step"] for r in _records(path)] == list(range(8))
    marker = read_end_marker(path)
    assert marker["state"] == "closed"
    assert json.loads(end_marker_path(path).read_text())["records_bytes"] == path.stat().st_size


# ---- SDK-01: the router-only mode and a parameter unfrozen later ------------------------------


def test_sdk01_router_only_records_no_parameter_unfrozen_after_init(tmp_path):
    """A stated limit, not a fault: the router-only mode never adds a parameter frozen at `init`,
    also when `record_grads()` is called by hand; the parameters trainable at `init` are recorded."""
    path = tmp_path / "frozen.jsonl"
    torch.manual_seed(0)
    model = _Moe()
    model.out.weight.requires_grad_(False)
    model.out.bias.requires_grad_(False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tensorquill.init(model, run_id="frozen", every=2, sink=JsonlSink(path), heartbeat_seconds=None,
                               routers={"router": tensorquill.Router(4, top_k=1)}, auto_step=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(4):
            if i == 2:
                model.out.weight.requires_grad_(True)
                model.out.bias.requires_grad_(True)
            opt.zero_grad()
            model(torch.randn(8, 6)).sum().backward()
            session.record_grads()
            opt.step()
            session.step()
        session.close()
    records = _records(path)
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    for record in records:
        names = set(record.get("grad_norms", {}))
        assert "inp.weight" in names
        assert not names & {"out.weight", "out.bias"}
    assert not any("unfrozen" in str(w.message) or "frozen when" in str(w.message) for w in caught)


# ---- the per-router tensor of expert ids -----------------------------------------------------


def test_no_per_router_expert_id_tensor_is_kept(tmp_path):
    session = tensorquill.init(_Moe(), run_id="ids", sink=JsonlSink(tmp_path / "ids.jsonl"),
                               heartbeat_seconds=None, routers={"router": tensorquill.Router(4, top_k=1)})
    try:
        assert not hasattr(session, "_expert_ids")
    finally:
        session.close()


@pytest.mark.parametrize("backend", [None, "eager", "aot_eager"])
def test_router_counts_eager_and_compiled_after_the_removal(tmp_path, backend):
    """Counts equal the router's own choices, eager and compiled, with no recompile across steps."""
    import torch._dynamo

    torch._dynamo.reset()
    torch.manual_seed(0)
    model = _Moe()
    path = tmp_path / f"c-{backend}.jsonl"
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    session = tensorquill.init(model, run_id="c", every=3, sink=JsonlSink(path), heartbeat_seconds=None,
                               routers={"router": tensorquill.Router(4, top_k=1)})
    run = model if backend is None else torch.compile(model, backend=backend)
    from torch._dynamo.utils import counters

    counters.clear()
    for _ in range(6):
        opt.zero_grad()
        run(torch.randn(8, 6)).sum().backward()
        opt.step()
    session.close()
    records = _records(path)
    assert [r["step"] for r in records] == list(range(6))
    for record in records:
        assert sum(record["moe"]["router"]) == 8
    if backend is not None:
        assert counters["stats"]["unique_graphs"] <= 2


# ---- SIGTERM: a stated limit --------------------------------------------------------------------

_SIGTERM_JOB = textwrap.dedent('''
    import signal, sys, time
    from pathlib import Path
    import torch
    from torch import nn
    import tensorquill
    from tensorquill.sink import JsonlSink
    path, handler, every = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    if handler == "raise":
        def stop(signum, frame):
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, stop)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with tensorquill.init(model, run_id="term", every=every, sink=JsonlSink(path), heartbeat_seconds=None,
                          capture="full") as session:
        for i in range(10_000):
            opt.zero_grad()
            model(torch.randn(3, 4)).sum().backward()
            opt.step()
            if i == 6:
                print("ready", flush=True)
            time.sleep(0.01)
''')


def _sigterm_run(tmp_path: Path, handler: str, every: int):
    path = tmp_path / f"term-{handler}-{every}.jsonl"
    env = {**os.environ, "PYTHONPATH": SRC, "PYTHONWARNINGS": "ignore"}
    job = subprocess.Popen([sys.executable, "-c", _SIGTERM_JOB, str(path), handler, str(every)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    try:
        assert job.stdout.readline().strip() == "ready"
        time.sleep(0.2)
        job.send_signal(signal.SIGTERM)
        job.wait(timeout=60)
    finally:
        if job.poll() is None:
            job.kill()
            job.wait()
    return job.returncode, path


def test_sigterm_with_no_handler_loses_the_steps_not_yet_written_and_leaves_no_marker(tmp_path):
    """The stated limit: TensorQuill installs no SIGTERM handler and no exit hook, so the job's own
    handling decides. Under the default disposition the process ends at once: no `close()` runs,
    the steps still on the device or queued are not written, and the file reads not closed. What
    was written ends in whole lines."""
    code, path = _sigterm_run(tmp_path, "default", every=4)
    assert code == -signal.SIGTERM
    text = path.read_text()
    assert text == "" or text.endswith("\n")
    steps = [r["step"] for r in _records(path)]
    assert steps == list(range(len(steps)))
    assert len(steps) >= 4  # at least one block of `every` was handed to the writer before the signal
    assert read_end_marker(path)["state"] == "not_closed"


def test_sigterm_with_a_job_handler_that_raises_closes_the_session(tmp_path):
    """A job that turns SIGTERM into an exception (as `sys.exit` in a handler does) leaves `with
    session:` by it: every step is written and the marker says raised."""
    code, path = _sigterm_run(tmp_path, "raise", every=50)
    assert code == 143
    steps = [r["step"] for r in _records(path)]
    assert steps == list(range(len(steps))) and len(steps) >= 7
    marker = read_end_marker(path)
    assert marker["state"] == "closed"
    assert json.loads(end_marker_path(path).read_text())["outcome"] == "raised"


def test_tensorquill_installs_no_signal_handler_and_no_exit_hook(tmp_path):
    import atexit

    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    registered = []
    original = atexit.register
    atexit.register = lambda f, *a, **k: registered.append(f) or original(f, *a, **k)
    try:
        session = tensorquill.init(nn.Linear(4, 1), run_id="sig", sink=JsonlSink(tmp_path / "sig.jsonl"),
                                   heartbeat_seconds=None, capture="full")
        session.close()
    finally:
        atexit.register = original
    assert {s: signal.getsignal(s) for s in before} == before
    assert registered == []
