"""Lane AR-7, item F (the second half of item 9 of the author's list): every warning `init` gives points
at the line that called `tensorquill.init`, not at a frame inside PyTorch.

`init` runs under `torch.compiler.disable`, whose wrapper adds frames between the caller and `init`
(one on PyTorch 2.15, maybe others on another version). The `stacklevel`s in `sink.py`, `tracker.py`,
`manifest.py` and `flight.attach` counted frames without them, so the warnings named
`torch/_dynamo/eval_frame.py`, and a `warnings.filterwarnings(module=...)` aimed at the user's code
missed them.

Each case below makes `init` give one warning (or two) and holds every warning said during the call
to this file and the line of the `init` call in `_init_here`. Run on `.venv` (PyTorch 2.15) and on
`.venv-torch26` (PyTorch 2.6). No server binary.
"""

from __future__ import annotations

import errno
import inspect
import os
import sys
import warnings
from pathlib import Path

import pytest
from torch import nn

import tensorquill
from tensorquill import manifest as mf
from tensorquill import sink as _sink
from tensorquill import tracker

NOWHERE = "grpc://127.0.0.1:9"  # nothing listens here


@pytest.fixture
def here(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MLFLOW_EXPERIMENT_NAME", mf.PROJECT_ENV, mf.NAME_ENV) \
                or name.startswith(("SLURM_", "OMPI_", "PMI_", "TORCHELASTIC_", "TENSORQUILL_", "MLFLOW_", "WANDB_")):
            if name != "TENSORQUILL_SERVER_BIN":
                monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def _init_here(**options):
    """`tensorquill.init`, called on the line `INIT_LINE` names; every warning said during it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(nn.Linear(2, 1), every=1, heartbeat_seconds=None, stack_seconds=None, **options)
    return session, list(caught)


INIT_LINE = next(number for number, line in enumerate(inspect.getsource(_init_here).splitlines(),
                                                     inspect.getsourcelines(_init_here)[1])
                 if "tensorquill.init(" in line)


def _held_to_the_call(caught, *expected: str) -> None:
    said = [w for w in caught if "tensorquill" in str(w.message).lower()]
    for text in expected:
        assert any(text in str(w.message) for w in said), (text, [str(w.message) for w in said])
    where = [(Path(w.filename).name, w.lineno, str(w.message)[:90]) for w in said]
    assert all(Path(w.filename) == Path(__file__) and w.lineno == INIT_LINE for w in said), (INIT_LINE, where)


def _close(*sessions) -> None:
    for session in sessions:
        if session is not None:
            session.close()


def test_the_sinks_lock_unavailable(here, monkeypatch):
    monkeypatch.setattr(_sink, "_try_lock", lambda file: ("unavailable", OSError(errno.ENOLCK, "no locks")))
    session, caught = _init_here(run_id="nolock")
    _close(session)
    _held_to_the_call(caught, "could not lock")


def test_the_sinks_second_file_cannot_be_opened(here, monkeypatch):
    one, _ = _init_here(run_id="noopen")
    real = _sink.JsonlSink._open

    def refuse(self, path):
        if Path(path).name != "noopen.rank0.jsonl":
            raise OSError(errno.EACCES, "denied")
        return real(self, path)

    monkeypatch.setattr(_sink.JsonlSink, "_open", refuse)
    two, caught = _init_here(run_id="noopen")
    _close(two, one)
    _held_to_the_call(caught, "or open a file of its own")


def test_a_second_writer_with_an_endpoint_and_its_two_statements(here):
    pytest.importorskip("pyarrow.flight")
    one, _ = _init_here(run_id="two")
    two, caught = _init_here(run_id="two", endpoint=NOWHERE)
    _close(two, one)
    _held_to_the_call(caught, "held by another writer", "nothing of")


def test_a_third_writer_in_this_process(here):
    one, _ = _init_here(run_id="three")
    two, _ = _init_here(run_id="three")
    three, caught = _init_here(run_id="three")
    _close(three, two, one)
    _held_to_the_call(caught, "both held by other writers")


def test_the_placement_warning(here, monkeypatch):
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "2")
    session, caught = _init_here(run_id="placed")
    _close(session)
    _held_to_the_call(caught, "the rank is not below")


def test_no_run_id_the_ranks_share(here, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    session, caught = _init_here()
    _close(session)
    _held_to_the_call(caught, "found no run ID the ranks share")


def test_a_tracker_that_raises(here, monkeypatch):
    def broken():
        raise RuntimeError("tracker down")

    monkeypatch.setattr(tracker, "tracker_run_id", broken)
    session, caught = _init_here()
    _close(session)
    _held_to_the_call(caught, "could not read the tracker's run")


def test_a_run_id_made_to_fit(here, monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "a run/with a slash")
    session, caught = _init_here()
    _close(session)
    _held_to_the_call(caught, "cannot name TensorQuill's files")


def test_a_project_made_to_fit(here, monkeypatch):
    monkeypatch.setenv(mf.PROJECT_ENV, "a/b c")
    session, caught = _init_here(run_id="proj")
    _close(session)
    _held_to_the_call(caught, "cannot be a TensorQuill project name")


def test_a_manifest_that_cannot_be_written(here):
    (here / "tensorquill-out" / "nomanifest.run.json").mkdir(parents=True)
    session, caught = _init_here(run_id="nomanifest")
    _close(session)
    _held_to_the_call(caught, "could not write the run manifest")


def test_a_settings_file_that_cannot_be_written(here, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    (here / "tensorquill-out" / "nosettings.rank1.settings.json").mkdir(parents=True)
    session, caught = _init_here(run_id="nosettings")
    _close(session)
    _held_to_the_call(caught, "numerical settings")


def test_a_session_over_the_servers_limit(here, monkeypatch):
    pytest.importorskip("pyarrow.flight")
    monkeypatch.setenv("RANK", "100000")
    monkeypatch.setenv("WORLD_SIZE", "100000")
    session, caught = _init_here(run_id="over", endpoint=NOWHERE)
    _close(session)
    _held_to_the_call(caught, "over the server's limit")


def test_the_warning_filter_by_module_takes_them(here, monkeypatch):
    """What the location is for: a filter on the caller's module silences `init`'s statements."""
    monkeypatch.setattr(_sink, "_try_lock", lambda file: ("unavailable", OSError(errno.ENOLCK, "no locks")))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)
        session = tensorquill.init(nn.Linear(2, 1), run_id="filtered", every=1, heartbeat_seconds=None,
                                   stack_seconds=None)
    session.close()
    assert not [w for w in caught if "could not lock" in str(w.message)], [str(w.message) for w in caught]


def test_after_init_a_warning_points_where_it_did(here):
    """The location is `init`'s only for the length of the call: `close()`'s left-out scalar still
    points at the `close()` line (lane AR-5's guard)."""
    session, _ = _init_here(run_id="after")
    session.log({"loss": 1.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        session.close()
    (left,) = [w for w in caught if "left out the scalar" in str(w.message)]
    assert Path(left.filename) == Path(__file__) and left.lineno == line
