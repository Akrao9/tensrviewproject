"""Lane AS-7: a caller's sink whose `path` is not path-like.

A sink passed to `tensorquill.init` may carry any `path` attribute. Before this lane one that
`pathlib.Path` cannot take (None excepted: an int, bytes, any other object, a `__fspath__` that
raises) or that names no file (`""`, `"."`) failed `init` on every rank: at rank 0 in
`manifest.local_manifest_path`, at the other ranks in `sink.clear_end_marker`. The capture path's
rule is that capture never raises into training, also under `python -W error`.

Such a sink is now taken as the code already takes a sink with no `path` (a `MemorySink`): no
end marker, no rank settings file and no run manifest is written for it, and its records still
reach its `write`. Unlike a sink with no `path`, whose having no file is what it is, a `path`
that is set and cannot be read is said once at `init`, by its type and why, on every rank. A
sink with no `path` or a `path` of None stays silent, and one with a real path is unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import manifest as mf
from tensorquill import sink as _sink
from tensorquill.sink import JsonlSink, MemorySink


class _PathSink(MemorySink):
    """A caller's sink that keeps its records in memory and carries a `path` of any kind."""

    def __init__(self, path):
        super().__init__()
        self.path = path


class _Opaque:
    """An object that is not path-like."""


class _BadFspath:
    """An `os.PathLike` whose `__fspath__` raises."""

    def __fspath__(self):
        raise RuntimeError("no path here")


NOT_PATHS = [
    pytest.param(42, "int", id="int"),
    pytest.param(_Opaque(), "_Opaque", id="object"),
    pytest.param(b"r.rank0.jsonl", "bytes", id="bytes"),
    pytest.param(_BadFspath(), "_BadFspath", id="fspath-raises"),
    pytest.param("", "str", id="empty-str"),
    pytest.param(".", "str", id="dot"),
]


@pytest.fixture
def here(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MLFLOW_EXPERIMENT_NAME", mf.PROJECT_ENV, mf.NAME_ENV) \
                or name.startswith(("SLURM_", "OMPI_", "PMI_", "TORCHELASTIC_", "TENSORQUILL_", "MLFLOW_", "WANDB_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def _as_rank(monkeypatch, rank: int) -> None:
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")


def _train(sink, steps: int = 2):
    """`init` with `sink`, `steps` optimizer steps and `close`; what TensorQuill said on the way."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, run_id="as7", every=1, heartbeat_seconds=None,
                                   stack_seconds=None)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(steps):
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        session.close()
    return session, [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()]


def _files(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


# --- The fault: init failed on such a sink --------------------------------------------------------


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("path, type_name", NOT_PATHS)
def test_init_and_close_do_not_raise_on_a_sink_whose_path_is_not_path_like(here, monkeypatch, rank, path,
                                                                           type_name):
    _as_rank(monkeypatch, rank)
    sink = _PathSink(path)
    session, said = _train(sink)
    assert [r["step"] for r in sink.records if "step" in r and r.get("record") != "scalars"] == [0, 1]
    assert session.writer_error is None
    # Nothing is written for a sink with no records file: no end marker, settings file or manifest.
    assert _files(here) == []


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("path, type_name", NOT_PATHS)
def test_such_a_sink_is_said_once_at_init_by_its_type_and_nothing_at_close(here, monkeypatch, rank, path, type_name):
    _as_rank(monkeypatch, rank)
    _, said = _train(_PathSink(path))
    assert len(said) == 1, said
    assert "_PathSink" in said[0] and f"({type_name}" in said[0], said[0]
    assert "no records file" in said[0] and "end marker" in said[0], said[0]


@pytest.mark.parametrize("path", [p.values[0] for p in NOT_PATHS], ids=[p.id for p in NOT_PATHS])
def test_clear_end_marker_never_raises(path):
    assert _sink.clear_end_marker(path) is None


# --- The controls ---------------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("kind", ["no-path", "none"])
def test_a_sink_with_no_path_or_a_path_of_none_writes_nothing_and_says_nothing(here, monkeypatch, rank, kind):
    _as_rank(monkeypatch, rank)
    sink = MemorySink() if kind == "no-path" else _PathSink(None)
    session, said = _train(sink)
    assert said == []
    assert len([r for r in sink.records if "step" in r]) == 2
    assert _files(here) == []


@pytest.mark.parametrize("rank", [0, 1])
def test_a_jsonl_sink_with_a_real_path_keeps_its_marker_settings_and_manifest(here, monkeypatch, rank):
    _as_rank(monkeypatch, rank)
    records = here / "out" / f"as7.rank{rank}.jsonl"
    session, said = _train(JsonlSink(records))
    assert said == []
    names = _files(here / "out")
    assert f"as7.rank{rank}.jsonl" in names and f"as7.rank{rank}.end.json" in names
    assert ("as7.run.json" if rank == 0 else f"as7.rank{rank}.settings.json") in names
    assert _sink.read_end_marker(records)["state"] == "closed"


# --- Under python -W error --------------------------------------------------------------------------

_W_ERROR_SCRIPT = r"""
import os, sys, tempfile
import torch
from torch import nn
import tensorquill
from tensorquill.sink import MemorySink

class PathSink(MemorySink):
    def __init__(self, path):
        super().__init__()
        self.path = path

os.chdir(tempfile.mkdtemp())
os.environ["WORLD_SIZE"] = "2"
for rank in ("0", "1"):
    os.environ["RANK"] = rank
    for path in (42, object(), b"x.jsonl", ""):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        sink = PathSink(path)
        session = tensorquill.init(model, sink=sink, run_id="as7w", every=1, heartbeat_seconds=None,
                                   stack_seconds=None)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        session.close()
        assert len([r for r in sink.records if "step" in r]) == 1, sink.records
print("trained", sorted(os.listdir(".")))
"""


def test_under_python_w_error_init_and_close_do_not_raise():
    src = os.path.join(os.path.dirname(tensorquill.__file__), os.pardir)
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("TENSORQUILL_", "RANK", "LOCAL_RANK", "WORLD_SIZE", "WANDB_", "MLFLOW_"))}
    env["PYTHONPATH"] = os.path.abspath(src)
    # Without NumPy, importing PyTorch itself warns ("Failed to initialize NumPy"), which `-W error` would turn
    # into a failed import before TensorQuill runs. Only that message is let through; every other warning is an error.
    done = subprocess.run([sys.executable, "-W", "error", "-W", "ignore:Failed to initialize NumPy", "-c",
                           _W_ERROR_SCRIPT], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    assert "trained []" in done.stdout, done.stdout
