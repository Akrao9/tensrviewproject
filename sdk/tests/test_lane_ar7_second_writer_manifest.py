"""Lane AR-7, item C (item 8 of the author's list): a second writer at rank 0 does not write the run's
manifest.

A second writer of a rank (another session of the same run ID and rank, which finds the rank's file
locked and writes `<run_id>.rank<N>.<pid>.jsonl`; its sink's `other_writer` is set) shares the run's
manifest path, `<run_id>.run.json`. Before, `init` called `write_manifest` on it for every rank-0
session, so a second writer of rank 0 was taken for a later start of the run: it merged its settings
in, counted a start, and replaced `pinned_flush`, `config` and `tags` with its own. Its `init`'s undo
also put the manifest back by rewriting it.

The author's answer: "A second writer at rank 0 merges into the run's manifest: it does not write
the manifest." The second writer's own numerical settings go to `<run_id>.rank0.<pid>.settings.json`,
as a second writer of a rank above 0 writes its own (lane AR-6).

No server binary.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture
from tensorquill import manifest as mf


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


def _init(run_id: str, **options):
    model = nn.Linear(2, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id=run_id, every=1, heartbeat_seconds=None, stack_seconds=None,
                                   **options)
    return session, [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()], model


def _train(model, steps: int) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        model(torch.randn(3, 2)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()


def _stat(path: Path) -> tuple[int, int]:
    found = path.stat()
    return found.st_ino, found.st_mtime_ns


def test_a_second_writer_at_rank_0_leaves_the_first_writers_manifest_byte_for_byte(here, monkeypatch):
    one, _, model_one = _init("same", config={"lr": 0.1}, tags=["first"], pinned_flush=False)
    manifest = here / "tensorquill-out" / "same.run.json"
    before, stat = manifest.read_bytes(), _stat(manifest)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # a numerical setting the first start did not have
    two, said, model_two = _init("same", config={"lr": 0.2}, tags=["second"], pinned_flush=True)
    try:
        assert two._sink.other_writer is not None
        assert two._sink.path.name == f"same.rank0.{os.getpid()}.jsonl", said
        _train(model_two, 3)
        _train(model_one, 2)
    finally:
        two.close()
    assert manifest.read_bytes() == before
    assert _stat(manifest) == stat  # not written at all, not even with the same bytes
    one.close()
    assert manifest.read_bytes() == before


def test_a_second_writer_at_rank_0_keeps_its_own_settings_beside_its_file(here, monkeypatch):
    one, _, _ = _init("own")
    out = here / "tensorquill-out"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    two, _, model = _init("own")
    try:
        _train(model, 1)
    finally:
        two.close()
        one.close()
    own = out / f"own.rank0.{os.getpid()}.settings.json"
    document = json.loads(own.read_text())
    assert (document["run_id"], document["rank"]) == ("own", 0)
    assert document["numerical_settings"]["settings"]["os.environ.get('CUBLAS_WORKSPACE_CONFIG')"] == {
        "value": ":16:8"}
    assert mf.rank_settings_path(two._sink.path, "own", 0).resolve() == own.resolve()
    # The first writer of rank 0 keeps its settings in the manifest, and writes no settings file.
    assert not (out / "own.rank0.settings.json").exists()
    assert mf.rank_settings(out, "own") == {}
    assert two.manifest is None  # it wrote no run manifest, as a rank above 0 writes none
    assert one.manifest is not None


def test_a_failed_init_of_a_second_writer_neither_removes_nor_restores_the_first_writers_manifest(here, monkeypatch):
    one, _, _ = _init("undo")
    out = here / "tensorquill-out"
    manifest = out / "undo.run.json"
    before, stat = manifest.read_bytes(), _stat(manifest)

    def refuse(*args, **kwargs):
        raise RuntimeError("no session today")

    monkeypatch.setattr(capture, "Session", refuse)
    try:
        with pytest.raises(RuntimeError, match="no session today"):
            _init("undo", config={"other": True})
    finally:
        monkeypatch.undo()
        one.close()
    assert manifest.read_bytes() == before
    assert _stat(manifest) == stat
    # Its own settings file is taken back, as any file a failed `init` made.
    assert not (out / f"undo.rank0.{os.getpid()}.settings.json").exists()


def test_the_folder_readers_do_not_take_a_second_writers_rank_0_settings_file(here):
    from tensorquill import runs as _runs

    one, _, model_one = _init("pair")
    two, _, model_two = _init("pair")
    try:
        _train(model_one, 2)
        _train(model_two, 2)
    finally:
        two.close()
        one.close()
    out = here / "tensorquill-out"
    own = out / f"pair.rank0.{os.getpid()}.settings.json"
    assert own.is_file()
    (with_file,) = tensorquill.list_runs(out)
    hidden = here / "aside.settings.json"
    own.rename(hidden)
    (without,) = tensorquill.list_runs(out)
    hidden.rename(own)
    assert with_file == without
    (found,) = _runs.discover_runs(out)
    assert found["run_id"] == "pair"
    assert mf.read_manifest(out / "pair.run.json")["run_id"] == "pair"


def test_the_first_writer_and_a_later_start_still_write_the_manifest(here):
    """The guard: one writer at a time is unchanged. A run started again under its run ID, after the
    first closed, is no second writer: it merges into the manifest as before."""
    one, _, model = _init("again", config={"lr": 0.1})
    _train(model, 1)
    one.close()
    manifest = here / "tensorquill-out" / "again.run.json"
    first = json.loads(manifest.read_text())
    two, _, model = _init("again", config={"lr": 0.2})
    try:
        assert two._sink.other_writer is None
        _train(model, 1)
    finally:
        two.close()
    later = json.loads(manifest.read_text())
    assert later["config"] == {"lr": 0.2} and later["created_at"] == first["created_at"]
    assert two.manifest is not None
