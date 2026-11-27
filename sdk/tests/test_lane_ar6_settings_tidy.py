"""Lane AR-6: the other ranks' numerical settings, and a tidy-up of the run manifest.

- `manifest._RANK_FILE` reads a second writer's records file, `<run_id>.rank<N>.<pid>.jsonl`
  (`sink.JsonlSink`), as rank N's file of the run, so `run_id_of_file` and `manifest_candidates`
  find the run's manifest from it when no run ID is passed. The rank and the pid are ASCII
  digits, as `sink.records_file_parts` reads them, and the two readers agree on every name.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest
from torch import nn

import tensorquill
from tensorquill import manifest as mf
from tensorquill import sink as _sink


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


def _init(run_id: str, model=None, **options):
    model = nn.Linear(2, 1) if model is None else model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id=run_id, every=1, heartbeat_seconds=None, stack_seconds=None,
                                   **options)
    return session, [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()], model


# --- Item 2: a second writer's file names its run ------------------------------------------------


NAMES = [
    # (file name, the run ID it carries, or None)
    ("x.rank0.jsonl", "x"),
    ("x.rank3.4242.jsonl", "x"),  # a second writer of rank 3
    ("a.rank3.rank0.jsonl", "a.rank3"),  # a run ID that itself ends in `.rank3`
    ("a.rank3.rank0.77.jsonl", "a.rank3"),
    ("a.rank3.12.rank1.jsonl", "a.rank3.12"),  # one that ends in `.rank3` and digits
    ("a.rank3.12.rank1.9.jsonl", "a.rank3.12"),
    ("run-slurm4242.0.rank1.jsonl", "run-slurm4242.0"),  # a run ID that ends in digits
    ("run-slurm4242.0.rank1.31.jsonl", "run-slurm4242.0"),
    ("7.rank2.jsonl", "7"),
    ("x.rank0.bak.jsonl", None),  # a backup: `bak` is not a pid
    ("x.rank0.jsonl.bak", None),
    ("x.rank٣.jsonl", None),  # a digit of another script is not a rank (DET-17)
    ("x.rank0.٣.jsonl", None),
    ("x.rank.jsonl", None),
    ("x.rank0.jsonl.settings.json", None),
    ("x.rank1.settings.json", None),  # the other ranks' settings file is no records file
    ("x.run.json", None),
]


@pytest.mark.parametrize("name, run_id", NAMES)
def test_run_id_of_file_reads_a_second_writers_file_and_agrees_with_the_sink(name, run_id):
    assert mf.run_id_of_file(Path("out") / name) == run_id
    parts = _sink.records_file_parts(name)
    assert (parts[0] if parts else None) == run_id


@pytest.mark.parametrize("name, run_id", [(n, r) for n, r in NAMES if r is not None])
def test_manifest_candidates_of_a_records_file_is_its_runs_manifest(tmp_path, name, run_id):
    assert mf.manifest_candidates(tmp_path / name) == [tmp_path / f"{run_id}.run.json"]


def test_a_second_writers_file_finds_the_runs_manifest_with_no_run_id_given(here):
    one, _, _ = _init("pair", project="granite", name="baseline")
    two, said, _ = _init("pair", project="granite", name="baseline")
    try:
        second = two._sink.path
        assert second.name == f"pair.rank0.{os.getpid()}.jsonl" and len(said) == 1
    finally:
        two.close()
        one.close()
    found = mf.find_manifest(second)
    assert found is not None and (found["run_id"], found["project"], found["name"]) == ("pair", "granite", "baseline")
