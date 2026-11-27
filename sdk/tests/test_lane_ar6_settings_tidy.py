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


# --- Item 3: the `pinned_flush` switch in the run manifest ---------------------------------------


def _train(model, steps: int) -> None:
    import torch

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        model(torch.randn(3, 2)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()


def _run(run_id: str, steps: int = 2, **options) -> list[str]:
    session, said, model = _init(run_id, **options)
    _train(model, steps)
    session.close()
    return said


@pytest.mark.parametrize("asked", [True, False])
def test_the_pinned_flush_switch_is_in_the_manifest_as_init_was_given_it(here, asked):
    # On a CPU the pinned path falls back to the blocking copy, said once: the manifest records
    # the switch as given, not the path the flush took.
    session, _, model = _init("p", pinned_flush=asked)
    _train(model, 1)
    session.close()
    written = mf.read_manifest(here / "tensorquill-out" / "p.run.json")
    assert written["pinned_flush"] is asked and session.manifest["pinned_flush"] is asked


def test_the_default_records_pinned_flush_off(here):
    _run("d")
    assert mf.read_manifest(here / "tensorquill-out" / "d.run.json")["pinned_flush"] is False


def test_a_manifest_built_without_the_switch_has_no_key():
    # Only `init` knows the switch: a manifest built by another caller does not claim it off.
    built = mf.build_manifest(run_id="r", run_id_source="explicit", project="p", name="n")
    assert "pinned_flush" not in built
    assert mf.build_manifest(run_id="r", run_id_source="explicit", project="p", name="n",
                             pinned_flush=True)["pinned_flush"] is True


def test_the_switch_rides_on_the_stream_copy_of_the_manifest():
    built = mf.build_manifest(run_id="r", run_id_source="explicit", project="p", name="n", pinned_flush=True,
                              config={"vocab": "x" * 70_000})
    sent, why = mf.manifest_to_send(built)
    assert sent["pinned_flush"] is True and sent["config_left_out"] and why


def test_compare_shows_runs_that_differ_in_the_switch(here):
    _run("off", pinned_flush=False)
    _run("on", pinned_flush=True)
    result = tensorquill.compare_runs("off", "on", source=here / "tensorquill-out")
    setup = result["setup"]
    assert {"field": "pinned_flush", "values": {"A": False, "B": True}} in setup["differs"]
    assert "pinned_flush" not in setup["same"] and "pinned_flush_not_recorded" not in setup
    assert "   pinned_flush: A false, B true" in result["summary"].splitlines()


def test_compare_of_runs_with_the_same_switch_says_nothing_of_it(here):
    _run("one")
    _run("two")
    setup = tensorquill.compare_runs("one", "two", source=here / "tensorquill-out")["setup"]
    assert not [d for d in setup["differs"] if d["field"] == "pinned_flush"]
    assert setup["same"] == ["project", "group", "world_size", "sdk_version", "torch_version"]
    assert "pinned_flush_not_recorded" not in setup


def _drop_switch(path: Path) -> None:
    import json

    document = json.loads(path.read_text())
    del document["pinned_flush"]
    path.write_text(json.dumps(document))


def test_an_older_manifest_without_the_switch_is_not_recorded_never_read_as_off(here):
    _run("old", pinned_flush=False)
    _run("new", pinned_flush=False)
    out = here / "tensorquill-out"
    _drop_switch(out / "old.run.json")
    result = tensorquill.compare_runs("old", "new", source=out)
    setup = result["setup"]
    assert not [d for d in setup["differs"] if d["field"] == "pinned_flush"]
    assert setup["pinned_flush_not_recorded"] == ["A"]
    assert any(line.startswith("   pinned_flush not recorded for A:") for line in result["summary"].splitlines())


def test_two_older_manifests_compare_as_before(here):
    _run("old1")
    _run("old2")
    out = here / "tensorquill-out"
    _drop_switch(out / "old1.run.json")
    _drop_switch(out / "old2.run.json")
    result = tensorquill.compare_runs("old1", "old2", source=out)
    assert "pinned_flush_not_recorded" not in result["setup"]
    assert "pinned_flush" not in result["summary"]
