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


# --- Item 1: each rank above 0 writes its numerical settings beside its records ------------------
#
# The author's answer to item 9 (option A of lane AQ-6): `<run_id>.rank<N>.settings.json` beside each
# rank's records, written at `init` by one line of `capture.py`. Rank 0's settings are in the run
# manifest, so rank 0 writes no such file.

SETTING = "os.environ.get('CUBLAS_WORKSPACE_CONFIG')"


def _as_rank(monkeypatch, rank: int, world: int = 2) -> None:
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world))


def _read(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


def test_a_rank_above_0_writes_its_settings_beside_its_records(here, monkeypatch):
    _as_rank(monkeypatch, 1)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    session, said, model = _init("two")
    _train(model, 1)
    session.close()
    out = here / "tensorquill-out"
    assert sorted(p.name for p in out.iterdir()) == ["two.rank1.end.json", "two.rank1.jsonl", "two.rank1.settings.json"]
    assert session.manifest is None and not [s for s in said if "settings" in s]
    written = _read(out / "two.rank1.settings.json")
    assert (written["schema_version"], written["run_id"], written["rank"]) == (1, "two", 1)
    block = written["numerical_settings"]
    assert block["read_at"] == "init" and block["starts"] == 1 and block["read_at_utc"].endswith("Z")
    assert block["settings"][SETTING] == {"value": ":4096:8"}
    # The same block as the manifest's on rank 0, read by the same reader.
    assert set(block["settings"]) == set(mf.numerical_settings_block()["settings"])
    assert mf.numerical_settings(written)["recorded"] is True
    assert mf.read_rank_settings(out / "two.rank1.settings.json") == written
    assert mf.rank_settings(out, "two") == {1: written}
    assert mf.rank_settings_path(out / "two.rank1.jsonl", "two", 1) == out / "two.rank1.settings.json"


def test_rank_0_writes_no_settings_file_its_settings_are_in_the_manifest(here, monkeypatch):
    _as_rank(monkeypatch, 0)
    _run("two")
    out = here / "tensorquill-out"
    assert not list(out.glob("*.settings.json"))
    assert mf.read_manifest(out / "two.run.json")["numerical_settings"]["read_at"] == "init"
    assert mf.rank_settings(out, "two") == {}


def test_a_second_writer_of_a_rank_writes_its_own_settings_file_and_leaves_the_first_writers(here, monkeypatch):
    _as_rank(monkeypatch, 2, 4)
    one, _, _ = _init("w")
    out = here / "tensorquill-out"
    first = (out / "w.rank2.settings.json").read_bytes()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    two, said, _ = _init("w")
    try:
        assert two._sink.path.name == f"w.rank2.{os.getpid()}.jsonl" and len(said) == 1, said
    finally:
        two.close()
        one.close()
    assert (out / "w.rank2.settings.json").read_bytes() == first
    second = _read(out / f"w.rank2.{os.getpid()}.settings.json")
    assert (second["run_id"], second["rank"]) == ("w", 2)
    assert second["numerical_settings"]["settings"][SETTING] == {"value": ":16:8"}
    assert mf.rank_settings_path(two._sink.path, "w", 2) == out / f"w.rank2.{os.getpid()}.settings.json"
    # The reader of a folder gives each rank's first writer's file.
    assert mf.rank_settings(out, "w") == {2: _read(out / "w.rank2.settings.json")}


def test_a_resume_keeps_both_starts_settings_when_they_differ(here, monkeypatch):
    _as_rank(monkeypatch, 1)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _run("again")
    out = here / "tensorquill-out"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    _run("again")
    block = _read(out / "again.rank1.settings.json")["numerical_settings"]
    assert block["settings"][SETTING] == {"value": ":16:8"} and block["starts"] == 1
    (earlier,) = block["earlier_starts"]
    assert earlier["settings"][SETTING] == {"value": ":4096:8"} and earlier["starts"] == 1


def test_a_resume_with_the_same_settings_counts_the_start(here, monkeypatch):
    _as_rank(monkeypatch, 1)
    _run("same")
    _run("same")
    block = _read(here / "tensorquill-out" / "same.rank1.settings.json")["numerical_settings"]
    assert block["starts"] == 2 and "earlier_starts" not in block


def test_another_runs_settings_file_at_the_path_is_replaced_not_merged(tmp_path):
    path = tmp_path / "r.rank1.settings.json"
    path.write_text('{"schema_version": 1, "run_id": "other", "rank": 1, "numerical_settings": '
                    '{"read_at": "init", "settings": {}, "starts": 5}}')
    undo: list = []
    mf.write_rank_settings(tmp_path / "r.rank1.jsonl", "r", 1, undo)
    written = _read(path)
    assert written["run_id"] == "r" and written["numerical_settings"]["starts"] == 1
    assert "earlier_starts" not in written["numerical_settings"]


def test_a_settings_file_that_cannot_be_written_never_raises_and_is_said_once(here, monkeypatch):
    _as_rank(monkeypatch, 1)
    out = here / "tensorquill-out"
    (out / "bad.rank1.settings.json").mkdir(parents=True)  # a folder where the file would go
    session, said, model = _init("bad")
    _train(model, 2)
    session.close()
    about = [s for s in said if "bad.rank1.settings.json" in s]
    assert len(about) == 1 and "the run goes on" in about[0], said
    assert (out / "bad.rank1.settings.json").is_dir()
    assert not [p for p in out.iterdir() if p.name.endswith(".tmp")]
    assert [p.name for p in out.glob("bad.rank1.jsonl")] == ["bad.rank1.jsonl"]


def test_a_settings_file_that_cannot_be_written_never_raises_under_w_error(here, monkeypatch):
    _as_rank(monkeypatch, 1)
    out = here / "tensorquill-out"
    (out / "e.rank1.settings.json").mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session = tensorquill.init(nn.Linear(2, 1), run_id="e", every=1, heartbeat_seconds=None,
                                   stack_seconds=None)
    session.close()


def test_a_failed_init_takes_the_settings_file_back(here, monkeypatch):
    _as_rank(monkeypatch, 1)
    import tensorquill.flight as flight

    def refuse(*args, **kwargs):
        raise RuntimeError("no stream today")

    monkeypatch.setattr(flight, "attach", refuse)
    with pytest.raises(RuntimeError, match="no stream today"):
        _init("undone")
    out = here / "tensorquill-out"
    assert not (out / "undone.rank1.settings.json").exists()
    # A file that was there before the failed init is put back as it was.
    monkeypatch.undo()
    _as_rank(monkeypatch, 1)
    _run("undone")
    before = (out / "undone.rank1.settings.json").read_bytes()
    monkeypatch.setattr(flight, "attach", refuse)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="no stream today"):
        _init("undone")
    assert (out / "undone.rank1.settings.json").read_bytes() == before


def test_a_sink_with_no_file_writes_no_settings_file(here, monkeypatch):
    _as_rank(monkeypatch, 1)
    session, _, model = _init("mem", sink=tensorquill.MemorySink())
    _train(model, 1)
    session.close()
    assert not (here / "tensorquill-out").exists() or not list((here / "tensorquill-out").iterdir())


def test_a_sink_of_a_name_of_its_own_gets_the_ranks_settings_file_beside_it(here, monkeypatch):
    _as_rank(monkeypatch, 1)
    folder = here / "mine"
    folder.mkdir()
    session, _, model = _init("own", sink=tensorquill.JsonlSink(folder / "learner.jsonl"))
    _train(model, 1)
    session.close()
    assert _read(folder / "own.rank1.settings.json")["rank"] == 1


def test_the_reader_takes_only_files_of_the_run_and_rank_their_names_say(tmp_path):
    def put(name: str, run_id: str, rank: int) -> None:
        (tmp_path / name).write_text('{"schema_version": 1, "run_id": "%s", "rank": %d, "numerical_settings": '
                                     '{"read_at": "init", "settings": {}}}' % (run_id, rank))

    put("r.rank1.settings.json", "r", 1)
    put("r.rank2.settings.json", "r", 3)  # its document names another rank: not rank 2's
    put("r.rank1.rank3.settings.json", "r.rank1", 3)  # run `r.rank1`, not run `r`
    put("r.rank4.77.settings.json", "r", 4)  # a second writer's
    (tmp_path / "r.rank5.settings.json").write_text("not json")
    found = mf.rank_settings(tmp_path, "r")
    assert list(found) == [1]
    assert list(mf.rank_settings(tmp_path, "r.rank1")) == [3]
    assert mf.read_rank_settings(tmp_path / "r.rank5.settings.json") is None
    assert mf.read_rank_settings(tmp_path / "missing.settings.json") is None


def _two_ranks(monkeypatch, run_id: str, rank1_env: str | None = None) -> None:
    _as_rank(monkeypatch, 0)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    _run(run_id)
    _as_rank(monkeypatch, 1)
    if rank1_env is not None:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", rank1_env)
    _run(run_id)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    for name in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)


def test_the_folder_readers_do_not_take_the_settings_file_for_records_or_a_manifest(here, monkeypatch):
    from tensorquill import runs as _runs

    _two_ranks(monkeypatch, "pair")
    out = here / "tensorquill-out"
    assert (out / "pair.rank1.settings.json").is_file()
    (found,) = _runs.discover_runs(out)
    assert found["run_id"] == "pair" and sorted(found["files"]) == [0, 1]
    assert "more_files" not in found and "other_writers" not in found
    assert _sink.writers_of(out / "pair.rank1.jsonl") == [out / "pair.rank1.jsonl"]
    (with_file,) = tensorquill.list_runs(out)
    hidden = here / "aside.settings.json"
    (out / "pair.rank1.settings.json").rename(hidden)
    (without,) = tensorquill.list_runs(out)
    hidden.rename(out / "pair.rank1.settings.json")
    assert with_file == without
    with pytest.raises(ValueError, match="not a folder of runs or a .jsonl capture"):
        _runs.discover_runs(out / "pair.rank1.settings.json")
    # The file given alone is a rank's records, and the evidence of rank 1's file is as it was.
    assert tensorquill.build_evidence(out / "pair.rank1.jsonl")["run"]["run_id"] == "pair"


def test_compare_says_when_a_rank_differs_from_rank_0_in_its_settings(here, monkeypatch):
    _two_ranks(monkeypatch, "base")
    _two_ranks(monkeypatch, "odd", rank1_env=":4096:8")
    result = tensorquill.compare_runs("base", "odd", source=here / "tensorquill-out")
    by_rank = result["setup"]["numerical_settings_ranks"]
    assert by_rank == {"A": {"compared_with_rank_0": [1]},
                       "B": {"compared_with_rank_0": [1], "differ": [{"rank": 1, "settings": [SETTING]}]}}
    assert (f"   B's rank 1 differs from its rank 0 in its numerical settings: {SETTING}"
            in result["summary"].splitlines())


def test_compare_says_when_the_other_ranks_settings_were_not_recorded(here, monkeypatch):
    _two_ranks(monkeypatch, "base")
    _two_ranks(monkeypatch, "older")
    out = here / "tensorquill-out"
    (out / "older.rank1.settings.json").unlink()  # as a run recorded before the file existed
    result = tensorquill.compare_runs("base", "older", source=out)
    assert result["setup"]["numerical_settings_ranks"]["B"] == {"compared_with_rank_0": [], "not_recorded": [1]}
    assert any(line.startswith("   numerical settings of B's rank 1 not recorded") for line in
               result["summary"].splitlines())


def test_compare_of_runs_of_one_rank_adds_nothing_for_the_other_ranks(here):
    _run("solo1")
    _run("solo2")
    setup = tensorquill.compare_runs("solo1", "solo2", source=here / "tensorquill-out")["setup"]
    assert "numerical_settings_ranks" not in setup
