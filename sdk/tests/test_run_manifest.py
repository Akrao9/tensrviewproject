"""Project, run name and the run manifest (plan: "Projects, run names and comparing runs", piece 1).

- `tensorquill.init(project=, name=, group=, tags=, config=)` writes `<run_id>.run.json`
  beside the `.jsonl` file, on rank 0 only, and never stops a run;
- the labels resolve as argument, environment, tracker, default;
- every evidence file carries a `run` block, the same one whichever way it is built, and a
  run recorded before the manifest still reads;
- the manifest rides on the first batch of each of rank 0's streams, and the real server
  writes `runs/<run_id>/run.json` from it. A sender without one still works.

The real-server tests need the binary (`TENSORQUILL_SERVER_BIN` or `server/target`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

import tensorquill
from tensorquill import evidence as ev
from tensorquill import manifest as mf

GOLDEN_KEYS = ("incidents", "kind_checks", "not_checked", "run_state", "samples")


@pytest.fixture(autouse=True)
def no_tracker(monkeypatch):
    for name in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_NAME", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID",
                 "RANK", "LOCAL_RANK", "WORLD_SIZE", mf.PROJECT_ENV, mf.NAME_ENV, "TENSORQUILL_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


def run(model, tmp_path: Path, run_id: str, steps: int = 3, **labels) -> Path:
    """A short run of the toy model; returns its `.jsonl` file."""
    path = tmp_path / f"{run_id}.rank0.jsonl"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with tensorquill.init(model, run_id=run_id, routers={"router": 4}, capture="full", every=2,
                          sink=tensorquill.JsonlSink(path), **labels) as session:
        for _ in range(steps):
            optimizer.zero_grad()
            model(torch.randn(16, 8)).sum().backward()
            optimizer.step()
        assert session.manifest is not None
    return path


# --- The local file -----------------------------------------------------------------------


def test_init_writes_the_manifest_beside_the_records_with_every_field(toy_moe, tmp_path):
    path = run(toy_moe, tmp_path, "run-1", project="granite", name="baseline", group="sweep-a",
               tags=["moe", "bf16"], config={"lr": 0.01, "layers": 4})
    written = json.loads((tmp_path / "run-1.run.json").read_text())
    assert written.pop("created_at").endswith("Z")
    assert written == {
        "schema_version": 1, "project": "granite", "name": "baseline", "run_id": "run-1", "group": "sweep-a",
        "tags": ["moe", "bf16"], "world_size": 1, "sdk_version": tensorquill.__version__,
        "torch_version": str(torch.__version__), "run_id_source": "explicit", "config": {"lr": 0.01, "layers": 4},
    }
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["run-1.rank0.jsonl", "run-1.run.json"]
    assert ev.build_evidence(path)["run"] == {"project": "granite", "name": "baseline", "run_id": "run-1"}


def test_two_runs_of_one_project_are_told_apart_by_name_from_their_local_files(tmp_path, toy_moe):
    blocks = [ev.build_evidence(run(toy_moe, tmp_path, run_id, project="granite", name=name))["run"]
              for run_id, name in (("run-1", "baseline"), ("run-2", "after-the-fix"))]
    assert blocks == [
        {"project": "granite", "name": "baseline", "run_id": "run-1"},
        {"project": "granite", "name": "after-the-fix", "run_id": "run-2"},
    ]


def test_the_default_sink_gets_its_manifest_in_tensorquill_out(toy_moe, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tensorquill.init(toy_moe, run_id="r", name="mine").close()
    assert json.loads(Path("tensorquill-out/r.run.json").read_text())["name"] == "mine"


def test_a_sink_with_no_file_writes_no_manifest_and_the_session_still_holds_it(toy_moe, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = tensorquill.init(toy_moe, run_id="r", project="p", sink=tensorquill.MemorySink())
    session.close()
    assert session.manifest["project"] == "p" and list(tmp_path.iterdir()) == []


def test_the_run_block_is_the_same_however_the_evidence_is_built(toy_moe, tmp_path, monkeypatch, capsys):
    path = run(toy_moe, tmp_path, "run-1", project="granite", name="baseline")
    block = {"project": "granite", "name": "baseline", "run_id": "run-1"}
    records = ev.load_records(path)
    assert ev.build_evidence(path)["run"] == block
    assert ev.build_evidence(records, run_manifest=path)["run"] == block
    assert ev.build_evidence(records, run_manifest=tmp_path / "run-1.run.json")["run"] == block
    assert ev.build_evidence(records, run_manifest={"project": "granite", "name": "baseline"})["run"] == block
    # Records alone say nothing of a project.
    assert ev.build_evidence(records)["run"] == {"project": "uncategorized", "name": "run-1", "run_id": "run-1"}
    # The command line, and one look of the watcher.
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(path)])
    ev.main()
    assert json.loads(path.with_suffix(".evidence.json").read_text())["run"] == block
    out = ev.watch(path, tmp_path / "watched.json", polls=1)
    assert json.loads(out.read_text())["run"] == block
    # The compact file keeps the block.
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(path), "--compact", "-o", str(tmp_path / "c.json")])
    ev.main()
    assert json.loads((tmp_path / "c.json").read_text())["run"] == block


def test_the_run_block_changes_none_of_the_keys_the_golden_files_freeze(toy_moe, tmp_path):
    path = run(toy_moe, tmp_path, "run-1", steps=12, project="granite", name="baseline")
    records = ev.load_records(path)
    with_manifest, without = ev.build_evidence(path), ev.build_evidence(records)
    assert with_manifest["run"] != without["run"]
    for key in GOLDEN_KEYS:
        assert with_manifest.get(key) == without.get(key), key
    assert {k for k in with_manifest if with_manifest[k] != without.get(k)} == {"run"}


def test_a_run_recorded_before_the_manifest_still_reads(toy_moe, tmp_path):
    path = run(toy_moe, tmp_path, "old-run", project="granite", name="baseline")
    (tmp_path / "old-run.run.json").unlink()
    assert ev.build_evidence(path)["run"] == {"project": "uncategorized", "name": "old-run", "run_id": "old-run"}
    # A file that is not a manifest, or one of another run, is not this run's.
    (tmp_path / "old-run.run.json").write_text("not json")
    assert ev.build_evidence(path)["run"]["project"] == "uncategorized"
    (tmp_path / "old-run.run.json").write_text(json.dumps({"run_id": "another", "project": "theirs", "name": "x"}))
    assert ev.build_evidence(path)["run"] == {"project": "uncategorized", "name": "old-run", "run_id": "old-run"}
    # A project name in a hand-edited manifest that breaks the rule is not passed on.
    (tmp_path / "old-run.run.json").write_text(json.dumps({"run_id": "old-run", "project": "a/b", "name": 7}))
    assert ev.build_evidence(path)["run"] == {"project": "uncategorized", "name": "old-run", "run_id": "old-run"}


def test_a_run_id_with_dots_finds_its_manifest_from_the_file_name(tmp_path):
    assert mf.run_id_of_file("out/run-slurm1234.0.rank12.jsonl") == "run-slurm1234.0"
    assert mf.run_id_of_file("out/trainer.jsonl") is None
    (tmp_path / "run-slurm1234.0.run.json").write_text(json.dumps({"run_id": "run-slurm1234.0", "project": "p"}))
    assert mf.find_manifest(tmp_path / "run-slurm1234.0.rank3.jsonl")["project"] == "p"
    # A file with a name of its own is found by the run ID in its records.
    assert mf.find_manifest(tmp_path / "trainer.jsonl") is None
    assert mf.find_manifest(tmp_path / "trainer.jsonl", "run-slurm1234.0")["project"] == "p"


# --- How the labels resolve -----------------------------------------------------------------


def labels_of(model, tmp_path, monkeypatch, **labels) -> dict:
    monkeypatch.chdir(tmp_path)
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), **labels)
    session.close()
    return session.manifest


def test_project_and_name_resolve_as_argument_environment_tracker_default(toy_moe, tmp_path, monkeypatch):
    found = labels_of(toy_moe, tmp_path, monkeypatch, run_id="r")
    assert (found["project"], found["name"], found["group"], found["run_id_source"]) == ("uncategorized", "r", None, "explicit")

    wandb_run = types.SimpleNamespace(id="w-1", project="from-wandb", name="wandb-name", group="wandb-group")
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=wandb_run))
    found = labels_of(toy_moe, tmp_path, monkeypatch)
    assert (found["project"], found["name"], found["group"], found["run_id"], found["run_id_source"]) == (
        "from-wandb", "wandb-name", "wandb-group", "w-1", "wandb")

    monkeypatch.setenv(mf.PROJECT_ENV, "from-env")
    monkeypatch.setenv(mf.NAME_ENV, "env-name")
    found = labels_of(toy_moe, tmp_path, monkeypatch)
    assert (found["project"], found["name"], found["group"]) == ("from-env", "env-name", "wandb-group")

    found = labels_of(toy_moe, tmp_path, monkeypatch, project="mine", name="my-name", group="my-group")
    assert (found["project"], found["name"], found["group"], found["run_id"]) == ("mine", "my-name", "my-group", "w-1")


def test_mlflow_gives_the_run_name_and_the_experiment_only_from_the_environment(toy_moe, tmp_path, monkeypatch):
    def no_round_trip(*args, **kwargs):
        raise AssertionError("the tracking server must not be asked at init")

    active = types.SimpleNamespace(info=types.SimpleNamespace(run_id="m-1", run_name="mlflow-name", experiment_id="7"))
    monkeypatch.setitem(sys.modules, "mlflow", types.SimpleNamespace(
        active_run=lambda: active, get_experiment=no_round_trip, get_run=no_round_trip, MlflowClient=no_round_trip))
    found = labels_of(toy_moe, tmp_path, monkeypatch)
    assert (found["project"], found["name"], found["run_id_source"]) == ("uncategorized", "mlflow-name", "mlflow")
    # An MLflow experiment is often a path: it is made to fit, and said.
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "/Users/me/granite")
    with pytest.warns(RuntimeWarning, match="cannot be a TensorQuill project name"):
        found = labels_of(toy_moe, tmp_path, monkeypatch)
    assert found["project"] == "-Users-me-granite"


def test_a_tracker_whose_fields_raise_does_not_stop_the_run(toy_moe, tmp_path, monkeypatch):
    class Run:
        id = "w-2"

        @property
        def project(self):
            raise RuntimeError("shared mode")

        name = None

        @property
        def step(self):
            raise AssertionError("wandb.run.step must not be read")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=Run()))
    found = labels_of(toy_moe, tmp_path, monkeypatch)
    assert (found["project"], found["name"], found["group"]) == ("uncategorized", "w-2", None)


@pytest.mark.parametrize("bad", ["a/b", "a\\b", "a#b", "a?b", "a%b", "a:b", "", "x" * 129])
def test_a_project_argument_that_breaks_w_and_bs_rule_is_refused_before_anything_is_written(bad, toy_moe, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="project"):
        tensorquill.init(toy_moe, run_id="r", project=bad)
    assert list(tmp_path.iterdir()) == []
    assert mf.project_problem("x" * 128) is None and mf.project_problem("granite moe-v2.1") is None


def test_a_project_from_the_environment_that_breaks_the_rule_is_made_to_fit_with_a_warning(toy_moe, tmp_path, monkeypatch):
    monkeypatch.setenv(mf.PROJECT_ENV, "team/granite:" + "x" * 200)
    with pytest.warns(RuntimeWarning, match="TENSORQUILL_PROJECT"):
        found = labels_of(toy_moe, tmp_path, monkeypatch, run_id="r")
    assert found["project"] == ("team-granite-" + "x" * 200)[:128]
    assert mf.project_problem(found["project"]) is None


def test_other_bad_labels_are_refused_and_config_values_become_plain(toy_moe, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for labels in ({"config": ["lr", 0.1]}, {"name": 5}, {"group": 5}, {"tags": 5}, {"project": 5}):
        with pytest.raises(ValueError):
            tensorquill.init(toy_moe, run_id="r", sink=tensorquill.MemorySink(), **labels)
    found = labels_of(toy_moe, tmp_path, monkeypatch, run_id="r", tags="one", config={
        "lr": 1e-3, "warmup": float("nan"), "path": Path("data"), "betas": (0.9, 0.99), 7: None,
        "dtype": torch.bfloat16, "nested": {"a": float("inf")},
    })
    assert found["tags"] == ["one"]
    assert found["config"] == {"lr": 1e-3, "warmup": "nan", "path": "data", "betas": [0.9, 0.99], "7": None,
                               "dtype": "torch.bfloat16", "nested": {"a": "inf"}}
    json.loads(json.dumps(found, allow_nan=False))


# --- Nothing here stops a run -----------------------------------------------------------------


def test_a_manifest_that_cannot_be_written_is_warned_about_and_the_run_goes_on(toy_moe, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(mf.os, "replace", refuse)
    with pytest.warns(RuntimeWarning, match="could not write the run manifest"):
        path = run(toy_moe, tmp_path, "run-1", project="granite")
    assert len(ev.load_records(path)) == 3
    assert ev.build_evidence(path)["run"]["project"] == "uncategorized"


def test_a_run_started_again_under_its_run_id_keeps_the_first_start_time(toy_moe, tmp_path):
    run(toy_moe, tmp_path, "run-1", name="first")
    file = tmp_path / "run-1.run.json"
    earlier = {**json.loads(file.read_text()), "created_at": "2026-01-02T03:04:05Z"}
    file.write_text(json.dumps(earlier))
    run(toy_moe, tmp_path, "run-1", name="second")
    again = json.loads(file.read_text())
    assert (again["created_at"], again["name"]) == ("2026-01-02T03:04:05Z", "second")
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# --- Other ranks ------------------------------------------------------------------------------


_RANK_JOB = """
import sys, torch, tensorquill
from torch import nn
model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
with tensorquill.init(model, run_id="two", project="granite", name="pair", config={"lr": 0.01}, every=2) as session:
    for _ in range(4):
        optimizer.zero_grad()
        model(torch.randn(8, 4)).sum().backward()
        optimizer.step()
print("manifest" if session.manifest else "none")
"""


def test_a_two_process_run_writes_one_manifest_and_both_ranks_read_it(tmp_path):
    jobs = []
    for rank in (1, 0):
        env = {**os.environ, "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
               "PYTHONPATH": os.pathsep.join(sys.path)}
        env.pop("TENSORQUILL_ENDPOINT", None)
        jobs.append((rank, subprocess.Popen([sys.executable, "-c", _RANK_JOB], cwd=tmp_path, env=env,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    said = {}
    for rank, job in jobs:
        out, err = job.communicate(timeout=120)
        assert job.returncode == 0, err
        said[rank] = out.strip()
    assert said == {0: "manifest", 1: "none"}
    out = tmp_path / "tensorquill-out"
    assert sorted(p.name for p in out.iterdir()) == ["two.rank0.jsonl", "two.rank1.jsonl", "two.run.json"]
    written = json.loads((out / "two.run.json").read_text())
    assert (written["world_size"], written["name"], written["config"]) == (2, "pair", {"lr": 0.01})
    block = {"project": "granite", "name": "pair", "run_id": "two"}
    assert [ev.build_evidence(out / f"two.rank{rank}.jsonl")["run"] for rank in (0, 1)] == [block, block]


def test_another_rank_checks_its_arguments_and_writes_nothing(toy_moe, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RANK", "1")
    with pytest.raises(ValueError, match="project"):
        tensorquill.init(toy_moe, run_id="r", project="a/b")
    session = tensorquill.init(toy_moe, run_id="r", project="granite")
    session.close()
    assert session.manifest is None
    assert [p.name for p in (tmp_path / "tensorquill-out").iterdir()] == ["r.rank1.jsonl"]


# --- The sender ---------------------------------------------------------------------------------


def test_the_manifest_rides_on_the_first_batch_of_each_stream_and_a_sender_without_one_adds_no_key(tmp_path):
    pytest.importorskip("pyarrow.flight")
    from test_flight import FakeServer, Store, append, fast_sender, record, wait_for

    store = Store(tmp_path / "server-store")
    server = FakeServer(store)
    try:
        path = tmp_path / "run-a.rank0.jsonl"
        manifest = {"schema_version": 1, "project": "granite", "name": "baseline", "run_id": "run-a", "config": {"lr": 0.1}}
        append(path, [record(step) for step in range(3)])
        sender = fast_sender(path, server.endpoint, manifest=manifest)
        wait_for(lambda: len(store.batches) == 1)
        append(path, [record(step) for step in range(3, 6)])
        wait_for(lambda: len(store.batches) == 2)
        port = server.port
        server.stop()  # the next stream is a new one
        server = FakeServer(store, port)
        append(path, [record(step) for step in range(6, 9)])
        assert sender.close() is True
        carried = [batch.get(mf.METADATA_KEY) for batch in store.batches]
        assert carried[0] == manifest and carried[1] is None
        assert carried[-1] == manifest, "a server that restarted is told again"
        assert store.steps() == list(range(9))

        # A sender with no manifest (every rank but 0, and every sender older than the key).
        other = tmp_path / "old" / "run-a.rank0.jsonl"
        other.parent.mkdir()
        append(other, [record(step) for step in range(9, 12)])
        before = len(store.batches)
        assert fast_sender(other, server.endpoint).close() is True
        assert set(store.batches[before]) == {"schema_version", "run_id", "rank", "min_step", "max_step",
                                              "end_byte_offset", "rows"}
    finally:
        server.stop()


# --- The real server ------------------------------------------------------------------------------


def real_server(store: Path):
    pytest.importorskip("pyarrow.flight")
    import test_flight_e2e as e2e

    if e2e.BINARY is None:
        pytest.skip("tensorquill-server is not built")
    return e2e.Server(store, 0, "--flush-seconds", "0.2", "--watchdog-seconds", "0.2")


def streamed(model, tmp_path: Path, monkeypatch, endpoint: str, run_id: str, **labels) -> None:
    monkeypatch.setenv("TENSORQUILL_ENDPOINT", endpoint)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with tensorquill.init(model, run_id=run_id, routers={"router": 4}, capture="full", every=5,
                          sink=tensorquill.JsonlSink(tmp_path / "local" / f"{run_id}.rank0.jsonl"), **labels):
        for _ in range(10):
            optimizer.zero_grad()
            model(torch.randn(16, 8)).sum().backward()
            optimizer.step()
    monkeypatch.delenv("TENSORQUILL_ENDPOINT")


def test_two_runs_of_one_project_are_told_apart_in_a_servers_store_by_every_reader(toy_moe, tmp_path, monkeypatch):
    store = tmp_path / "store"
    server = real_server(store)
    try:
        streamed(toy_moe, tmp_path, monkeypatch, server.endpoint, "run-1", project="granite", name="baseline",
                 config={"lr": 0.01})
        streamed(toy_moe, tmp_path, monkeypatch, server.endpoint, "run-2", project="granite", name="after-the-fix")
    finally:
        assert server.terminate() == 0
    for run_id, name in (("run-1", "baseline"), ("run-2", "after-the-fix")):
        folder = store / "runs" / run_id
        block = {"project": "granite", "name": name, "run_id": run_id}
        # What the server wrote is what rank 0 wrote locally.
        assert json.loads((folder / "run.json").read_text()) == json.loads(
            (tmp_path / "local" / f"{run_id}.run.json").read_text())
        from_local = ev.build_evidence(tmp_path / "local" / f"{run_id}.rank0.jsonl")
        from_parts = ev.build_evidence(folder)
        from_rank_folder = ev.build_evidence(folder / "rank=0")
        from_report = json.loads(ev.watch(folder, tmp_path / f"{run_id}.report.json", polls=1).read_text())
        recomputed = json.loads(ev.watch(folder, tmp_path / f"{run_id}.here.json", polls=1,
                                         server_reports=False).read_text())
        all_ranks = json.loads(ev.watch(folder, tmp_path / f"{run_id}.all.json", rank="all", polls=1).read_text())
        assert from_report["detection"]["source"] == "server"  # built by `_report_evidence`, no part read
        assert recomputed.get("detection", {}).get("source") != "server" and recomputed["samples"]
        for evidence in (from_local, from_parts, from_rank_folder, from_report, recomputed, all_ranks):
            assert evidence["run"] == block
        assert ev.all_ranks_evidence(folder)["run"] == block


def test_a_sender_older_than_the_manifest_still_streams_and_its_run_reads_as_uncategorized(toy_moe, tmp_path, monkeypatch):
    import tensorquill.flight as flight_module

    # An older sender: the same code with the manifest never handed to it.
    real_attach = flight_module.attach
    monkeypatch.setattr(flight_module, "attach", lambda *a, manifest=None, **k: real_attach(*a, **k))
    store = tmp_path / "store"
    server = real_server(store)
    try:
        streamed(toy_moe, tmp_path, monkeypatch, server.endpoint, "old-sender", project="granite", name="unseen")
    finally:
        assert server.terminate() == 0
    folder = store / "runs" / "old-sender"
    assert not (folder / "run.json").exists()
    assert len(ev.load_records(folder)) == 10
    block = {"project": "uncategorized", "name": "old-sender", "run_id": "old-sender"}
    assert ev.build_evidence(folder)["run"] == block
    assert json.loads(ev.watch(folder, tmp_path / "all.json", rank="all", polls=1).read_text())["run"] == block
