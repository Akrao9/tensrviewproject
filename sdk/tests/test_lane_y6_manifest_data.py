"""Lane Y6 (wave Y): the run manifest and checkpoint and comparison metadata.

- XC3: three recoverable metadata problems never stop `tensorquill.init`, even when warnings
  are errors (`python -W error`): a project name from the environment or the tracker that
  has to be made to fit, a manifest whose rename fails, and a manifest too large for a
  stream. The run trains and records, the message is still said (on the `tensorquill`
  logger when the warning cannot be), and an explicit bad argument still raises.
- XC9: a mapping of tensors is the model component only when it looks like a model's state
  dict (its key names a model, or its entries are parameter paths), so a mapping of cached
  tensors cannot stand in for the model; and `grads_not_read` is a note, not a series that
  `first_divergence` lists as not compared.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

import tensorquill
from tensorquill import manifest as mf
from tensorquill.data import checkpoint_manifest
from tensorquill.divergence import first_divergence


def test_the_package_under_test_is_this_checkout():
    here = Path(__file__).resolve().parents[1] / "src" / "tensorquill"
    assert Path(tensorquill.__file__).resolve().parent == here


@pytest.fixture(autouse=True)
def no_tracker(monkeypatch):
    for name in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_NAME", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID",
                 "RANK", "LOCAL_RANK", "WORLD_SIZE", mf.PROJECT_ENV, mf.NAME_ENV, "TENSORQUILL_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


def train(tmp_path: Path, run_id: str = "r", steps: int = 4, *, folder: str = "", **options) -> Path:
    """A short run under an error filter for every warning, as `python -W error` sets it;
    returns its `.jsonl` file."""
    path = tmp_path / folder / f"{run_id}.rank0.jsonl"
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with tensorquill.init(model, run_id=run_id, every=2, heartbeat_seconds=None,
                              sink=tensorquill.JsonlSink(path), **options):
            for _ in range(steps):
                optimizer.zero_grad()
                model(torch.randn(2, 4)).sum().backward()
                optimizer.step()
    return path


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def said(caplog, text: str) -> bool:
    return any(text in r.getMessage() for r in caplog.records if r.name == "tensorquill")


# --- XC3: the project name from the environment or the tracker ------------------------------


def test_xc3_a_project_from_the_environment_is_made_to_fit_under_an_error_filter(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(mf.PROJECT_ENV, "my/project")
    caplog.set_level(logging.WARNING, logger="tensorquill")
    path = train(tmp_path)
    assert len(records(path)) == 4
    assert json.loads((tmp_path / "r.run.json").read_text())["project"] == "my-project"
    assert said(caplog, "TENSORQUILL_PROJECT is 'my/project'")


def test_xc3_a_tracker_project_is_made_to_fit_under_an_error_filter(tmp_path, monkeypatch, caplog):
    wandb_run = types.SimpleNamespace(id="w-1", project="team/granite", name="wandb-name", group=None)
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=wandb_run))
    caplog.set_level(logging.WARNING, logger="tensorquill")
    path = train(tmp_path)
    assert len(records(path)) == 4
    assert json.loads((tmp_path / "r.run.json").read_text())["project"] == "team-granite"
    assert said(caplog, "the tracker's project is 'team/granite'")


def test_xc3_without_an_error_filter_the_project_warning_is_still_a_warning_at_the_callers_line(tmp_path, monkeypatch):
    monkeypatch.setenv(mf.PROJECT_ENV, "my/project")
    with pytest.warns(RuntimeWarning, match="TENSORQUILL_PROJECT") as caught:
        tensorquill.init(torch.nn.Linear(2, 2), run_id="r", heartbeat_seconds=None,
                         sink=tensorquill.JsonlSink(tmp_path / "r.rank0.jsonl")).close()
    [warning] = [w for w in caught if "TENSORQUILL_PROJECT" in str(w.message)]
    assert warning.filename == __file__


# --- XC3: a manifest whose rename fails --------------------------------------------------------


def refuse_manifest_rename(monkeypatch):
    """`os.replace` fails for the manifest only: the end marker's rename (`sink.py`) is another
    file's and still works."""
    replace = os.replace

    def refuse(source, target, *args, **kwargs):
        if str(target).endswith(mf.MANIFEST_SUFFIX):
            raise PermissionError("read-only file system")
        return replace(source, target, *args, **kwargs)

    monkeypatch.setattr(mf.os, "replace", refuse)


def test_xc3_a_manifest_whose_rename_fails_lets_the_run_train_under_an_error_filter(tmp_path, monkeypatch, caplog):
    refuse_manifest_rename(monkeypatch)
    caplog.set_level(logging.WARNING, logger="tensorquill")
    path = train(tmp_path, project="granite")
    assert len(records(path)) == 4
    assert not (tmp_path / "r.run.json").exists()
    assert said(caplog, "could not write the run manifest")
    assert tensorquill.evidence.build_evidence(path)["run"]["project"] == "uncategorized"


def test_xc3_without_an_error_filter_the_rename_warning_is_still_a_warning_at_the_callers_line(tmp_path, monkeypatch):
    refuse_manifest_rename(monkeypatch)
    with pytest.warns(RuntimeWarning, match="could not write the run manifest") as caught:
        tensorquill.init(torch.nn.Linear(2, 2), run_id="r", heartbeat_seconds=None,
                         sink=tensorquill.JsonlSink(tmp_path / "r.rank0.jsonl")).close()
    [warning] = [w for w in caught if "could not write the run manifest" in str(w.message)]
    assert warning.filename == __file__


# --- XC3: a manifest too large for a stream ---------------------------------------------------


def test_xc3_a_70_kb_config_streams_without_it_under_an_error_filter(tmp_path, monkeypatch, caplog):
    pytest.importorskip("pyarrow.flight")
    from test_flight import FakeServer, Store

    store = Store(tmp_path / "server-store")
    server = FakeServer(store)
    caplog.set_level(logging.WARNING, logger="tensorquill")
    config = {"lr": 0.1, "blob": "x" * 70_000}
    try:
        monkeypatch.setenv("TENSORQUILL_ENDPOINT", server.endpoint)
        path = train(tmp_path, folder="local", project="granite", name="wide", config=config)
        monkeypatch.delenv("TENSORQUILL_ENDPOINT")
    finally:
        server.stop()
    assert len(records(path)) == 4
    assert store.steps() == [0, 1, 2, 3], "every record was streamed"
    sent = store.batches[0][mf.METADATA_KEY]
    assert (sent["project"], sent["name"], sent["config"], sent["config_left_out"]) == ("granite", "wide", {}, True)
    assert json.loads((tmp_path / "local" / "r.run.json").read_text())["config"] == config
    assert said(caplog, "more than the 60,000 a stream carries")


def test_xc3_a_70_kb_config_reaches_the_real_server_under_an_error_filter(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow.flight")
    import test_flight_e2e as e2e

    if e2e.BINARY is None:
        pytest.skip("tensorquill-server is not built")
    store = tmp_path / "store"
    server = e2e.Server(store, 0, "--flush-seconds", "0.2", "--watchdog-seconds", "0.2")
    try:
        monkeypatch.setenv("TENSORQUILL_ENDPOINT", server.endpoint)
        train(tmp_path, "big", folder="local", project="granite", name="wide", config={"blob": "x" * 70_000})
        monkeypatch.delenv("TENSORQUILL_ENDPOINT")
    finally:
        assert server.terminate() == 0
    stored = json.loads((store / "runs" / "big" / "run.json").read_text())
    assert (stored["project"], stored["name"], stored["config"], stored["config_left_out"]) == ("granite", "wide", {}, True)
    assert len(tensorquill.evidence.load_records(store / "runs" / "big")) == 4


# --- XC3: an explicit bad argument still raises -----------------------------------------------


@pytest.mark.parametrize("options", [{"project": "my/project"}, {"config": "lr=0.1"}, {"name": 3}])
def test_xc3_an_explicit_bad_argument_still_raises_under_an_error_filter(tmp_path, monkeypatch, options):
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            tensorquill.init(torch.nn.Linear(2, 2), run_id="r", heartbeat_seconds=None, **options)
    assert list(tmp_path.iterdir()) == []


def test_xc3_a_filter_that_ignores_the_warning_still_silences_it(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(mf.PROJECT_ENV, "my/project")
    caplog.set_level(logging.WARNING, logger="tensorquill")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensorquill.init(torch.nn.Linear(2, 2), run_id="r", heartbeat_seconds=None,
                         sink=tensorquill.JsonlSink(tmp_path / "r.rank0.jsonl")).close()
    assert not said(caplog, "TENSORQUILL_PROJECT")


# --- XC9: what a checkpoint's model component is -----------------------------------------------


def test_xc9_a_mapping_of_cached_tensors_is_not_the_model():
    manifest = checkpoint_manifest({
        "embedding_cache": {"cached_embedding": torch.zeros(4)},
        "optimizer": {"state": {}, "param_groups": [{}]},
    })
    assert "model" not in manifest["components"]
    assert manifest["other"] == ["embedding_cache.cached_embedding"]


@pytest.mark.parametrize("key", ["embedding_cache", "cache", "model_cache", "weights_cache", "ema_cache", "module_cache",
                                 "state", "dict", "cache_state_dict"])
def test_xc9_a_cache_is_not_the_model_whatever_word_its_name_holds(key):
    manifest = checkpoint_manifest({key: {"cached_embedding": torch.zeros(4), "cached_keys": torch.zeros(4)}})
    assert "model" not in manifest["components"]


def test_xc9_a_mapping_of_cached_tensors_beside_the_model_is_not_listed_as_the_model():
    model = torch.nn.Linear(2, 2)
    manifest = checkpoint_manifest({
        "model": model.state_dict(),
        "cache": {"cached_embedding": torch.zeros(4), "cached_keys": torch.zeros(4)},
    })
    assert manifest["components"]["model"]["keys"] == ["model"]
    assert manifest["components"]["model"]["tensors"] == 2


def test_xc9_a_resume_whose_checkpoint_holds_only_cached_tensors_finds_the_model_missing():
    from tensorquill import data_checks

    first = {"checkpoint": {"load": checkpoint_manifest({
        "embedding_cache": {"cached_embedding": torch.zeros(4)},
        "optimizer": {"state": {0: {"step": 3}}, "param_groups": [{}]},
    })}}
    findings, _, checked = data_checks._state_findings(first, 3, 3, None, False, None, None)
    # Wave AB (STEP_AND_EVIDENCE_CONTRACT.md, section 4): not found is `not_identified`, not `missing`.
    assert checked and {"component": "model", "finding": "not_identified"} in findings


class _Batched(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.norm = torch.nn.BatchNorm1d(2)


class _OnlyCache(torch.nn.Module):
    """A model whose state dict is a single buffer named like a cache."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cached_embedding", torch.zeros(4))


def _ddp_keys(module: torch.nn.Module) -> dict:
    return {f"module.{k}": v for k, v in module.state_dict().items()}


def _compiled_keys(module: torch.nn.Module) -> dict:
    return {f"_orig_mod.{k}": v for k, v in module.state_dict().items()}


@pytest.mark.parametrize("key, state, tensors", [
    ("model", torch.nn.Linear(2, 2).state_dict(), 2),  # weight, bias
    ("state_dict", torch.nn.Linear(2, 2).state_dict(), 2),  # Lightning
    ("ema", torch.nn.BatchNorm1d(2).state_dict(), 5),  # running_mean, running_var, num_batches_tracked
    ("generator", torch.nn.LSTM(2, 2).state_dict(), 4),  # weight_ih_l0, ...
    ("teacher", torch.nn.MultiheadAttention(4, 2).state_dict(), 4),  # in_proj_weight, out_proj.weight
    ("net", _ddp_keys(_Batched()), 7),  # DistributedDataParallel's `module.` prefix
    ("weights", _compiled_keys(_Batched()), 7),  # torch.compile's `_orig_mod.` prefix
    ("model", _OnlyCache().state_dict(), 1),  # named a model by its key
    ("model_state_dict", _OnlyCache().state_dict(), 1),
    ("student_model", _OnlyCache().state_dict(), 1),
    ("model_ema", _OnlyCache().state_dict(), 1),
    ("model_state", _OnlyCache().state_dict(), 1),
    ("models", _OnlyCache().state_dict(), 1),
    ("teacher_net", _OnlyCache().state_dict(), 1),
])
def test_xc9_a_model_state_dict_is_still_the_model(key, state, tensors):
    manifest = checkpoint_manifest({key: state})
    assert manifest["components"]["model"]["keys"] == [key]
    assert manifest["components"]["model"]["tensors"] == tensors


def test_xc9_a_model_in_a_nested_or_listed_entry_is_still_the_model():
    model = torch.nn.Linear(2, 2)
    nested = checkpoint_manifest({"training": {"model": model.state_dict()}})
    assert nested["components"]["model"]["keys"] == ["training.model"]
    listed = checkpoint_manifest({"models": [model.state_dict(), model.state_dict()]})
    assert listed["components"]["model"]["count"] == 2


def test_xc9_an_empty_model_entry_is_still_not_the_model():
    assert "model" not in checkpoint_manifest({"model": {}})["components"]


# --- XC9: `grads_not_read` is a note to `first_divergence` -------------------------------------


def _run(run_id: str, *, note_on: bool = True, extra: dict | None = None) -> list[dict]:
    rows = [{"schema_version": 1, "run_id": run_id, "rank": 0, "step": step,
             "scalars": {"loss": 1.0 / (step + 1)}, "grad_norm": 0.5} for step in range(40)]
    if note_on:
        rows[0]["grads_not_read"] = {"w": "the gradient is on the meta device"}
    if extra:
        rows[0].update(extra)
    return rows


def test_xc9_grads_not_read_on_both_runs_is_not_a_series_not_compared():
    result = first_divergence(_run("a"), _run("b"))
    assert result["answer"] == "no_divergence"
    assert result["left_out"]["not_compared"] == []


def test_xc9_grads_not_read_on_one_run_only_is_not_a_series_not_compared():
    result = first_divergence(_run("a"), _run("b", note_on=False))
    assert result["answer"] == "no_divergence"
    assert result["left_out"]["not_compared"] == []


def test_xc9_a_key_that_is_not_known_is_still_listed_as_not_compared():
    result = first_divergence(_run("a", extra={"mystery": [1, 2]}), _run("b"))
    assert result["left_out"]["not_compared"] == ["mystery"]
