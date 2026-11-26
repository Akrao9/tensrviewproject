"""The run's numerical settings in the run manifest (lane AQ-6, item OG1).

Whether healthy data-parallel replicas are bit-identical can only be read against the settings
that decide it, so the manifest records them as `tensorquill.init` found them:

- the seven settings (deterministic algorithms and whether warn-only, `cudnn.deterministic`,
  `cudnn.benchmark`, `cuda.matmul.allow_tf32`, `utils.deterministic.fill_uninitialized_memory`,
  `CUBLAS_WORKSPACE_CONFIG`, `CUDA_DEVICE_MAX_CONNECTIONS`) are read, never set; a read never
  raises, never initialises CUDA and never fails `init`, and one that cannot be made is recorded
  as not read with its reason;
- they sit under `numerical_settings` in the manifest, each under the name of what was read;
- `compare`'s `setup` shows where two runs differ in them, and a manifest written before they
  were recorded reads as "not recorded";
- a server keeps them in `runs/<run_id>/run.json` as they were sent.

The real-server test needs the binary (`TENSORQUILL_SERVER_BIN` or `server/target`) and `pyarrow`.
"""

from __future__ import annotations

import json
import os
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

import tensorquill
from tensorquill import manifest as mf

KEY = "numerical_settings"
NAMES = [
    "torch.are_deterministic_algorithms_enabled()",
    "torch.is_deterministic_algorithms_warn_only_enabled()",
    "torch.backends.cudnn.deterministic",
    "torch.backends.cudnn.benchmark",
    "torch.backends.cuda.matmul.allow_tf32",
    "torch.utils.deterministic.fill_uninitialized_memory",
    "os.environ.get('CUBLAS_WORKSPACE_CONFIG')",
    "os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')",
]
ENV = ("CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_MAX_CONNECTIONS")


@pytest.fixture(autouse=True)
def no_tracker(monkeypatch):
    for name in ("MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_NAME", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID",
                 "RANK", "LOCAL_RANK", "WORLD_SIZE", mf.PROJECT_ENV, mf.NAME_ENV, "TENSORQUILL_ENDPOINT", *ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)


def torch_values() -> dict:
    """The seven, read here directly, in the order of `NAMES`."""
    import torch.utils.deterministic  # noqa: F401  loaded by `import torch` on every supported version

    return {
        NAMES[0]: torch.are_deterministic_algorithms_enabled(),
        NAMES[1]: torch.is_deterministic_algorithms_warn_only_enabled(),
        NAMES[2]: torch.backends.cudnn.deterministic,
        NAMES[3]: torch.backends.cudnn.benchmark,
        NAMES[4]: torch.backends.cuda.matmul.allow_tf32,
        NAMES[5]: torch.utils.deterministic.fill_uninitialized_memory,
        NAMES[6]: os.environ.get(ENV[0]),
        NAMES[7]: os.environ.get(ENV[1]),
    }


def manifest(**more) -> dict:
    return mf.build_manifest(run_id="r", run_id_source="explicit", project="p", name="n", **more)


@pytest.fixture
def changed_settings(monkeypatch):
    """Every setting away from its default, put back afterwards."""
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
              torch.backends.cuda.matmul.allow_tf32, torch.utils.deterministic.fill_uninitialized_memory)
    monkeypatch.setenv(ENV[0], ":4096:8")
    monkeypatch.setenv(ENV[1], "1")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[2], before[3]
        torch.backends.cuda.matmul.allow_tf32 = before[4]
        torch.utils.deterministic.fill_uninitialized_memory = before[5]


# --- Reading them ---------------------------------------------------------------------------------


def test_the_manifest_holds_each_setting_under_the_name_of_what_was_read():
    block = manifest()[KEY]
    assert block["read_at"] == "init"
    assert list(block["settings"]) == NAMES
    assert block["settings"] == {name: {"value": value} for name, value in torch_values().items()}
    # On this PyTorch every one of them is there to read, and none is a guess.
    assert all(set(entry) == {"value"} for entry in block["settings"].values())


def test_settings_away_from_their_defaults_are_recorded_and_left_as_they_were(changed_settings):
    environment = dict(os.environ)
    before = torch_values()
    settings = manifest()[KEY]["settings"]
    assert torch_values() == before and dict(os.environ) == environment, "reading set nothing"
    assert {name: entry["value"] for name, entry in settings.items()} == {
        NAMES[0]: True, NAMES[1]: True, NAMES[2]: True, NAMES[3]: True, NAMES[4]: True, NAMES[5]: False,
        NAMES[6]: ":4096:8", NAMES[7]: "1",
    }


def test_an_environment_variable_that_is_not_set_is_none_which_is_not_the_same_as_not_read():
    settings = manifest()[KEY]["settings"]
    assert settings[NAMES[6]] == {"value": None} and settings[NAMES[7]] == {"value": None}


def test_reading_them_never_initialises_cuda(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("CUDA was initialised")

    monkeypatch.setattr(torch.cuda, "_lazy_init", refuse)
    monkeypatch.setattr(torch.cuda, "init", refuse)
    settings = manifest()[KEY]["settings"]
    assert all(set(entry) == {"value"} for entry in settings.values()), settings


class _Raises:
    """A stand-in for `torch.backends.cudnn` whose `deterministic` raises when read."""

    benchmark = False

    @property
    def deterministic(self):
        raise RuntimeError("cuDNN is not available in this build")


def _old_torch(*, warn: bool = False) -> types.ModuleType:
    """A stand-in for a PyTorch that has some of the settings and not others."""
    fake = types.ModuleType("torch")
    fake.__version__ = "1.99.0"

    def enabled():
        if warn:
            warnings.warn("are_deterministic_algorithms_enabled is deprecated", DeprecationWarning)
        return False

    def warn_only():
        raise TypeError("not in this build")

    fake.are_deterministic_algorithms_enabled = enabled
    fake.is_deterministic_algorithms_warn_only_enabled = warn_only
    fake.backends = types.SimpleNamespace(cudnn=_Raises(), cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()))
    fake.utils = types.SimpleNamespace()  # no `torch.utils.deterministic`
    return fake


def test_a_setting_this_pytorch_lacks_or_one_that_raises_is_recorded_as_not_read_with_its_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _old_torch())
    monkeypatch.setenv(ENV[0], ":16:8")
    settings = manifest()[KEY]["settings"]
    assert list(settings) == NAMES, "none is left out"
    assert settings[NAMES[0]] == {"value": False}
    assert settings[NAMES[1]] == {"not_read": "reading it raised TypeError: not in this build"}
    assert settings[NAMES[2]] == {"not_read": "reading it raised RuntimeError: cuDNN is not available in this build"}
    assert settings[NAMES[3]] == {"value": False}
    assert settings[NAMES[4]] == {"not_read": "this PyTorch (1.99.0) has no torch.backends.cuda.matmul.allow_tf32"}
    assert settings[NAMES[5]] == {"not_read": "this PyTorch (1.99.0) has no torch.utils.deterministic"}
    assert settings[NAMES[6]] == {"value": ":16:8"} and settings[NAMES[7]] == {"value": None}


def test_a_warning_while_reading_never_raises_and_is_kept_beside_the_value(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _old_torch(warn=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # as under `python -W error`
        settings = manifest()[KEY]["settings"]
    assert settings[NAMES[0]] == {"value": False,
                                  "warning": "DeprecationWarning: are_deterministic_algorithms_enabled is deprecated"}


def test_without_torch_imported_the_torch_settings_are_not_read_and_the_environment_still_is(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setenv(ENV[1], "8")
    settings = manifest()[KEY]["settings"]
    assert all(settings[name] == {"not_read": "torch is not imported"} for name in NAMES[:6])
    assert settings[NAMES[7]] == {"value": "8"}


def test_a_value_that_is_not_plain_json_is_kept_as_its_text_and_a_long_one_is_cut(monkeypatch):
    monkeypatch.setenv(ENV[0], "x" * 5000)
    settings = manifest()[KEY]["settings"]
    assert settings[NAMES[6]] == {"value": "x" * mf.SETTING_TEXT_MAX, "cut_from_length": 5000}
    fake = _old_torch()
    fake.backends.cudnn = types.SimpleNamespace(deterministic=0.5, benchmark=object)
    monkeypatch.setitem(sys.modules, "torch", fake)
    settings = manifest()[KEY]["settings"]
    assert settings[NAMES[2]] == {"value": 0.5}
    assert settings[NAMES[3]] == {"value": str(object)}


def test_a_failure_of_the_whole_read_still_gives_a_manifest(monkeypatch):
    def broken():
        raise MemoryError("no room")

    monkeypatch.setattr(mf, "_read_numerical_settings", broken)
    built = manifest()
    assert built[KEY] == {"read_at": "init", "not_read": "reading them raised MemoryError: no room"}
    assert built["project"] == "p"


def test_reading_them_takes_little_time():
    import time

    started = time.perf_counter()
    for _ in range(100):
        mf.build_manifest(run_id="r", run_id_source="explicit", project="p", name="n")
    assert (time.perf_counter() - started) / 100 < 0.005


# --- A real init ----------------------------------------------------------------------------------


def _run(model, path: Path, run_id: str, **labels) -> tensorquill.Session:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with tensorquill.init(model, run_id=run_id, routers={"router": 4}, capture="full", every=2,
                          sink=tensorquill.JsonlSink(path), **labels) as session:
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.randn(16, 8)).sum().backward()
            optimizer.step()
    return session


def test_a_real_init_writes_the_settings_into_the_manifest_beside_the_records(toy_moe, tmp_path, changed_settings):
    session = _run(toy_moe, tmp_path / "run-1.rank0.jsonl", "run-1")
    written = json.loads((tmp_path / "run-1.run.json").read_text())
    assert written[KEY]["settings"] == {name: {"value": value} for name, value in torch_values().items()}
    assert written[KEY]["settings"][NAMES[3]] == {"value": True}
    assert session.manifest[KEY] == written[KEY]


def test_a_run_started_again_records_the_settings_of_the_later_start(toy_moe, tmp_path, monkeypatch):
    path = tmp_path / "run-1.rank0.jsonl"
    _run(toy_moe, path, "run-1")
    first = json.loads((tmp_path / "run-1.run.json").read_text())
    monkeypatch.setenv(ENV[1], "1")
    _run(toy_moe, path, "run-1")
    again = json.loads((tmp_path / "run-1.run.json").read_text())
    assert again["created_at"] == first["created_at"]
    assert first[KEY]["settings"][NAMES[7]] == {"value": None} and again[KEY]["settings"][NAMES[7]] == {"value": "1"}


def test_the_copy_sent_without_its_config_keeps_the_settings():
    built = manifest(config={"vocabulary": "x" * (mf.MANIFEST_SEND_BYTES + 10)})
    sent, why = mf.manifest_to_send(built)
    assert sent["config_left_out"] is True and why
    assert sent[KEY] == built[KEY]


# --- Reading a manifest ---------------------------------------------------------------------------


def test_a_manifest_written_before_the_settings_were_recorded_reads_as_not_recorded():
    old = {"schema_version": 1, "project": "p", "name": "n", "run_id": "r"}
    assert mf.numerical_settings(old) == {"recorded": False, "why": mf.NOT_RECORDED}
    assert mf.numerical_settings(None) == {"recorded": False, "why": "the run has no manifest"}
    assert mf.numerical_settings({**old, KEY: "garbled"}) == {"recorded": False, "why": mf.NOT_RECORDED}
    new = manifest()
    assert mf.numerical_settings(new) == {"recorded": True, **new[KEY]}


# --- compare --------------------------------------------------------------------------------------


def _local(folder: Path, run_id: str, settings: dict | None, records: bool = True) -> None:
    """A local run with a hand-written manifest: `settings` is the `numerical_settings` block's
    `settings`, or None for a manifest written before they were recorded."""
    from test_mcp_server import dense, write_capture

    folder.mkdir(parents=True, exist_ok=True)
    write_capture(folder / f"{run_id}.rank0.jsonl", dense(60, run=run_id))
    document = {
        "schema_version": 1, "project": "granite", "name": run_id, "run_id": run_id, "group": None, "tags": [],
        "world_size": 1, "sdk_version": "0.0.1", "torch_version": "2.15.0", "run_id_source": "explicit",
        "config": {}, "created_at": "2026-11-26T10:00:00Z",
    }
    if settings is not None:
        document[KEY] = {"read_at": "init", "settings": settings}
    (folder / f"{run_id}.run.json").write_text(json.dumps(document))


def _settings(**changes) -> dict:
    values = {NAMES[0]: False, NAMES[1]: False, NAMES[2]: False, NAMES[3]: False, NAMES[4]: False,
              NAMES[5]: True, NAMES[6]: None, NAMES[7]: None}
    values.update({NAMES[int(k[1:])]: v for k, v in changes.items()})
    return {name: (value if isinstance(value, dict) else {"value": value}) for name, value in values.items()}


def test_compare_shows_each_numerical_setting_that_differs_between_two_runs(tmp_path):
    folder = tmp_path / "out"
    _local(folder, "run-1", _settings())
    _local(folder, "run-2", _settings(s0=True, s3=True, s6=":4096:8"))
    setup = tensorquill.compare_runs("run-1", "run-2", source=folder)["setup"]
    assert setup["differs"] == [
        {"field": f"{KEY}.{NAMES[0]}", "values": {"A": False, "B": True}},
        {"field": f"{KEY}.{NAMES[3]}", "values": {"A": False, "B": True}},
        {"field": f"{KEY}.{NAMES[6]}", "values": {"A": None, "B": ":4096:8"}},
    ]
    assert setup["numerical_settings_same"] == [NAMES[i] for i in (1, 2, 4, 5, 7)]
    assert setup["same"] == ["project", "group", "world_size", "sdk_version", "torch_version"]
    assert "numerical_settings_not_recorded" not in setup
    summary = tensorquill.compare_runs("run-1", "run-2", source=folder)["summary"]
    assert f"{KEY}.{NAMES[3]}: A false, B true" in summary
    assert f"{KEY}.{NAMES[6]}: A not set, B ':4096:8'" in summary


def test_compare_shows_a_setting_read_in_one_run_and_not_read_in_the_other(tmp_path):
    folder = tmp_path / "out"
    _local(folder, "run-1", _settings())
    _local(folder, "run-2", _settings(s5={"not_read": "this PyTorch (2.0.0) has no torch.utils.deterministic"}))
    setup = tensorquill.compare_runs("run-1", "run-2", source=folder)["setup"]
    assert setup["differs"] == [{
        "field": f"{KEY}.{NAMES[5]}", "values": {"A": True, "B": None},
        "not_read_in": {"B": "this PyTorch (2.0.0) has no torch.utils.deterministic"},
    }]
    summary = tensorquill.compare_runs("run-1", "run-2", source=folder)["summary"]
    assert "B not read (this PyTorch (2.0.0) has no torch.utils.deterministic)" in summary


def test_identical_settings_give_no_difference_and_an_older_manifest_reads_as_not_recorded(tmp_path):
    folder = tmp_path / "out"
    _local(folder, "run-1", _settings(s3=True))
    _local(folder, "run-2", _settings(s3=True))
    _local(folder, "old-run", None)
    result = tensorquill.compare_runs("run-1", "run-2", "old-run", source=folder)
    setup = result["setup"]
    assert setup["differs"] == [] and setup["numerical_settings_same"] == NAMES
    assert setup["numerical_settings_not_recorded"] == ["C"]
    assert "numerical settings not recorded for C" in result["summary"]
    # Two older manifests: nothing to compare, both said, never "the same".
    alone = tensorquill.compare_runs("run-1", "old-run", source=folder)["setup"]
    assert alone["differs"] == [] and alone["numerical_settings_same"] == []
    assert alone["numerical_settings_not_recorded"] == ["B"]


def test_two_real_runs_with_different_settings_differ_in_compare(toy_moe, tmp_path, monkeypatch):
    _run(toy_moe, tmp_path / "run-1.rank0.jsonl", "run-1")
    torch.backends.cudnn.benchmark, before = True, torch.backends.cudnn.benchmark
    monkeypatch.setenv(ENV[1], "1")
    try:
        _run(toy_moe, tmp_path / "run-2.rank0.jsonl", "run-2")
    finally:
        torch.backends.cudnn.benchmark = before
    differs = tensorquill.compare_runs("run-1", "run-2", source=tmp_path)["setup"]["differs"]
    assert [item["field"] for item in differs] == [f"{KEY}.{NAMES[3]}", f"{KEY}.{NAMES[7]}"]


# --- The server -----------------------------------------------------------------------------------


def test_a_streamed_run_ends_with_its_settings_in_the_servers_run_json_unchanged(toy_moe, tmp_path, monkeypatch,
                                                                                changed_settings):
    from test_run_manifest import real_server, streamed

    store = tmp_path / "store"
    server = real_server(store)
    try:
        streamed(toy_moe, tmp_path, monkeypatch, server.endpoint, "run-1", project="granite", name="baseline")
    finally:
        assert server.terminate() == 0
    local = json.loads((tmp_path / "local" / "run-1.run.json").read_text())
    stored = json.loads((store / "runs" / "run-1" / "run.json").read_text())
    assert stored[KEY] == local[KEY]
    assert stored[KEY]["settings"][NAMES[6]] == {"value": ":4096:8"}
    assert stored[KEY]["settings"][NAMES[0]] == {"value": True}
    # Read back from the store, compare finds them.
    assert mf.numerical_settings(stored)["recorded"] is True
