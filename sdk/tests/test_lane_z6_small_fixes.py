"""Lane Z6 (wave SDK-22): the small fixes of OPEN_BUGS L23 in `sink.py`, `manifest.py` and `data.py`.

- The end marker (`sink.write_end_marker`): a marker that cannot be written is said without
  ever raising, under `python -W error` too (logged on the `tensorquill` logger when the
  warning cannot be), and the scratch file a failed rename left is removed.
- The run manifest (`manifest.write_manifest`): the scratch file a failed write or rename left
  (`.<name>.<pid>.tmp`) is removed, a removal that fails is let be, and the manifest already
  there is left as it was.
- The model-state rule (`data._model_state`): what it gets wrong today, and the one part closed.

Each test that fails on the base (89bef4e) says so in a "Fail before" line; the others pin
behaviour that held already and must keep holding.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import pytest
import torch

import tensorquill
from tensorquill import manifest as mf
from tensorquill import sink as sk
from tensorquill.data import checkpoint_manifest
from tensorquill.sink import END_SUFFIX, end_marker_path, read_end_marker, write_end_marker


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


def said(caplog, text: str) -> bool:
    return any(text in r.getMessage() for r in caplog.records if r.name == "tensorquill")


def scratch_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(".tmp"))


def block_end_marker(records: Path) -> None:
    """Make the marker's place a non-empty folder, so its rename fails for real (no patching)."""
    place = end_marker_path(records)
    place.mkdir(parents=True)
    (place / "keep").write_text("x")


def refuse_replace(monkeypatch, suffix: str) -> None:
    """`os.replace` fails for targets ending in `suffix` only; `os` is one module, so the patch
    is the same whichever module's `os` it is set on."""
    replace = os.replace

    def refuse(source, target, *args, **kwargs):
        if str(target).endswith(suffix):
            raise PermissionError(13, "refused by the test", str(target))
        return replace(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", refuse)


def short_run(path: Path, run_id: str = "r"):
    """A session over a short run, left open for the caller to close."""
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tensorquill.init(model, run_id=run_id, every=2, heartbeat_seconds=None, sink=tensorquill.JsonlSink(path))
    for _ in range(4):
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
    return session


# --- 1. The end marker: said, never raised --------------------------------------------------


def test_z6_an_end_marker_that_cannot_be_written_is_logged_not_raised_under_an_error_filter(tmp_path, caplog):
    """Fail before: `write_end_marker` raised the RuntimeWarning out of its own `except` under
    `simplefilter("error")`, though it says it never raises."""
    caplog.set_level(logging.WARNING, logger="tensorquill")
    records = tmp_path / "r.rank0.jsonl"
    records.write_text("{}\n")
    block_end_marker(records)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert write_end_marker(records, run_id="r", rank=0, attempt=1) is False
    assert said(caplog, "could not write the end marker")
    assert said(caplog, "the run reads as not closed")


def test_z6_a_session_whose_end_marker_fails_says_so_under_an_error_filter(tmp_path, caplog):
    """Fail before: `Session.close` caught the raised warning (`except Exception: pass`), so the
    message was lost: nothing on the logger, nothing on stderr. Close itself never raised."""
    caplog.set_level(logging.WARNING, logger="tensorquill")
    records = tmp_path / "r.rank0.jsonl"
    session = short_run(records)
    block_end_marker(records)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session.close()
    assert said(caplog, "could not write the end marker")
    assert read_end_marker(records)["state"] == "unreadable"  # a folder is no marker: not known to have closed
    assert [json.loads(line)["step"] for line in records.read_text().splitlines()]  # the run recorded


def test_z6_an_end_marker_failure_is_still_a_runtime_warning_at_the_caller_of_close(tmp_path, caplog):
    """Held before too: under the default filter it is the same RuntimeWarning, with the same
    text, pointing at the line that called `close`, and nothing is logged."""
    caplog.set_level(logging.WARNING, logger="tensorquill")
    records = tmp_path / "r.rank0.jsonl"
    session = short_run(records)
    block_end_marker(records)
    with pytest.warns(RuntimeWarning, match="could not write the end marker .* the run reads as not closed") as seen:
        session.close()
    found = [w for w in seen if "end marker" in str(w.message)]
    assert len(found) == 1 and found[0].filename == __file__
    assert not said(caplog, "could not write the end marker")


def test_z6_a_filter_that_ignores_the_end_marker_warning_still_silences_it(tmp_path, caplog):
    """Held before too: an "ignore" filter is still obeyed; the logger is the fallback for a
    warning that raises, not a second channel."""
    caplog.set_level(logging.WARNING, logger="tensorquill")
    records = tmp_path / "r.rank0.jsonl"
    records.write_text("{}\n")
    block_end_marker(records)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("ignore")
        assert write_end_marker(records, run_id="r", rank=0, attempt=1) is False
    assert not seen and not said(caplog, "could not write the end marker")


def test_z6_a_failed_end_marker_rename_leaves_no_scratch_file(tmp_path, monkeypatch):
    """Fail before: `.r.rank0.end.json.<pid>.tmp` stayed beside the records file. (Not named in
    L23, the same leak as the manifest's in a file of this lane.)"""
    records = tmp_path / "r.rank0.jsonl"
    records.write_text("{}\n")
    refuse_replace(monkeypatch, END_SUFFIX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert write_end_marker(records, run_id="r", rank=0, attempt=1) is False
    assert scratch_files(tmp_path) == []
    assert not end_marker_path(records).exists()


def test_z6_a_scratch_file_that_cannot_be_removed_does_not_make_the_end_marker_raise(tmp_path, monkeypatch, caplog):
    """Fail before: under the error filter the warning raised (item 1). Now a removal that fails
    is let be too, and the failure is still said."""
    caplog.set_level(logging.WARNING, logger="tensorquill")
    records = tmp_path / "r.rank0.jsonl"
    records.write_text("{}\n")
    refuse_replace(monkeypatch, END_SUFFIX)
    unlink = Path.unlink

    def stuck(self, *args, **kwargs):
        if self.name.endswith(".tmp"):
            raise PermissionError(13, "cannot remove", str(self))
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", stuck)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert write_end_marker(records, run_id="r", rank=0, attempt=1) is False
    assert said(caplog, "could not write the end marker")


def test_z6_a_marker_that_is_written_leaves_no_scratch_file(tmp_path):
    """Held before too: the ordinary path, a marker by rename and nothing else."""
    records = tmp_path / "r.rank0.jsonl"
    records.write_text("{}\n")
    assert write_end_marker(records, run_id="r", rank=0, attempt=1) is True
    assert read_end_marker(records)["state"] == "closed"
    assert scratch_files(tmp_path) == []


def test_z6_sink_says_by_the_manifest_helper():
    """Fail before: `sink.py` had no `_say`. It uses `manifest._say` (that module imports nothing
    of the package), not a bare warning."""
    assert sk._say is mf._say


# --- 2. The manifest: no scratch file left, the old manifest kept ---------------------------


def _manifest(run_id: str = "r", name: str = "first") -> dict:
    return {"schema_version": 1, "run_id": run_id, "project": "p", "name": name, "created_at": "2026-11-20T00:00:00Z"}


def test_z6_a_failed_manifest_rename_leaves_no_scratch_file_and_keeps_the_old_manifest(tmp_path, monkeypatch):
    """Fail before: `.r.run.json.<pid>.tmp` stayed beside the manifest after the rename failed."""
    path = tmp_path / "r.run.json"
    assert mf.write_manifest(path, _manifest()) is True
    before = path.read_bytes()
    refuse_replace(monkeypatch, mf.MANIFEST_SUFFIX)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mf.write_manifest(path, _manifest(name="second")) is False
    assert scratch_files(tmp_path) == []
    assert path.read_bytes() == before


def test_z6_a_failed_scratch_write_leaves_no_scratch_file(tmp_path, monkeypatch):
    """Fail before: a write that failed part way (here: after the file was created, as a full
    disk does) left the partial scratch file."""
    path = tmp_path / "r.run.json"
    write_text = Path.write_text

    def partial(self, data, *args, **kwargs):
        if self.name.endswith(".tmp"):
            write_text(self, data[:10], *args, **kwargs)
            raise OSError(28, "No space left on device", str(self))
        return write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", partial)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mf.write_manifest(path, _manifest()) is False
    assert scratch_files(tmp_path) == []
    assert not path.exists()


def test_z6_a_manifest_scratch_file_that_cannot_be_removed_is_let_be(tmp_path, monkeypatch, caplog):
    """Held before too (there was no removal): the removal's own failure never raises, under an
    error filter too, and the failure is still said."""
    caplog.set_level(logging.WARNING, logger="tensorquill")
    path = tmp_path / "r.run.json"
    refuse_replace(monkeypatch, mf.MANIFEST_SUFFIX)
    unlink = Path.unlink

    def stuck(self, *args, **kwargs):
        if self.name.endswith(".tmp"):
            raise PermissionError(13, "cannot remove", str(self))
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", stuck)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mf.write_manifest(path, _manifest()) is False
    assert said(caplog, "could not write the run manifest")


def test_z6_a_manifest_whose_folder_cannot_be_made_does_not_raise(tmp_path):
    """Held before too: a failure before the scratch file is named (the folder is a file) is said
    and returns False; the removal has nothing to remove."""
    (tmp_path / "blocked").write_text("a file, not a folder")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mf.write_manifest(tmp_path / "blocked" / "r.run.json", _manifest()) is False


def test_z6_init_with_a_refused_manifest_rename_leaves_no_scratch_file(tmp_path, monkeypatch):
    """Fail before: the run trained (lane Y6) but left `.r.run.json.<pid>.tmp` in its folder."""
    refuse_replace(monkeypatch, mf.MANIFEST_SUFFIX)
    records = tmp_path / "r.rank0.jsonl"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        short_run(records).close()
    assert scratch_files(tmp_path) == []
    assert not (tmp_path / "r.run.json").exists()
    assert records.read_text()


# --- 3. The model-state rule ----------------------------------------------------------------
#
# The rule (`data._model_state`): a mapping of tensors is the model when its entry's name says
# so or one of its keys is a parameter path. Two misses in L23. The Rust twin
# (`server/src/detect/data.rs`) holds no copy of this rule: it reads the `components` the SDK
# wrote into the record, so a change here changes both engines' verdicts alike.

T = torch.zeros(2)


def test_z6_known_limit_a_custom_model_with_only_unusual_names_under_a_plain_key_is_not_the_model():
    """KNOWN LIMIT, not closed (today's behaviour, pinned). A model whose only parameters are
    registered at its top level under names no PyTorch module uses (`cls_token`, `pos_embed`),
    saved under a key that names no model (`generator`), is a mapping of tensors with nothing
    that tells it from a cache: it goes to `other`, and a resume from it reads the model as
    missing. Closing it from the dict alone would flip the pinned "a cache is not the model"
    cases (`state`, `dict`, `cache`); the way to close it is to compare the entry's keys with
    the live model's (`capture.py`, not this lane's file)."""

    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 4))
            self.pos_embed = torch.nn.Parameter(torch.zeros(8, 4))

    for key in ("generator", "g", "backbone"):
        manifest = checkpoint_manifest({key: Generator().state_dict()})
        assert "model" not in manifest["components"], key
        assert manifest["other"] == [f"{key}.cls_token", f"{key}.pos_embed"]


@pytest.mark.parametrize("key, entries", [
    ("cache", {"layers.0.k": T, "layers.0.v": T}),  # dotted
    ("kv_cache", {"attn_weight": T}),  # a `_weight` suffix
    ("embedding_cache", {"emb_weight": T, "emb_bias": T}),
    ("cache", {"weight": T}),  # a PyTorch parameter name
    ("model_cache", {"blocks.0.weight": T}),  # the last word is `cache`
    ("cache_state_dict", {"layers.0.k": T}),  # `state_dict` set aside, as for a model's name
    ("caches", {"weight_ih_l0": T}),
])
def test_z6_a_cache_with_parameter_like_keys_is_not_the_model(key, entries):
    """Fail before: any dotted or `*_weight` key made a cache the model, though its name said it
    was a cache. Now an entry whose name's last word is `cache`/`caches` (once a trailing
    `state`/`states`/`dict` is set aside) is never the model, whatever its keys."""
    manifest = checkpoint_manifest({key: entries, "optimizer": {"state": {}, "param_groups": [{}]}})
    assert "model" not in manifest["components"]
    assert manifest["other"] == [f"{key}.{k}" for k in entries]


def test_z6_a_cache_beside_the_model_is_not_listed_with_it():
    """Fail before: the cache was listed as a second model key (`keys == ["model", "kv_cache"]`)."""
    manifest = checkpoint_manifest({"model": torch.nn.Linear(2, 2).state_dict(),
                                    "kv_cache": {"layers.0.k": T, "layers.0.v": T}})
    assert manifest["components"]["model"]["keys"] == ["model"]
    assert manifest["components"]["model"]["tensors"] == 2


def test_z6_a_resume_from_a_checkpoint_holding_only_a_dotted_cache_finds_the_model_missing():
    """Fail before: the cache stood in for the model, so a resume that restored no model said
    nothing about it."""
    from tensorquill import data_checks

    first = {"checkpoint": {"load": checkpoint_manifest({
        "kv_cache": {"layers.0.k": T, "layers.0.v": T},
        "optimizer": {"state": {0: {"step": 3}}, "param_groups": [{}]},
    })}}
    findings, _, checked = data_checks._state_findings(first, 3, 3, None, False, None, None)
    # Wave AB (STEP_AND_EVIDENCE_CONTRACT.md, section 4): not found is `not_identified`, not `missing`.
    assert checked and {"component": "model", "finding": "not_identified"} in findings


@pytest.mark.parametrize("key, entries", [
    ("buffers", {"layers.0.k": T}),
    ("stats", {"loss_weight": T}),
    ("history", {"grad_norm.mean": T}),
    ("model", {"cached_embedding": T}),  # stored under a model's name
])
def test_z6_known_limit_a_cache_under_a_name_that_is_not_cache_is_still_the_model(key, entries):
    """KNOWN LIMIT, not closed (today's behaviour, pinned). Only a name that says `cache` vetoes
    the key evidence; a mapping of non-parameter tensors under any other name with a dotted or
    `*_weight` key, or under a model's name, is still the model. Narrowing the dotted rule (a
    dotted key counting only when its last part is a parameter name) was not tried: it would
    close `layers.0.k` but miss a model whose every key is a custom name in a submodule
    (`blocks.0.scale`). `loss_weight` cannot be told from a custom `pos_weight` parameter by
    its name. (Real `torch.nn` modules have other keys that qualify, so neither change would
    flip the module sweep below; that is not measured.)"""
    assert checkpoint_manifest({key: entries})["components"]["model"]["keys"] == [key]


def _modules() -> list[tuple[str, torch.nn.Module]]:
    nn = torch.nn
    return [
        ("Linear", nn.Linear(2, 2)), ("Bilinear", nn.Bilinear(2, 2, 2)), ("Embedding", nn.Embedding(4, 2)),
        ("EmbeddingBag", nn.EmbeddingBag(4, 2)), ("Conv2d", nn.Conv2d(1, 1, 1)), ("LayerNorm", nn.LayerNorm(2)),
        ("GroupNorm", nn.GroupNorm(1, 2)), ("BatchNorm1d", nn.BatchNorm1d(2)), ("PReLU", nn.PReLU()),
        ("LSTM", nn.LSTM(2, 2)), ("GRU", nn.GRU(2, 2, bidirectional=True)), ("RNN", nn.RNN(2, 2)),
        ("MultiheadAttention", nn.MultiheadAttention(4, 2)),
        ("MultiheadAttention-kdim", nn.MultiheadAttention(4, 2, kdim=3, vdim=3)),
        ("TransformerEncoderLayer", nn.TransformerEncoderLayer(4, 2, dim_feedforward=8)),
        ("Sequential", nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))),
    ]


@pytest.mark.parametrize("label, module", _modules(), ids=[label for label, _ in _modules()])
@pytest.mark.parametrize("key", ["generator", "teacher", "encoder", "model", "state"])
def test_z6_a_real_modules_state_dict_is_the_model_under_any_name_but_a_cache(label, module, key):
    """Held before too: the cache veto changes nothing for an entry not named a cache. Every
    module here is the model under a plain key by its keys alone."""
    state = module.state_dict()
    manifest = checkpoint_manifest({key: state})
    assert manifest["components"]["model"]["keys"] == [key], label
    assert manifest["components"]["model"]["tensors"] == len(state)


@pytest.mark.parametrize("label, module", _modules(), ids=[label for label, _ in _modules()])
def test_z6_a_real_modules_state_dict_under_a_cache_name_is_not_the_model(label, module):
    """Fail before (each module): a model's state dict saved under a name whose last word is
    `cache` was the model. This is the price of the veto, a call for the author: a checkpoint
    that keeps its model under `model_cache` now reads as missing its model."""
    assert "model" not in checkpoint_manifest({"model_cache": module.state_dict()})["components"], label
