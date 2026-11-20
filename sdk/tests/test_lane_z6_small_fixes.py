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
