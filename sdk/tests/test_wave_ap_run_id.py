"""Wave AP, lane AP-3 (audit finding AO-04): `tensorquill.init` and a run ID that is not a string.

A `Path` (any `os.PathLike` whose path is text) is taken by its text, and from then on is that
string everywhere: judged, written to the manifest and the records, shared, streamed. Anything else
that is not None or a string (`123`, `0`, `False`, `1.5`, `b"x"`, `[]`) is refused at `init` before
anything is opened, as a non-string `group`, `project` or `name` is.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import torch.nn as nn

import tensorquill
from tensorquill import manifest as _manifest


def _work(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    for name in ("MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "TENSORQUILL_PROJECT",
                 "TENSORQUILL_ENDPOINT", "TENSORQUILL_RUN_NAME"):
        monkeypatch.delenv(name, raising=False)
    return work


def _capture(run_id):
    session = tensorquill.init(nn.Linear(2, 2), run_id=run_id, project="p", auto_step=False,
                               mirror_scalars=False, heartbeat_seconds=None)
    session.log(loss=1.0)
    session.step()
    session.close()
    return session


def _files(out: Path) -> list[str]:
    return sorted(p.name for p in out.iterdir())


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_path_run_id_gives_the_same_files_manifest_and_records_as_its_text(tmp_path, monkeypatch):
    work = _work(tmp_path, monkeypatch)
    session = _capture(Path("myrun"))
    assert session.writer_error is None
    assert type(session.run.run_id) is str and session.run.run_id == "myrun"
    out = work / "tensorquill-out"
    assert _files(out) == ["myrun.rank0.end.json", "myrun.rank0.jsonl", "myrun.run.json"]
    records = _records(out / "myrun.rank0.jsonl")
    assert records and all(type(r["run_id"]) is str and r["run_id"] == "myrun" for r in records)
    found = json.loads((out / "myrun.run.json").read_text())
    assert type(found["run_id"]) is str and found["run_id"] == "myrun"
    assert found["run_id_source"] == "explicit"
    assert session.manifest["run_id"] == "myrun"


@pytest.mark.parametrize("text", ["a/b", "../x", "..", ".", "/abs"])
def test_a_path_run_id_is_refused_like_its_text(tmp_path, monkeypatch, text):
    work = _work(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"^run_id .* cannot be used: "):
        tensorquill.init(nn.Linear(2, 2), run_id=text, project="p", heartbeat_seconds=None)
    with pytest.raises(ValueError, match=r"^run_id .* cannot be used: "):
        tensorquill.init(nn.Linear(2, 2), run_id=Path(text), project="p", heartbeat_seconds=None)
    assert not (work / "tensorquill-out").exists()


def test_a_path_with_a_leading_dot_is_taken_like_its_text(tmp_path, monkeypatch):
    # `run_id_problem` refuses only "." and ".." and the separators: a leading dot is accepted for both.
    work = _work(tmp_path, monkeypatch)
    session = _capture(Path(".hidden"))
    assert session.run.run_id == ".hidden" and type(session.run.run_id) is str
    assert ".hidden.run.json" in _files(work / "tensorquill-out")


@pytest.mark.parametrize("run_id", [123, 0, False, True, 1.5, b"x", [], ["a"]])
def test_a_run_id_that_is_neither_a_string_nor_a_path_is_refused_before_anything_is_opened(
        tmp_path, monkeypatch, run_id):
    work = _work(tmp_path, monkeypatch)
    before = set(threading.enumerate())
    with pytest.raises(ValueError, match=rf"^run_id must be a string or a path, not {type(run_id).__name__}$"):
        tensorquill.init(nn.Linear(2, 2), run_id=run_id, project="p", heartbeat_seconds=None)
    assert not (work / "tensorquill-out").exists()
    assert set(threading.enumerate()) <= before


@pytest.mark.parametrize("run_id", [None, ""])
def test_none_and_the_empty_string_are_not_given(tmp_path, monkeypatch, run_id):
    _work(tmp_path, monkeypatch)
    session = _capture(run_id)
    assert session.run.source == "generated" and session.run.run_id.startswith("run-")


def test_check_run_id_returns_the_text():
    assert _manifest.check_run_id(Path("x")) == "x" and type(_manifest.check_run_id(Path("x"))) is str
    assert _manifest.check_run_id("y") == "y"
    assert _manifest.check_run_id(None) is None


@pytest.mark.parametrize("shared, text", [(123, "123"), (1.5, "1.5"), (Path("p"), "p")])
def test_a_shared_or_tracker_run_id_that_is_not_a_string_is_fitted_to_its_text(shared, text):
    # An older rank 0 given run_id=123 shares the integer through the store; a fake tracker may give anything.
    with pytest.warns(RuntimeWarning, match="not a string"):
        fitted = _manifest.fit_run_id(shared, "explicit")
    assert fitted == text and type(fitted) is str
