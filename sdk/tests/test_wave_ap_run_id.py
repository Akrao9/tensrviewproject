"""Wave AP, lane AP-3 (audit finding AO-04): `tensorquill.init` and a run ID that is not a string.

A `Path` (any `os.PathLike` whose path is text) is taken by its text, and from then on is that
string everywhere: judged, written to the manifest and the records, shared, streamed. Anything else
that is not None or a string (`123`, `0`, `False`, `1.5`, `b"x"`, `[]`) is refused at `init` before
anything is opened, as a non-string `group`, `project` or `name` is.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
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


_TIMES = ("ts", "attempt", "closed_at", "created_at")  # the time fields of a record, the end marker and the manifest


def _untimed(value):
    if isinstance(value, dict):
        return {k: _untimed(v) for k, v in value.items() if k not in _TIMES}
    if isinstance(value, list):
        return [_untimed(v) for v in value]
    return value


def test_a_path_run_id_and_the_equal_string_give_the_same_capture(tmp_path, monkeypatch):
    captured = {}
    for kind, run_id in (("text", "same"), ("path", Path("same"))):
        work = tmp_path / kind
        work.mkdir()
        _work(work, monkeypatch)
        session = _capture(run_id)
        out = work / "work" / "tensorquill-out"
        captured[kind] = (
            _files(out),
            _untimed(json.loads((out / "same.run.json").read_text())),
            _untimed(_records(out / "same.rank0.jsonl")),
            _untimed(json.loads((out / "same.rank0.end.json").read_text())),
            (session.run.run_id, type(session.run.run_id), session.run.source),
        )
    assert captured["path"] == captured["text"]


def test_with_paths_turned_off_a_path_is_refused_like_an_integer(tmp_path, monkeypatch):
    work = _work(tmp_path, monkeypatch)
    monkeypatch.setattr(_manifest, "RUN_ID_TAKES_PATHS", False)
    with pytest.raises(ValueError, match=r"^run_id must be a string, not PosixPath$|^run_id must be a string, not WindowsPath$"):
        tensorquill.init(nn.Linear(2, 2), run_id=Path("myrun"), project="p", heartbeat_seconds=None)
    with pytest.raises(ValueError, match=r"^run_id must be a string, not int$"):
        tensorquill.init(nn.Linear(2, 2), run_id=123, project="p", heartbeat_seconds=None)
    assert not (work / "tensorquill-out").exists()


class _BadPath:
    def __init__(self, gives):
        self.gives = gives

    def __fspath__(self):
        if isinstance(self.gives, BaseException):
            raise self.gives
        return self.gives


@pytest.mark.parametrize("gives", [b"bytes", 7, RuntimeError("no path")])
def test_a_path_whose_text_cannot_be_read_is_refused_with_value_error(tmp_path, monkeypatch, gives):
    work = _work(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"^run_id must be a string or a path"):
        tensorquill.init(nn.Linear(2, 2), run_id=_BadPath(gives), project="p", heartbeat_seconds=None)
    assert not (work / "tensorquill-out").exists()


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


# ---- a second rank: the ID rank 0 shares is the string ------------------------------------------

_SCRUBBED = (
    "MLFLOW_RUN_ID", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "RANK", "LOCAL_RANK", "WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "TENSORQUILL_ENDPOINT", "TENSORQUILL_PROJECT", "TENSORQUILL_RUN_NAME",
    "MLFLOW_EXPERIMENT_NAME",
)

_RANK_SCRIPT = """
import datetime, json, sys, warnings
from pathlib import Path
import torch.distributed as dist
from torch import nn

rank, world, store_path, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
dist.init_process_group("gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=15))
import tensorquill
with warnings.catch_warnings(record=True) as said:
    warnings.simplefilter("always")
    # Rank 0 alone is given the run ID, as a Path; the other rank takes rank 0's.
    session = tensorquill.init(nn.Linear(2, 2), run_id=Path("shared") if rank == 0 else None, project="p",
                               auto_step=False, mirror_scalars=False, heartbeat_seconds=None)
    session.log(loss=1.0)
    session.step()
    session.close()
records = [json.loads(line) for line in Path(session._sink.path).read_text().splitlines() if line.strip()]
found = {"run_id": session.run.run_id, "type": type(session.run.run_id).__name__, "source": session.run.source,
         "record_ids": sorted({repr(r.get("run_id")) for r in records}), "error": repr(session.writer_error),
         "warnings": [str(w.message) for w in said if "run ID" in str(w.message)]}
with open(out, "w") as file:
    json.dump(found, file)
dist.destroy_process_group()
"""


def test_on_a_second_rank_the_shared_id_is_the_string(tmp_path):
    script = tmp_path / "rank.py"
    script.write_text(_RANK_SCRIPT)
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED}
    env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    source = str(Path(tensorquill.__file__).resolve().parents[1])  # the tensorquill this test imported
    env["PYTHONPATH"] = os.pathsep.join([source, *filter(None, [os.environ.get("PYTHONPATH")])])
    work = tmp_path / "work"
    work.mkdir()
    procs = [subprocess.Popen([sys.executable, str(script), str(rank), "2", str(tmp_path / "store"),
                               str(tmp_path / f"rank{rank}.json")],
                              cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for rank in range(2)]
    logs = []
    try:
        for proc in procs:
            logs.append(proc.communicate(timeout=90)[0])
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
            proc.communicate()
        pytest.fail("a rank did not finish within 90 s")
    found = []
    for rank in range(2):
        result = tmp_path / f"rank{rank}.json"
        assert result.exists(), f"rank {rank} wrote nothing:\n{logs[rank][-3000:]}"
        found.append(json.loads(result.read_text()))
    for rank, one in enumerate(found):
        assert (one["run_id"], one["type"], one["source"]) == ("shared", "str", "explicit"), (rank, one)
        assert one["record_ids"] == ["'shared'"], (rank, one)
        assert one["error"] == "None", (rank, one)
        assert one["warnings"] == [], (rank, one)
    out = work / "tensorquill-out"
    assert sorted(p.name for p in out.iterdir()) == [
        "shared.rank0.end.json", "shared.rank0.jsonl", "shared.rank1.end.json", "shared.rank1.jsonl",
        "shared.run.json"]


# ---- streamed: a Path run ID reaches the store under its text -----------------------------------

def _server_binary() -> str | None:
    from tensorquill.server_detect import find_binary
    found = find_binary()
    return found if found and os.path.isfile(found) else None


def test_a_streamed_capture_with_a_path_run_id_reaches_the_store_under_its_text(tmp_path, monkeypatch):
    try:
        import pyarrow.flight  # noqa: F401
        from tensorquill import flight
        from tensorquill.flight import offset_path, read_offset
    except ImportError:
        pytest.skip("pyarrow.flight is not installed")
    binary = _server_binary()
    if binary is None:
        pytest.skip("tensorquill-server is not built")
    work = _work(tmp_path, monkeypatch)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    store = tmp_path / "store"
    env = {k: v for k, v in os.environ.items() if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_ENDPOINT")}
    server = subprocess.Popen([binary, "--listen", "127.0.0.1:0", "--store", str(store), "--flush-steps", "1"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = server.stdout.readline()
        assert "listening on" in line, line
        endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"
        session = tensorquill.init(nn.Linear(2, 2), run_id=Path("streamed"), project="p", auto_step=False,
                                   mirror_scalars=False, heartbeat_seconds=None, endpoint=endpoint, every=1)
        records_file = work / "tensorquill-out" / "streamed.rank0.jsonl"
        for _ in range(3):
            session.log(loss=1.0)
            session.step()
        # The session's own stream must have stored a part before close(), whose wait is 5 s at most.
        deadline = time.monotonic() + 60
        while read_offset(offset_path(records_file)) <= 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert read_offset(offset_path(records_file)) > 0, "the stream stored nothing within 60 s"
        session.close()
        assert session.writer_error is None
        # What close() had no time for under load: the rest, from the offset the stream left.
        if read_offset(offset_path(records_file)) < records_file.stat().st_size:
            assert flight.send_file(records_file, endpoint, "streamed", 0)
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(30)
        err = server.stderr.read()
        server.stdout.close()
        server.stderr.close()
    assert sorted(p.name for p in (store / "runs").iterdir()) == ["streamed"], err
    stored = tensorquill.load_records(store / "runs" / "streamed")
    assert stored, err
    assert {r["run_id"] for r in stored} == {"streamed"} and all(type(r["run_id"]) is str for r in stored)
    assert stored == _records(work / "tensorquill-out" / "streamed.rank0.jsonl")
