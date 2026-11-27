"""One records file per writer: an advisory lock on `<run_id>.rank<N>.jsonl`, held for the session.

The records file is opened for appending on purpose (a run started again under its run ID appends
to it), so it cannot be created exclusively. `init`'s default sink takes an advisory lock on it
without blocking. A second process (or a second session in this one) that cannot take it writes
`<run_id>.rank<N>.<pid>.jsonl` and says so once. A lock that fails for another reason (a file
system without locking) is "no lock here": the session writes the file unlocked and says so once.
`init` never blocks and never raises for any of this.

The readers read `<run_id>.rank<N>.<pid>.jsonl` as rank N's second writer, never as a restart of
it: the listing's row is `several_writers`, with no run state, last step or incident; `compare`
refuses the rank; the evidence of either file, by `build_evidence` or the MCP server, has a
`writers` block naming both.

Real processes, for the cases that need two: each child has a time limit and is killed if it is
still running.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import sink as _sink

LIMIT = 120.0

CHILD = textwrap.dedent(
    """
    import json, os, sys, time, warnings
    from pathlib import Path
    import torch
    from torch import nn
    import tensorquill

    run_id, steps, ready, go = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    limit = float(sys.argv[5])
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id=run_id, every=1, heartbeat_seconds=None, stack_seconds=None)
    ready.write_text(str(session._sink.path))
    deadline = time.monotonic() + limit
    while not go.exists():
        if time.monotonic() > deadline:
            sys.exit("never told to go")
        time.sleep(0.01)
    for _ in range(steps):
        model(torch.randn(3, 2)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    session.close()
    print(json.dumps({"pid": os.getpid(), "path": str(session._sink.path), "attempt": session.attempt,
                      "said": [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()]}))
    """
)


def _child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "PYTHONWARNINGS")
           and not k.startswith(("SLURM_", "OMPI_", "PMI_", "TORCHELASTIC_", "TENSORQUILL_", "MLFLOW_", "WANDB_"))}
    src = str(Path(tensorquill.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


class Child:
    def __init__(self, work: Path, name: str, run_id: str, steps: int) -> None:
        self.ready, self.go = work.parent / f"{work.name}-{name}-ready", work.parent / f"{work.name}-{name}-go"
        script = work.parent / f"{work.name}-child.py"
        script.write_text(CHILD)
        self.process = subprocess.Popen(
            [sys.executable, str(script), run_id, str(steps), str(self.ready), str(self.go), str(LIMIT)],
            cwd=work, env=_child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_ready(self) -> str:
        deadline = time.monotonic() + LIMIT
        while not self.ready.exists() or not self.ready.read_text():
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.kill()
                raise AssertionError(f"the child never got ready: {self.process.communicate()[1][-2000:]}")
            time.sleep(0.01)
        return self.ready.read_text()

    def finish(self) -> dict:
        self.go.write_text("")
        try:
            out, err = self.process.communicate(timeout=LIMIT)
        except subprocess.TimeoutExpired:
            self.kill()
            raise
        assert self.process.returncode == 0, err[-2000:]
        return json.loads(out.strip().splitlines()[-1])

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)
            self.process.wait()


def _init(run_id: str, model=None, **options):
    model = nn.Linear(2, 1) if model is None else model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, run_id=run_id, every=1, heartbeat_seconds=None, stack_seconds=None,
                                   **options)
    return session, [str(w.message) for w in caught if "tensorquill" in str(w.message).lower()], model


def _train(model, steps: int) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(steps):
        model(torch.randn(3, 2)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()


def _steps(path: Path) -> list[int]:
    return [json.loads(line)["step"] for line in path.read_text().splitlines() if '"step"' in line]


@pytest.fixture
def here(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") or name.startswith(
                ("SLURM_", "OMPI_", "PMI_", "TORCHELASTIC_", "TENSORQUILL_", "MLFLOW_", "WANDB_")):
            monkeypatch.delenv(name, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


# --- The name -------------------------------------------------------------------------------


@pytest.mark.parametrize("name, parts", [
    ("run-slurm4242.0.rank0.jsonl", ("run-slurm4242.0", 0, None)),
    ("run-slurm4242.0.rank1.31337.jsonl", ("run-slurm4242.0", 1, 31337)),
    ("a.rank1.rank0.jsonl", ("a.rank1", 0, None)),
    ("a.rank1.rank0.7.jsonl", ("a.rank1", 0, 7)),
    ("x.rank0.bak.jsonl", None),
    ("x.rank0.jsonl.bak", None),
    ("x.rank٣.jsonl", None),
    ("x.rank0.٣.jsonl", None),
    ("x.jsonl", None),
    (".rank0.jsonl", None),
])
def test_the_records_file_name_is_read_from_the_right(name, parts):
    assert _sink.records_file_parts(name) == parts


def test_the_records_file_name_is_written_as_it_is_read():
    assert _sink.records_file_name("run-slurm1.0", 3) == "run-slurm1.0.rank3.jsonl"
    assert _sink.records_file_name("run-slurm1.0", 3, 99) == "run-slurm1.0.rank3.99.jsonl"
    assert _sink.records_file_parts(_sink.records_file_name("r.rank2", 0, 5)) == ("r.rank2", 0, 5)


# --- The lock and the second file ------------------------------------------------------------


def test_two_processes_with_one_run_id_and_rank_write_two_files(here):
    first = Child(here, "first", "same", 4)
    first_path = first.wait_ready()
    second = Child(here, "second", "same", 3)
    second_path = second.wait_ready()
    a, b = first.finish(), second.finish()
    out = here / "tensorquill-out"
    assert Path(a["path"]).name == Path(first_path).name == "same.rank0.jsonl"
    assert Path(second_path).name == f"same.rank0.{b['pid']}.jsonl"
    assert a["said"] == []
    assert len(b["said"]) == 1, b["said"]
    assert "same.rank0.jsonl" in b["said"][0] and f"same.rank0.{b['pid']}.jsonl" in b["said"][0]
    assert _steps(out / "same.rank0.jsonl") == [0, 1, 2, 3]
    assert _steps(out / f"same.rank0.{b['pid']}.jsonl") == [0, 1, 2]
    # Each file has its own end marker.
    assert (out / "same.rank0.end.json").exists()
    assert (out / f"same.rank0.{b['pid']}.end.json").exists()


def test_a_second_session_in_this_process_writes_the_pid_file(here):
    one, said_one, model_one = _init("same")
    two, said_two, model_two = _init("same")
    try:
        assert one._sink.path.name == "same.rank0.jsonl"
        assert two._sink.path.name == f"same.rank0.{os.getpid()}.jsonl"
        assert said_one == [] and len(said_two) == 1
        _train(model_one, 2)
        _train(model_two, 1)
    finally:
        two.close()
        one.close()
    out = here / "tensorquill-out"
    assert _steps(out / "same.rank0.jsonl") == [0, 1]
    assert _steps(out / f"same.rank0.{os.getpid()}.jsonl") == [0]


def test_a_third_writer_in_this_process_goes_on_unlocked_in_the_pid_file_and_says_so(here):
    one, _, _ = _init("same")
    two, _, _ = _init("same")
    three, said, model = _init("same")
    try:
        assert three._sink.path == two._sink.path
        assert len(said) == 1 and "lock" in said[0], said
        _train(model, 1)
    finally:
        for session in (three, two, one):
            session.close()


def test_a_resumed_run_appends_to_its_own_file(here):
    one, said_one, model = _init("resumed")
    _train(model, 3)
    one.close()
    two, said_two, model = _init("resumed", model, start_step=3)
    _train(model, 2)
    two.close()
    out = here / "tensorquill-out"
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["resumed.rank0.jsonl"]
    assert _steps(out / "resumed.rank0.jsonl") == [0, 1, 2, 3, 4]
    assert said_one == said_two == []


def test_a_run_resumed_after_its_writer_was_killed_appends_to_its_own_file(here):
    """A process killed with SIGKILL holds no lock any more: the next start takes the file."""
    first = Child(here, "killed", "resumed", 2)
    first.wait_ready()
    first.kill()
    session, said, model = _init("resumed", start_step=0)
    try:
        assert session._sink.path.name == "resumed.rank0.jsonl"
        assert said == []
        _train(model, 2)
    finally:
        session.close()
    out = here / "tensorquill-out"
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["resumed.rank0.jsonl"]


@pytest.mark.parametrize("error", [OSError(errno.ENOLCK, "No locks available"),
                                   OSError(errno.EOPNOTSUPP, "Operation not supported"),
                                   ValueError("not a lock")])
def test_a_lock_that_fails_is_no_lock_here(here, monkeypatch, error):
    def fails(fd, operation):
        raise error

    monkeypatch.setattr(_sink._fcntl, "flock", fails)
    session, said, model = _init("unlocked")
    try:
        assert session._sink.path.name == "unlocked.rank0.jsonl"
        assert getattr(session._sink, "locked", None) is False
        assert len(said) == 1 and "lock" in said[0], said
        _train(model, 2)
    finally:
        session.close()
    assert _steps(here / "tensorquill-out" / "unlocked.rank0.jsonl") == [0, 1]


def test_where_there_is_no_fcntl_the_file_is_written_unlocked_and_nothing_is_said(here, monkeypatch):
    monkeypatch.setattr(_sink, "_fcntl", None)
    session, said, model = _init("nofcntl")
    try:
        assert session._sink.path.name == "nofcntl.rank0.jsonl"
        assert said == []
        _train(model, 1)
    finally:
        session.close()


def test_init_does_not_block_while_another_process_holds_the_file(here):
    first = Child(here, "holder", "held", 1)
    first.wait_ready()
    try:
        began = time.monotonic()
        session, _, _ = _init("held")
        assert time.monotonic() - began < 10
        session.close()
    finally:
        first.finish()


def test_a_torn_line_of_a_file_another_writer_holds_is_not_touched(here):
    """The torn-line repair is made only by the writer that holds the lock: a newline written into
    the other writer's line would cut it in two."""
    one, _, _ = _init("torn")
    try:
        with open(one._sink.path, "a", encoding="utf-8") as file:
            file.write('{"partial":')
        two, _, _ = _init("torn")
        two.close()
        assert one._sink.path.read_text().endswith('{"partial":')
    finally:
        one.close()


def test_a_session_whose_init_fails_removes_the_empty_pid_file(here, monkeypatch):
    one, _, _ = _init("undone")
    from tensorquill import capture

    def fails(*args, **kwargs):
        raise RuntimeError("the session could not be made")

    # A second writer of rank 0 builds no run manifest (lane AR-7), so its `init` is made to fail later.
    monkeypatch.setattr(capture, "Session", fails)
    with pytest.raises(RuntimeError):
        _init("undone")
    one.close()
    assert sorted(p.name for p in (here / "tensorquill-out").glob("*.jsonl")) == ["undone.rank0.jsonl"]


def test_a_locked_sink_never_opens_its_file_a_second_time(here, monkeypatch):
    """Not at the torn-line check when it opens, nor after a write that raised. Where `flock` is
    a POSIX record lock in disguise (Linux's NFS client emulates it as one), closing any other
    descriptor of the file in this process would drop the lock."""
    path = here / "tensorquill-out" / "once.rank0.jsonl"
    path.parent.mkdir()
    path.write_text('{"step": 0}\n{"cut')
    opened: list[str] = []
    real_open = Path.open

    def counting(self, *args, **kwargs):
        if self.name == path.name:
            opened.append(str(args[0] if args else kwargs.get("mode", "r")))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting)
    sink = _sink.JsonlSink(path, lock=True)
    try:
        assert sink.locked is True
        sink._failed = True  # as after a write that raised
        sink.write([{"step": 1}])
    finally:
        sink.close()
    assert len(opened) == 1, opened
    assert path.read_text() == '{"step": 0}\n{"cut\n{"step":1}\n'


def test_a_jsonl_sink_made_by_hand_takes_no_lock(here):
    one = _sink.JsonlSink(here / "x.rank0.jsonl")
    two = _sink.JsonlSink(here / "x.rank0.jsonl")
    try:
        assert one.path == two.path and one.locked is False and two.locked is False
    finally:
        one.close()
        two.close()


# --- The readers ----------------------------------------------------------------------------


@pytest.fixture
def two_writers(here):
    """One rank of one run written by two sessions at once: `same.rank0.jsonl` (4 steps) and
    `same.rank0.<pid>.jsonl` (3 steps), each closed."""
    one, _, model_one = _init("same")
    two, _, model_two = _init("same")
    _train(model_one, 4)
    _train(model_two, 3)
    two.close()
    one.close()
    out = here / "tensorquill-out"
    return out, out / "same.rank0.jsonl", out / f"same.rank0.{os.getpid()}.jsonl"


def test_writers_of_names_both_files_from_either(two_writers):
    out, first, second = two_writers
    (out / "same.rank0.bak.jsonl").write_text(first.read_text())  # a copy: not a writer
    assert _sink.writers_of(first) == [first, second]
    assert _sink.writers_of(second) == [first, second]
    block = _sink.writers_block(first)
    assert block["state"] == "several" and block["files"] == [first.name, second.name]
    assert _sink.writers_block(out / "same.rank0.bak.jsonl") is None


@pytest.mark.parametrize("given", ["folder", "first", "second"])
def test_the_listing_reads_one_rank_of_two_writers_as_several_writers_not_healthy(two_writers, given):
    from tensorquill.runs import discover_runs, list_runs

    out, first, second = two_writers
    source = {"folder": out, "first": first, "second": second}[given]
    runs = discover_runs(source)
    assert [run["run_id"] for run in runs] == ["same"]
    assert runs[0]["other_writers"] == {0: [second]}
    assert runs[0]["files"] == {0: first}
    rows = list_runs(source)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "same" and row["ranks"] == [0]
    assert row["evidence"]["state"] == "several_writers"
    assert row["evidence"]["writers"] == [{"rank": 0, "files": [first.name, second.name]}]
    assert row["run_state"] is None and row["incidents"] is None and row["last_step"] is None
    assert second.name in row["evidence"]["detail"]


def test_the_listing_of_a_rank_named_only_by_its_pid_file_is_several_writers(two_writers):
    from tensorquill.runs import list_runs

    out, first, second = two_writers
    first.unlink()
    rows = list_runs(out)
    assert [(row["run_id"], row["evidence"]["state"]) for row in rows] == [("same", "several_writers")]


def test_a_backup_beside_the_file_is_not_a_writer(here):
    from tensorquill.runs import discover_runs, list_runs

    session, _, model = _init("solo")
    _train(model, 2)
    session.close()
    out = here / "tensorquill-out"
    (out / "solo.rank0.bak.jsonl").write_text((out / "solo.rank0.jsonl").read_text())
    runs = discover_runs(out)
    assert not runs[0].get("other_writers")
    assert list_runs(out)[0]["evidence"]["state"] != "several_writers"


def test_compare_refuses_a_rank_of_two_writers(two_writers):
    from tensorquill.runs import RunLookupError

    out, first, second = two_writers
    with pytest.raises(RunLookupError) as raised:
        tensorquill.compare_runs("same", "same", source=out)
    assert second.name in str(raised.value) and "two processes" in str(raised.value)


def test_the_evidence_of_either_file_names_both_writers(two_writers):
    from tensorquill.evidence import build_evidence

    out, first, second = two_writers
    for path in (first, second):
        evidence = build_evidence(path)
        assert evidence["writers"]["state"] == "several"
        assert evidence["writers"]["files"] == [first.name, second.name]


def test_the_evidence_of_either_file_is_not_healthy(two_writers):
    """Each file holds one process's records whole, and the detectors find nothing in either; the
    rank is still not one healthy writer, so the evidence says so and the summary does not open
    with a clean bill."""
    from tensorquill.evidence import build_evidence, incident_summary, run_state_line

    out, first, second = two_writers
    for path in (first, second):
        evidence = build_evidence(path)
        assert evidence["run_state"]["state"] == "unknown"
        assert any(r.get("code") == "several_writers" for r in evidence["run_state"]["reasons"])
        assert "Writers:" in incident_summary(evidence) and "unknown" in run_state_line(evidence)


def test_the_mcp_servers_evidence_of_either_file_is_not_healthy(two_writers):
    from tensorquill import mcp_server

    out, first, second = two_writers
    for path in (first, second):
        evidence = mcp_server.Source(path).capture_evidence(path, 0)
        assert evidence["run_state"]["state"] == "unknown"


def test_a_records_file_that_cannot_be_read_is_still_written(here):
    """The locked sink opens its file for reading too; a file it may write and not read is opened
    for appending alone, as before, and the session goes on."""
    path = here / "tensorquill-out" / "wo.rank0.jsonl"
    path.parent.mkdir()
    path.write_text('{"step": 0}\n')
    path.chmod(0o200)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("this user can read a file of mode 0o200 (root)")
        sink = _sink.JsonlSink(path, lock=True)
        sink.write([{"step": 1}])
        sink.close()
    finally:
        path.chmod(0o600)
    assert path.read_text() == '{"step": 0}\n{"step":1}\n'


def test_the_evidence_of_a_lone_writer_has_no_writers_block(here):
    from tensorquill.evidence import build_evidence

    session, _, model = _init("solo")
    _train(model, 2)
    session.close()
    assert "writers" not in build_evidence(here / "tensorquill-out" / "solo.rank0.jsonl")


def test_the_mcp_server_names_both_writers(two_writers):
    from tensorquill import mcp_server

    out, first, second = two_writers
    for path in (first, second):
        evidence = mcp_server.Source(path).capture_evidence(path, 0)
        assert evidence["writers"]["files"] == [first.name, second.name]
        answer = mcp_server.list_incidents(mcp_server.Source(path), {})
        assert answer["writers"]["files"] == [first.name, second.name]
