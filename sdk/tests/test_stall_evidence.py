"""Row S4: the end marker a local run leaves when its session closes, and the stopped ranks'
stacks grouped in the all-ranks view with the odd group named only when one is clearly apart.

The killed runs are real processes ended by SIGKILL, so nothing of `close()` runs in them."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import tensorquill
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill.runs import list_runs, listing_text
from tensorquill.sink import end_marker_path, read_end_marker

SRC = str(Path(tensorquill.__file__).resolve().parent.parent)


def closed_run(folder: Path, run_id: str, steps: int = 3) -> Path:
    path = folder / f"{run_id}.rank0.jsonl"
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with tensorquill.init(model, run_id=run_id, every=1, sink=tensorquill.JsonlSink(path)):
        for _ in range(steps):
            optimizer.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
    return path


KILLED = textwrap.dedent("""
    import os, signal, sys, time
    import torch, tensorquill
    path, run_id = sys.argv[1], sys.argv[2]
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = os.path.getsize(path) if os.path.exists(path) else 0
    session = tensorquill.init(model, run_id=run_id, every=1, sink=tensorquill.JsonlSink(path))
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
    session.flush()
    deadline = time.time() + 30
    while time.time() < deadline and os.path.getsize(path) <= before:
        time.sleep(0.05)  # the writer thread has put this attempt's steps in the file
    os.kill(os.getpid(), signal.SIGKILL)
""")


def killed_run(folder: Path, run_id: str) -> Path:
    path = folder / f"{run_id}.rank0.jsonl"
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("TENSORQUILL_ENDPOINT", None)
    done = subprocess.run([sys.executable, "-c", KILLED, str(path), run_id], env=env, cwd=folder,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == -signal.SIGKILL, done.stderr
    return path


# --- Part 1: the end marker -------------------------------------------------------------------


def test_a_closed_run_leaves_a_marker_that_every_reader_shows(tmp_path):
    path = closed_run(tmp_path, "closed-1")
    marker = json.loads(end_marker_path(path).read_text())
    assert end_marker_path(path).name == "closed-1.rank0.end.json"
    assert marker["marker"] == "tensorquill_run_end" and marker["outcome"] == "closed"
    assert (marker["run_id"], marker["rank"]) == ("closed-1", 0) and marker["records_bytes"] == path.stat().st_size
    assert read_end_marker(path)["state"] == "closed"

    evidence = ev.build_evidence(path)
    assert evidence["run_end"]["state"] == "closed" and evidence["run_end"]["outcome"] == "closed"
    assert "Run end: the session closed the file" in ev.incident_summary(evidence)
    # Records handed over with the file they came from read it too; a bare list has no file to read.
    assert ev.build_evidence(ev.load_records(path), records_file=path)["run_end"]["state"] == "closed"
    assert "run_end" not in ev.build_evidence(ev.load_records(path))

    [row] = list_runs(tmp_path)
    assert row["end"] == {"state": "closed", "closed_ranks": [0]}
    assert listing_text([row], tmp_path).splitlines()[1].split()[-1] == "end"
    assert listing_text([row], tmp_path).splitlines()[2].rstrip().endswith("closed")

    listed = mcp_server.list_incidents(mcp_server.Source(str(path)), {})
    assert listed["run_end"]["state"] == "closed"
    assert mcp_server.list_runs(mcp_server.Source(str(path)), {})["runs"][0]["end"]["state"] == "closed"


def test_a_session_closed_by_an_exception_says_so(tmp_path):
    path = tmp_path / "raised.rank0.jsonl"
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ZeroDivisionError):
        with tensorquill.init(model, run_id="raised", sink=tensorquill.JsonlSink(path)):
            model(torch.randn(2, 4)).sum().backward()
            1 / 0
    end = read_end_marker(path)
    assert (end["state"], end["outcome"], end["exception"]) == ("closed", "raised", "ZeroDivisionError")
    assert "closed" in ev.run_end_line({"run_end": end}) and "by an exception (ZeroDivisionError)" in ev.run_end_line(
        {"run_end": end})
    [row] = list_runs(tmp_path)
    assert row["end"]["raised_ranks"] == [0] and "closed, raised" in listing_text([row], tmp_path)


def test_a_killed_run_reads_as_not_closed(tmp_path):
    path = killed_run(tmp_path, "killed-1")
    assert path.stat().st_size > 0 and not end_marker_path(path).exists()
    evidence = ev.build_evidence(path)
    assert evidence["run_end"]["state"] == "not_closed"
    line = ev.run_end_line(evidence)
    assert line.startswith("Run end: not known to have finished") and "killed" in line and "crash" not in line
    [row] = list_runs(tmp_path)
    assert row["end"]["state"] == "not_closed" and row["end"]["not_closed_ranks"] == [0]
    assert listing_text([row], tmp_path).splitlines()[2].rstrip().endswith("not closed")
    assert mcp_server.list_incidents(mcp_server.Source(str(path)), {})["run_end"]["state"] == "not_closed"


def test_a_closed_run_started_again_under_its_run_id_and_killed_is_not_closed(tmp_path):
    path = closed_run(tmp_path, "again")
    assert read_end_marker(path)["state"] == "closed"
    killed_run(tmp_path, "again")  # appends to the same file, and is killed
    assert not end_marker_path(path).exists()  # the new session removed it as it opened the file
    assert read_end_marker(path)["state"] == "not_closed"
    assert ev.build_evidence(path)["run_end"]["state"] == "not_closed"


def test_a_marker_the_file_has_outgrown_or_that_is_not_one_is_not_a_close(tmp_path):
    path = closed_run(tmp_path, "grown")
    with path.open("a") as file:  # a writer that did not remove the marker
        file.write(path.read_text().splitlines()[-1] + "\n")
    end = read_end_marker(path)
    assert end["state"] == "written_after_close" and end["now_bytes"] > end["records_bytes"]
    end_marker_path(path).write_text("{not json")
    assert read_end_marker(path)["state"] == "unreadable"
    end_marker_path(path).write_text(json.dumps({"marker": "someone_else", "records_bytes": 1}))
    assert read_end_marker(path)["state"] == "unreadable"


def test_a_run_of_two_ranks_is_closed_only_when_both_are(tmp_path):
    closed_run(tmp_path, "pair")
    second = tmp_path / "pair.rank1.jsonl"
    second.write_text((tmp_path / "pair.rank0.jsonl").read_text().replace('"rank":0', '"rank":1'))
    [row] = list_runs(tmp_path)
    assert row["end"]["state"] == "not_closed" and row["end"]["closed_ranks"] == [0]
    assert row["end"]["not_closed_ranks"] == [1]


def test_the_watcher_rewrites_when_the_marker_appears(tmp_path):
    """The marker comes after the last record: the file does not change, and its arrival alone
    must be news to the watcher of a local file."""
    path = closed_run(tmp_path, "watched")
    marker = end_marker_path(path)
    kept = marker.read_text()
    marker.unlink()
    seen: list[str] = []

    class Ticks:  # the watcher's trigger hook: told of every look, before it
        looks = 0

        def tick(self):
            Ticks.looks += 1
            if Ticks.looks == 3:
                marker.write_text(kept)

        def look(self, evidence, written, rank=None):
            seen.append(evidence["run_end"]["state"])

    ev.watch(path, tmp_path / "out.json", interval=0.01, polls=4, trigger=Ticks())
    assert seen == ["not_closed", "closed"]
    assert json.loads((tmp_path / "out.json").read_text())["run_end"]["state"] == "closed"


# --- Part 2: the stopped ranks' stacks --------------------------------------------------------

TORCH = "/opt/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py"
LOADER = "/opt/venv/lib/python3.12/site-packages/torch/utils/data/dataloader.py"
QUEUE = "/usr/lib/python3.12/queue.py"


def frame(file, function, line, own=False):
    return {"file": file, "function": function, "line": line, **({"own": True} if own else {})}


def in_collective(train="/home/a/job/train.py", line=7):
    return [frame(TORCH, "wait", 3001), frame(TORCH, "all_reduce", 2950), frame(train, "step", line),
            frame(train, "<module>", 40)]


def in_loader(train="/home/a/job/train.py", line=4, queue_line=180):
    return [frame(QUEUE, "get", queue_line), frame(LOADER, "_next_data", 1300), frame(train, "step", line),
            frame(train, "<module>", 40)]


def rank(number, frames=None, state="silent"):
    entry = {"rank": number, "attempt": 1, "last_step": 99, "seconds_since_progress": 400.0, "stream": "open",
             "state": state, "phase": "train", "heartbeat_age_seconds": 3.0, "severity": "warning"}
    if frames is not None:
        entry["stack"] = {"still_seconds": 400.0, "unchanged_seconds": 380.0 + number,
                          "threads": [{"name": "MainThread", "main": True, "frames": frames},
                                      {"name": "pin", "main": False, "frames": [frame(QUEUE, "get", 100 + number)]}]}
    return entry


def status(*ranks, summary="stalled"):
    return {"schema_version": 1, "run_id": "r", "written_at_ms": 1_000, "silent_floor_seconds": 300.0,
            "silent_factor": 5.0, "world_size": len(ranks), "missing_ranks": [], "summary": summary,
            "ranks": list(ranks)}


def test_three_ranks_with_one_odd_stack_name_it_as_an_observation():
    found = ev.group_stopped_stacks(status(rank(0, in_collective()), rank(1, in_collective()), rank(2, in_loader())))
    assert found["verdict"] == "odd_group" and found["odd_group"] == 1
    assert [g["ranks"] for g in found["groups"]] == [[0, 1], [2]]
    assert found["groups"][1]["where"] == {"function": "get", "file": QUEUE, "line": 180}
    assert found["groups"][1]["caller"]["function"] == "step" and found["ranks_without_stack"] == []
    line = ev.stopped_stacks_line(found)
    assert line.startswith("Stacks: rank 2 is somewhere else than the other 2 stopped ranks: `get` (queue.py:180)")
    assert "An observation, not the cause" in line and "data-loader worker" in line and "slow rank" in line
    assert any("never the cause" in c for c in found["caveats"])


def test_ranks_all_alike_name_nothing():
    found = ev.group_stopped_stacks(status(*(rank(n, in_collective()) for n in range(4))))
    assert found["verdict"] == "all_alike" and found["odd_group"] is None and len(found["groups"]) == 1
    assert "all 4 stopped ranks that sent one are at the same place" in ev.stopped_stacks_line(found)
    assert "no rank stands apart" in ev.stopped_stacks_line(found)


def test_ranks_with_no_stack_are_said_plainly_and_never_compared():
    found = ev.group_stopped_stacks(status(rank(0, state="unresponsive"), rank(1, state="unresponsive")))
    assert (found["verdict"], found["groups"], found["ranks_without_stack"]) == ("no_stacks", [], [0, 1])
    line = ev.stopped_stacks_line(found)
    assert line.startswith("Stacks: no stopped rank sent its stack") and "Ranks 0-1 sent no stack" in line
    # Two alike and one with no stack: alike, and the one without is named as not compared.
    found = ev.group_stopped_stacks(status(rank(0, in_collective()), rank(1, in_collective()),
                                           rank(2, state="unresponsive")))
    assert found["verdict"] == "all_alike" and found["ranks_without_stack"] == [2]
    assert ("Rank 2 sent no stack (no heartbeat, stopped for less than `stack_seconds`, or unresponsive), so it is"
            " not compared.") in ev.stopped_stacks_line(found)
    # No stopped rank at all: nothing to say, no block.
    assert ev.group_stopped_stacks(status(rank(0, state="active"), summary="active")) is None
    assert ev.group_stopped_stacks(None) is None


@pytest.mark.parametrize("stacks, verdict", [
    ([in_collective()], "one_stack"),
    ([in_collective(), in_loader()], "two_stacks_differ"),  # 1 v 1: no telling which is odd
    ([in_collective(), in_collective(), in_loader(), in_loader()], "no_clear_odd_group"),  # 2 v 2
    ([in_collective(), in_loader(), in_collective(line=9)], "no_clear_odd_group"),  # 1 v 1 v 1
    ([in_collective()] * 5 + [in_loader(), in_collective(line=9)], "no_clear_odd_group"),  # two apart, elsewhere
    ([in_collective()] * 5 + [in_loader()] * 3, "no_clear_odd_group"),  # 3 of 8 is more than a quarter
    ([in_collective()] * 6 + [in_loader()] * 2, "odd_group"),  # 2 of 8 is a quarter
])
def test_what_is_and_is_not_clearly_odd(stacks, verdict):
    found = ev.group_stopped_stacks(status(*(rank(n, s) for n, s in enumerate(stacks))))
    assert found["verdict"] == verdict
    assert (found["odd_group"] is not None) == (verdict == "odd_group")
    line = ev.stopped_stacks_line(found)
    if verdict != "odd_group":
        assert "somewhere else" not in line and ("no rank is named" in line or "none is named" in line)


def test_what_the_same_stack_means():
    same = ev.stack_signature
    # A library's line is not compared (its versions may differ between hosts): the same wait.
    assert same(rank(0, in_loader(queue_line=180))) == same(rank(1, in_loader(queue_line=171)))
    # Two lines of the job's own code are two calls.
    assert same(rank(0, in_collective(line=7))) != same(rank(1, in_collective(line=9)))
    # Another host's checkout folder and install prefix: the same place.
    other = [{**f, "file": f["file"].replace("/opt/venv/lib/python3.12", "/usr/local/lib/python3.11")}
             for f in in_collective(train="/scratch/b/run/train.py")]
    assert same(rank(0, in_collective())) == same(rank(1, other))
    # The same collective reached from somewhere else in the job is a different stack.
    deeper = in_collective()[:2] + [frame("/home/a/job/model.py", "forward", 12)] + in_collective()[2:]
    assert same(rank(0, in_collective())) != same(rank(1, deeper))
    # TensorQuill's own frames keep their line, as the job's do.
    own = "/opt/venv/lib/python3.12/site-packages/tensorquill/capture.py"
    assert same(rank(0, [frame(own, "hook", 10, own=True)])) != same(rank(1, [frame(own, "hook", 11, own=True)]))
    # Other threads and the timings do not count.
    assert same(rank(0, in_collective())) == same(rank(3, in_collective()))
    assert same(rank(0)) is None


def all_ranks_store(tmp_path: Path, document) -> Path:
    folder = tmp_path / "runs" / "r"
    folder.mkdir(parents=True)
    (folder / "watchdog.json").write_text(json.dumps(document))
    return folder


def test_the_all_ranks_view_and_its_tool_carry_the_groups(tmp_path):
    document = status(rank(0, in_collective()), rank(1, in_collective()), rank(2, in_loader()))
    folder = all_ranks_store(tmp_path, document)
    view = ev.all_ranks_evidence(folder)
    assert view["stopped_stacks"]["verdict"] == "odd_group" and view["stopped_stacks"]["groups"][1]["ranks"] == [2]
    assert "Stacks: rank 2 is somewhere else" in ev.incident_summary(view)
    # The incidents themselves are as the server writes them: no new field there.
    assert all("stopped_stacks" not in i for i in view["run_incidents"])
    answer = mcp_server.get_all_ranks_status(mcp_server.Source(str(folder)), {})
    assert answer["stopped_stacks"]["verdict"] == "odd_group" and "Stacks: rank 2" in answer["summary"]
    # The local all-ranks rules, given the same status.
    combined = ev.combine_rank_evidence({}, status=document, status_expected=True)
    assert combined["stopped_stacks"] == view["stopped_stacks"]
    # A run with nobody stopped has no block.
    quiet = all_ranks_store(tmp_path / "q", status(rank(0, state="active"), summary="active"))
    assert "stopped_stacks" not in ev.all_ranks_evidence(quiet)


def test_a_stack_that_arrives_or_moves_is_news_and_a_timing_is_not():
    without = status(rank(0), rank(1))
    arrived = status(rank(0, in_collective()), rank(1))
    stamp = ev._status_stamp
    assert stamp(without, None) != stamp(arrived, None)
    later = json.loads(json.dumps(arrived))
    later["ranks"][0]["stack"]["unchanged_seconds"] += 15
    later["ranks"][0]["stack"]["still_seconds"] += 15
    assert stamp(arrived, None) == stamp(later, None)
    moved = status(rank(0, in_loader()), rank(1))
    assert stamp(arrived, None) != stamp(moved, None)


def test_the_all_ranks_watcher_rewrites_when_a_stack_arrives_and_not_when_only_its_timing_moves(tmp_path):
    """The ranks went silent before their stacks came (a stack comes after `stack_seconds` without
    work): the arrival must be written, and the next beat's longer `unchanged_seconds` must not."""
    silent = status(rank(0), rank(1), rank(2))
    folder = all_ranks_store(tmp_path, silent)
    arrived = status(rank(0, in_collective()), rank(1, in_collective()), rank(2, in_loader()))
    later = json.loads(json.dumps(arrived))
    for entry in later["ranks"]:
        entry["stack"]["unchanged_seconds"] += 15
    written: list[str] = []

    class Ticks:
        looks = 0

        def tick(self):
            Ticks.looks += 1
            if Ticks.looks == 3:
                (folder / "watchdog.json").write_text(json.dumps(arrived))
            if Ticks.looks == 5:
                (folder / "watchdog.json").write_text(json.dumps(later))

        def look(self, evidence, path, rank=None):
            pass

    def update(evidence, path):
        written.append(evidence["stopped_stacks"]["verdict"])

    ev.watch(folder, rank="all", interval=0.0, polls=6, on_update=update, trigger=Ticks())
    assert written == ["no_stacks", "odd_group"]
