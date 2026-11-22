"""Wave AH, P5-1: the trigger hands over an id that the MCP server's `get_incident` takes.

Before it, the trigger named an incident `<run_id>/rank<r>/<kind>/<step>` (the delivery id, kept
as it is for receivers that drop repeats by it) and `get_incident` took only `<kind>@<step>`:
an agent woken by `--on-incident` that passed the id it was given was refused (proof 5 of
`wave-af-proof-cpu.md`). The payload, the webhook body and the environment now carry `mcp_id`,
`<run_id>/<ranks>/<kind>@<step>`, which `get_incident` resolves for that run and those ranks,
whatever the server was started on.

End to end here means the real `python -m tensorquill ... --watch --on-incident` command and the
real `python -m tensorquill.mcp_server` over stdio, on a store a real server wrote, with two
ranks that show the same kind at the same step and name different things.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import mcp_server  # noqa: E402
from tensorquill.trigger import Trigger  # noqa: E402

import test_trigger as tt  # noqa: E402
import test_watch_all_ranks as twa  # noqa: E402
from test_mcp_server import Client, child_env  # noqa: E402

RUN = twa.RUN
LAYER = {0: "blk2", 1: "blk3"}  # what each rank's NaN at step 30 names


@pytest.fixture
def hook(tmp_path):
    script = tmp_path / "hook.py"
    script.write_text(tt.HOOK)

    def command(log: Path) -> str:
        return " ".join(shlex.quote(str(part)) for part in (sys.executable, script, log))

    def calls(log: Path) -> list[dict]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return command, calls


@pytest.fixture
def servers():
    yield
    for server in twa.STARTED:
        if server.process.poll() is None:
            server.process.kill()
            server.process.wait(20)
        if not server.process.stdout.closed:
            server.process.stdout.close()
    twa.STARTED.clear()


def watch_until(folder: Path, view: str, out: Path, command: str, enough, seconds: float = 90.0) -> str:
    """The real watcher on the run folder, until `enough()` or the time is up; what it printed."""
    env = {**child_env(), "TENSORQUILL_SERVER_BIN": str(twa.BINARY)}
    watcher = subprocess.Popen(
        [sys.executable, "-m", "tensorquill", str(folder), "--rank", view, "--watch", "--interval", "0.1",
         "--output", str(out), "--on-incident", command],
        cwd=out.parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + seconds
        while not enough() and time.monotonic() < deadline and watcher.poll() is None:
            time.sleep(0.05)
    finally:
        watcher.send_signal(signal.SIGINT)
        try:
            printed, _ = watcher.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            watcher.kill()
            printed, _ = watcher.communicate()
    assert "Traceback" not in printed, printed
    return printed


def ask(source: Path, tmp_path: Path, name: str):
    client = Client(source, cwd=tmp_path, log=tmp_path / f"mcp-{name}.err")
    client.initialize()
    return client


@pytest.mark.skipif(twa.BINARY is None, reason="tensorquill-server is not built")
@pytest.mark.parametrize("view", ["0", "1", "all"])
def test_the_id_the_trigger_hands_over_is_one_get_incident_takes_for_that_rank(tmp_path, hook, servers, view):
    command, calls = hook
    run = twa.Run(tmp_path, *twa.FAST)
    for rank, layer in LAYER.items():
        run.send(rank, twa.steady(60, nan_at=(30, layer)))
    log, out = tmp_path / f"hook-{view}.log", tmp_path / f"view-{view}.evidence.json"
    wanted = [0, 1] if view == "all" else [int(view)]
    printed = watch_until(run.folder, view, out, command(log), lambda: len(calls(log)) >= len(wanted))
    fired = calls(log)
    assert sorted(c["payload"]["rank"] for c in fired) == wanted, printed
    sources = {"run": run.folder, "rank0": run.rank_folder(0), "rank1": run.rank_folder(1)}
    for call in fired:
        payload, rank = call["payload"], call["payload"]["rank"]
        # The delivery id is what it always was; the lookup id is beside it, in all three places.
        assert payload["id"] == f"{RUN}/rank{rank}/non_finite/30"
        assert payload["mcp_id"] == f"{RUN}/rank{rank}/non_finite@30"
        assert call["env"]["TENSORQUILL_INCIDENT_ID"] == payload["id"]
        assert call["env"]["TENSORQUILL_INCIDENT_MCP_ID"] == payload["mcp_id"]
        assert payload["incident"]["layers"] == [LAYER[rank]]
        for name, source in {**sources, "evidence": Path(payload["evidence"])}.items():
            client = ask(source, tmp_path, f"{view}-{rank}-{name}")
            try:
                answer = client.call("get_incident", id=payload["mcp_id"])
                # This rank's incident, never the other rank's at the same kind and step.
                assert (answer["kind"], answer["step"]) == ("non_finite", 30), (name, answer)
                assert answer["where"]["layers"] == [LAYER[rank]], (name, answer["id"], answer["where"])
                if "on_ranks" in answer:
                    assert answer["on_ranks"] == str(rank), (name, answer["on_ranks"])
                # The id the answer gives is the list's own, and it finds the same incident here.
                again = client.call("get_incident", id=answer["id"])
                assert again["where"] == answer["where"]
                # The other rank's lookup finds the other rank's, or is refused by name: never this one.
                other = 1 - rank
                result = client.result("get_incident", id=f"{RUN}/rank{other}/non_finite@30")
                text = result["content"][0]["text"]
                if result["isError"]:
                    assert f"rank {other}" in text, (name, text)
                else:
                    assert json.loads(text)["where"]["layers"] == [LAYER[other]], (name, text)
                # The delivery id is still not a lookup id, and the error says which one is.
                refused = client.error("get_incident", id=payload["id"])
                assert "mcp_id" in refused and payload["mcp_id"] in refused, refused
                # Another run's incident is refused by name.
                refused = client.error("get_incident", id=f"some-other-run/rank{rank}/non_finite@30")
                assert "some-other-run" in refused and RUN in refused, refused
            finally:
                client.close()
    assert run.server.terminate() == 0


# --- on made evidence -----------------------------------------------------------------------


def nan(step: int, layer: str) -> dict:
    return {"kind": "non_finite", "class": "invariant", "step": step, "layers": [layer], "parameters": [],
            "scalars": {}, "modules": [], "gradients": []}


def fire(tmp_path: Path, hook, evidence: dict, name: str, rank: int | None = None) -> list[dict]:
    command, calls = hook
    path, log = tmp_path / f"{name}.evidence.json", tmp_path / f"{name}.log"
    path.write_text(json.dumps(evidence))
    trigger = Trigger(command(log))
    trigger.look(evidence, path, rank=rank)
    assert tt._settle(trigger, lambda: all(d["command"]["status"] == "done" for d in trigger.deliveries))
    trigger.close(wait=10)
    return [c["payload"] for c in calls(log)]


def get(path: Path, **arguments) -> dict:
    return mcp_server.get_incident(mcp_server.Source(path), arguments)


def test_the_same_kind_twice_at_one_step_on_one_rank_is_told_apart(tmp_path, hook):
    evidence = {"run_id": "r", "detection": {"rank": 3}, "incidents": [nan(30, "a"), nan(30, "b"), nan(31, "c")]}
    # One firing for the rank, kind and step, as before: it is of the first of the two.
    payloads = fire(tmp_path, hook, evidence, "twice", rank=3)
    assert [(p["id"], p["mcp_id"]) for p in payloads] == [
        ("r/rank3/non_finite/30", "r/rank3/non_finite@30"), ("r/rank3/non_finite/31", "r/rank3/non_finite@31")]
    path = tmp_path / "twice.evidence.json"
    assert get(path, id="r/rank3/non_finite@30")["where"]["layers"] == ["a"]
    assert get(path, id="r/rank3/non_finite@30#2")["where"]["layers"] == ["b"]
    assert get(path, id="r/rank3/non_finite@30")["id"] == "non_finite@30"
    assert get(path, id="r/rank3/non_finite@30#2")["id"] == "non_finite@30#2"
    with pytest.raises(mcp_server.ToolError, match="no incident"):
        get(path, id="r/rank3/non_finite@30#3")
    with pytest.raises(mcp_server.ToolError, match="rank 3's evidence"):
        get(path, id="r/rank4/non_finite@30")


def test_a_group_of_ranks_and_a_stall_of_the_run_have_lookup_ids_too(tmp_path, hook):
    stall = {"kind": "rank_silent", "class": "observation", "step": 49, "severity": "critical",
             "assessment": "suspected_hang", "ranks": [{"rank": 1, "state": "silent", "severity": "critical",
                                                        "last_step": 49, "attempt": 7}],
             "other_ranks": [{"rank": 0, "state": "active", "last_step": 90, "attempt": 7}], "missing_ranks": [],
             "newest_step": 90}
    shared, alone = nan(30, "blk2"), nan(30, "blk3")
    evidence = {
        "view": "all_ranks", "run_id": "r", "written_at_ms": 1,
        "incident_groups": [
            {"incident": shared, "kind": "non_finite", "step": 30, "ranks": [0, 1, 2], "incident_rank": 0,
             "on_all_judged_ranks": False, "numbers_differ": False},
            {"incident": alone, "kind": "non_finite", "step": 30, "ranks": [3], "incident_rank": 3,
             "on_all_judged_ranks": False, "numbers_differ": False},
        ],
        "run_incidents": [stall], "ranks": [],
    }
    payloads = {p["mcp_id"]: p for p in fire(tmp_path, hook, evidence, "groups")}
    assert sorted(payloads) == ["r/rank3/non_finite@30", "r/ranks0-2/non_finite@30", "r/run/rank_silent@49"]
    path = tmp_path / "groups.evidence.json"
    assert get(path, id="r/ranks0-2/non_finite@30")["where"]["layers"] == ["blk2"]
    assert get(path, id="r/rank3/non_finite@30")["where"]["layers"] == ["blk3"]
    assert get(path, id="r/rank3/non_finite@30")["id"] == "non_finite@30#2"  # the list's id, where it is the second
    assert get(path, id="r/rank1/non_finite@30")["where"]["layers"] == ["blk2"]  # one rank of the group
    assert get(path, id="r/run/rank_silent@49")["kind"] == "rank_silent"
    assert "looked_up_as" not in get(path, id="r/run/rank_silent@49")
    # The same stall later in its episode: the other ranks finished, and it reads job_stalled at
    # the run's step. The id the trigger gave when it fired still finds it, and the answer says so.
    later = {**evidence, "run_incidents": [{**stall, "kind": "job_stalled", "step": 90, "other_ranks": []}]}
    path.write_text(json.dumps(later))
    answer = get(path, id="r/run/rank_silent@49")
    assert (answer["kind"], answer["step"], answer["id"], answer["looked_up_as"]) == (
        "job_stalled", 90, "job_stalled@90", "rank_silent@49")
    assert "as it reads now" in answer["note"]
    # Once the stall is over, the id finds nothing; and no other kind is ever answered in its place.
    path.write_text(json.dumps({**evidence, "run_incidents": []}))
    with pytest.raises(mcp_server.ToolError, match="no incident"):
        get(path, id="r/run/rank_silent@49")
    with pytest.raises(mcp_server.ToolError, match="no incident"):
        get(path, id="r/rank9/non_finite@31")
    with pytest.raises(mcp_server.ToolError, match="no incident"):
        get(path, id="r/rank9/non_finite@30")


def test_a_lookup_id_is_parsed_from_the_right_and_a_list_id_is_never_taken_for_one():
    parse = mcp_server._parse_lookup
    assert parse("grad_spike@310") is None and parse("grad_spike@310#2") is None and parse("job_stalled") is None
    assert parse("team/run-7/rank0/non_finite@412") == {
        "run": "team/run-7", "ranks": [0], "kind": "non_finite", "step": 412, "nth": 1}
    assert parse("r/ranks0-3,7/moe_imbalance@71#2") == {
        "run": "r", "ranks": [0, 1, 2, 3, 7], "kind": "moe_imbalance", "step": 71, "nth": 2}
    assert parse("r/run/job_stalled@99") == {"run": "r", "ranks": [], "kind": "job_stalled", "step": 99, "nth": 1}
    assert parse("r/run/duplicate_batches") == {"run": "r", "ranks": [], "kind": "duplicate_batches", "step": None, "nth": 1}
    for bad in ("r/rank0/non_finite/412", "r/rankX/non_finite@1", "r/rank0/non_finite@x", "/rank0/non_finite@1",
                "r/rank0/non_finite@1#0", "r/ranks3-1/non_finite@1"):
        assert parse(bad) is None, bad
    assert mcp_server.lookup_id("r", [0, 1, 2, 7], "non_finite", 30) == "r/ranks0-2,7/non_finite@30"
    assert mcp_server.lookup_id("r", [], "job_stalled", 99, 2) == "r/run/job_stalled@99#2"
    assert parse(mcp_server.lookup_id("a/b", [5], "k", None)) == {"run": "a/b", "ranks": [5], "kind": "k", "step": None, "nth": 1}


def test_the_webhook_body_and_an_old_state_file_carry_on(tmp_path, hook):
    """The lookup id is made when the payload is, from the evidence looked at, and is not kept in
    the state file: a file written before it loads, and its firings are not sent again."""
    command, calls = hook
    evidence = {"run_id": "r", "detection": {"rank": 0}, "incidents": [nan(30, "a")]}
    path, log = tmp_path / "old.evidence.json", tmp_path / "old.log"
    first = Trigger(command(log))
    first.look(evidence, path, rank=0)
    assert tt._settle(first, lambda: all(d["command"]["status"] == "done" for d in first.deliveries))
    first.close(wait=10)
    state = json.loads((tmp_path / "old.evidence.fired.json").read_text())
    assert all("mcp_id" not in d for d in state["deliveries"])
    again = Trigger(command(log))
    again.look({**evidence, "incidents": [nan(30, "a"), nan(40, "b")]}, path, rank=0)
    assert tt._settle(again, lambda: all(d["command"]["status"] == "done" for d in again.deliveries))
    again.close(wait=10)
    assert [c["payload"]["mcp_id"] for c in calls(log)] == ["r/rank0/non_finite@30", "r/rank0/non_finite@40"]
    assert os.environ.get("TENSORQUILL_INCIDENT_MCP_ID") is None
