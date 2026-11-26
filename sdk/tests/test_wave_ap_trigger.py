"""Wave AP, lane AP-4: the trigger's state file and the compact evidence, audit finding AO-06.

For every name the watcher's output can have, with the trigger writing a compact copy or not, the
state file the trigger keeps and the one `tensorquill-mcp` started on `TENSORQUILL_EVIDENCE` reads
are one file. So a stall the trigger fired is returned by `get_incident` there. The state file
keeps the name it has today (a new name would fire every past incident again), the full evidence
the watcher wrote is never written over, and a state file already written is read and fires
nothing twice.

The names below are written out, not computed with `state_path_for`: they are what the code
before wave AP gives, measured on it.

They need what the SDK needs.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from tensorquill import compact as _compact
from tensorquill import mcp_server
from tensorquill.trigger import Trigger, add_arguments, from_args

HOOK = """
import json, os, sys
payload = json.load(sys.stdin)
env = {k: v for k, v in os.environ.items() if k.startswith("TENSORQUILL_")}
with open(sys.argv[1], "a") as log:
    log.write(json.dumps({"payload": payload, "env": env}) + "\\n")
"""

# One rank's evidence holding one stall, as the watcher writes it.
EVIDENCE = {"run_id": "r", "detection": {"rank": 0}, "run_state": {"state": "unhealthy"},
            "incidents": [{"kind": "rank_silent", "severity": "critical", "step": 4, "rank": 0, "silent_seconds": 950.0}]}

# The watcher's output name -> (the state file, the compact file the trigger writes with a window).
# Measured on the code before wave AP; neither may change.
NAMES = {
    "x.json": ("x.fired.json", "x.compact.json"),
    "x.evidence.json": ("x.evidence.fired.json", "x.evidence.compact.json"),
    "x.compact.json": ("x.fired.json", "x.compact.compact.json"),
    "x.evidence.compact.json": ("x.evidence.fired.json", "x.evidence.compact.compact.json"),
    "x": ("x.fired.json", "x.compact"),
    "x.txt": ("x.txt.fired.json", "x.compact.txt"),
    "x.compact": ("x.fired.json", "x.compact.compact"),
    "x.compact.compact.json": ("x.compact.fired.json", "x.compact.compact.compact.json"),
}
CASES = [pytest.param(name, window, id=f"{name}-{'compact' if window else 'full'}") for name in NAMES for window in (None, 5)]


class Run:
    """A folder holding the watcher's output, and a trigger whose command logs what it was given."""

    def __init__(self, tmp_path: Path, name: str) -> None:
        self.hook, self.log = tmp_path / "hook.py", tmp_path / "hook.log"
        self.hook.write_text(HOOK)
        self.folder = tmp_path / "run"
        self.folder.mkdir(exist_ok=True)
        self.path = self.folder / name
        self.text = json.dumps(EVIDENCE)
        self.path.write_text(self.text)

    def command(self) -> str:
        return " ".join(shlex.quote(str(part)) for part in (sys.executable, self.hook, self.log))

    def calls(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def look(self, trigger: Trigger, fired: int) -> None:
        trigger.look(json.loads(self.path.read_text()), self.path, rank=0)
        deadline = time.monotonic() + 20
        while len(self.calls()) < fired and time.monotonic() < deadline:
            trigger.close(wait=0.05)
            trigger.tick()
        trigger.close(wait=10)

    def files(self) -> list[str]:
        return sorted(path.name for path in self.folder.iterdir())


def _found(path: str | Path, mcp_id: str) -> dict:
    answer = mcp_server.get_incident(mcp_server.Source(Path(path)), {"id": mcp_id})
    assert answer.get("identified") is not False, answer
    return answer


@pytest.mark.parametrize("name, window", CASES)
def test_the_stall_fired_is_found_on_the_file_the_command_was_given(tmp_path, name, window):
    run = Run(tmp_path, name)
    run.look(Trigger(run.command(), compact_window=window), 1)
    [call] = run.calls()
    state, compact = NAMES[name]
    handed = compact if window else name
    assert Path(call["env"]["TENSORQUILL_EVIDENCE"]) == (run.folder / handed).resolve()
    # The state file under today's name, the compact copy beside the full file, nothing else.
    assert run.files() == sorted({name, state, *([compact] if window else [])})
    assert run.path.read_text() == run.text  # the full evidence is intact
    answer = _found(call["env"]["TENSORQUILL_EVIDENCE"], call["env"]["TENSORQUILL_INCIDENT_MCP_ID"])
    assert (answer["kind"], answer["step"]) == ("rank_silent", 4)
    # The full twin of a compact file answers as well.
    assert _found(run.path, call["env"]["TENSORQUILL_INCIDENT_MCP_ID"])["kind"] == "rank_silent"
    if window is None:
        # And the compact twin of the full file handed over, written afterwards (`--compact` on it).
        copy = _compact.write_compact(json.loads(run.path.read_text()), run.path, window=5)
        assert copy.name == compact and run.path.read_text() == run.text
        assert _found(copy, call["env"]["TENSORQUILL_INCIDENT_MCP_ID"])["kind"] == "rank_silent"


@pytest.mark.parametrize("name, window", CASES)
def test_a_state_file_already_written_is_read_and_nothing_fires_twice(tmp_path, name, window):
    """A watcher started again: a new trigger reads the state file the first wrote under its
    name, and fires nothing for the stall it already fired."""
    run = Run(tmp_path, name)
    run.look(Trigger(run.command(), compact_window=window), 1)
    before = run.files()
    state = (run.folder / NAMES[name][0]).read_text()
    again = Trigger(run.command(), compact_window=window)
    run.look(again, 1)
    deadline = time.monotonic() + 1.0  # a second firing would start its command within this
    while time.monotonic() < deadline:
        again.tick()
        again.close(wait=0.05)
    assert len(run.calls()) == 1 and run.files() == before
    assert json.loads((run.folder / NAMES[name][0]).read_text())["deliveries"] == json.loads(state)["deliveries"]


def test_the_trigger_the_command_line_makes_for_an_output_named_compact(tmp_path):
    """`--watch -o stall.compact.json --on-incident ...` with no --compact: the trigger writes the
    compact copy beside the full file, and the stall it fired is found on the copy."""
    run = Run(tmp_path, "stall.compact.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--compact-window", type=int)
    add_arguments(parser)
    args = parser.parse_args(["--watch", "--on-incident", run.command()])
    args.compact_window = None
    trigger = from_args(args, parser)
    assert trigger is not None and trigger.compact_window == _compact.DEFAULT_WINDOW
    run.look(trigger, 1)
    [call] = run.calls()
    assert Path(call["env"]["TENSORQUILL_EVIDENCE"]).name == "stall.compact.compact.json"
    assert run.files() == ["stall.compact.compact.json", "stall.compact.json", "stall.fired.json"]
    assert run.path.read_text() == run.text
    assert _found(call["env"]["TENSORQUILL_EVIDENCE"], call["env"]["TENSORQUILL_INCIDENT_MCP_ID"])["kind"] == "rank_silent"


def test_a_state_file_of_the_compact_name_is_taken_only_when_it_names_that_file(tmp_path):
    """The watcher's own output named `x.compact.compact.json`, with its state file
    `x.compact.fired.json`, beside a state file `x.fired.json` of another trigger that handed
    `x.compact.json`: the output is read with its own state file, as before wave AP."""
    run = Run(tmp_path, "x.compact.compact.json")
    run.look(Trigger(run.command()), 1)
    [call] = run.calls()
    other = {"version": 1, "evidence": str((run.folder / "x.compact.json").resolve()), "deliveries": [], "episodes": {}}
    (run.folder / "x.fired.json").write_text(json.dumps(other))
    assert _found(run.path, call["env"]["TENSORQUILL_INCIDENT_MCP_ID"])["kind"] == "rank_silent"


@pytest.mark.parametrize("name", list(NAMES))
def test_the_compact_name_leads_back_to_its_full_file_and_never_is_it(name):
    full = Path("/a") / name
    assert _compact.compact_path(full) != full
    assert _compact.compact_path(full).name == NAMES[name][1]
    assert _compact.full_path_of(_compact.compact_path(full)) == full
    assert _compact.full_path_of(Path("/a/y.json")) is None
