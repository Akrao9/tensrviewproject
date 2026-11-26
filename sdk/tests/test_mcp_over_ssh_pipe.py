"""The MCP server as SSH gives it to a client: `ssh <head> python -m tensorquill.mcp_server <store>/runs/<run_id>`.

SSH with a command and no `-t` gives the client two plain pipes and no terminal: the client
writes to the remote command's standard input and reads its standard output, and the remote
command is run by the user's login shell on the head node. Here the head node is this machine:
the command is handed to `/bin/sh -c`, as sshd hands it to the shell, in a new session with no
controlling terminal, its standard input and output pipes, and the run folder given as an
argument. `ssh` itself is not run.

What these tests hold:

- all six tools answer through that pipe on a store the real server wrote (two runs, one with
  two ranks and a gradient spike), and the server ends when the client closes its input;
- text the remote shell prints on standard output before the server starts (a `.bashrc` that
  echoes, a login banner) reaches the client ahead of the first reply: with a newline it is a
  first line that is not JSON, and without one it is joined to the reply to `initialize`, so
  that reply cannot be read at all;
- the same text on standard error leaves the protocol untouched.

The first test needs `pyarrow` and the server binary and is skipped without them; the rest
read a capture file and need what the SDK needs.
"""

from __future__ import annotations

import json
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from tensorquill import mcp_server

from test_mcp_server import TOOLS, Client, child_env, dense, write_capture

SHELL = "/bin/sh"
RUN, TWIN = "rm1", "rm1-twin"
SPIKE = (70, "blocks.2.weight", 400.0)
# The shell refuses to go on if it was given a terminal: the test is of pipes only.
NO_TERMINAL = "if [ -t 0 ] || [ -t 1 ]; then echo 'a terminal was given' >&2; exit 97; fi; "
BANNER = "Welcome to the head node"


def server_line(path: Path) -> str:
    """The remote command, as the client's SSH entry would send it: the interpreter, the module, the path."""
    return f"exec {shlex.quote(sys.executable)} -m tensorquill.mcp_server {shlex.quote(str(path))}"


class Remote(Client):
    """A client on `sh -c LINE`: the pipes SSH gives, no terminal, the shell's own start-up output
    included. `read` holds every line to be strict JSON within the cap; `raw` reads a line as it came."""

    def __init__(self, line: str, *, cwd: Path, log: Path) -> None:
        self.log = log.open("wb")
        self.log_path = log
        self.process = subprocess.Popen(
            [SHELL, "-c", NO_TERMINAL + line],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, cwd=cwd, env=child_env(),
            start_new_session=True,  # no controlling terminal, as under sshd without -t
        )
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.last_id = 0
        self.largest = 0

    def raw(self) -> bytes:
        try:
            line = self.lines.get(timeout=self.WAIT)
        except queue.Empty:
            raise AssertionError(f"no line in {self.WAIT} s") from None
        assert line is not None, f"the remote side closed its output: {self.log_path.read_text(errors='replace')}"
        return line


@pytest.fixture
def remote(tmp_path):
    """`remote(line) -> Remote`, each in its own empty working directory (SSH starts in the home
    folder, not beside the store), all killed when the test ends."""
    started: list[Remote] = []

    def begin(line: str) -> Remote:
        cwd = tmp_path / f"home{len(started)}"
        cwd.mkdir()
        client = Remote(line, cwd=cwd, log=tmp_path / f"stderr{len(started)}.log")
        started.append(client)
        return client

    yield begin
    for client in started:
        client.kill()


def hello() -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": mcp_server.PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "test", "version": "0"},
    }}).encode()


def is_json(line: bytes) -> bool:
    try:
        json.loads(line)
    except ValueError:
        return False
    return True


@pytest.fixture
def capture(tmp_path) -> Path:
    """A healthy 60-step capture, for the cases that are about the pipe and not about the store."""
    (tmp_path / "data").mkdir()
    return write_capture(tmp_path / "data" / f"{RUN}.rank0.jsonl", dense(60, run=RUN))


# --- The six tools, through the pipe, on a store the server wrote -------------------------------


@pytest.fixture
def store(tmp_path) -> Path:
    """`runs/rm1` (two ranks; rank 1 has a gradient spike in `blocks.2.weight` at step 70) and
    `runs/rm1-twin` (one rank, no spike), streamed to the real `tensorquill-server`, with every
    rank's report current. Returns the folder of `rm1`."""
    pytest.importorskip("pyarrow.flight")
    from tensorquill.flight import send_file
    from test_flight_e2e import BINARY, Server

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    root = tmp_path / "store"
    sent = [(RUN, 0, None), (RUN, 1, SPIKE), (TWIN, 0, None)]
    server = Server(root, 0, "--flush-steps", "50", "--flush-seconds", "0.2")
    try:
        for run_id, rank, spike in sent:
            path = write_capture(tmp_path / f"{run_id}.rank{rank}.jsonl", dense(120, rank=rank, run=run_id, spike=spike))
            assert send_file(path, server.endpoint, run_id, rank, timeout=30.0)
        deadline = time.monotonic() + 20
        reports = lambda: [ev.read_server_report(root / "runs" / r / f"rank={k}") for r, k, _ in sent]  # noqa: E731
        while any(look["state"] != "current" for look in reports()):
            assert time.monotonic() < deadline, [look["reason"] for look in reports()]
            time.sleep(0.05)
        assert server.terminate() == 0
    finally:
        if server.process.poll() is None:
            server.process.kill()
            server.process.wait(20)
    return root / "runs" / RUN


def test_all_six_tools_answer_through_a_plain_pipe_on_a_real_store(remote, store):
    client = remote(server_line(store))
    # The first line the client reads is the reply to `initialize`: the server prints nothing before it.
    started = client.initialize()
    assert started["serverInfo"]["name"] == "tensorquill" and started["protocolVersion"] == mcp_server.PROTOCOL_VERSION
    assert [tool["name"] for tool in client.request("tools/list")["result"]["tools"]] == TOOLS

    listed = client.call("list_incidents")
    assert listed["source"]["kind"] == "store" and listed["source"]["run_id"] == RUN
    [spike] = [i for i in listed["incidents"] if i["kind"] == "grad_spike"]
    assert spike["step"] == SPIKE[0]

    incident = client.call("get_incident", id=spike["id"])
    assert incident["incident"]["kind"] == "grad_spike" and incident["incident"]["step"] == SPIKE[0]

    # Read from the Parquet parts the server wrote: the spike is in the series of rank 1.
    series = client.call("get_series", name=SPIKE[1], kind="parameter", rank=1)
    assert series["recorded"] == {"first_step": 0, "last_step": 119, "steps": 120}
    [column] = series["values"]
    values = dict(zip(series["steps"], series["values"][column]))
    assert max(values, key=values.get) == SPIKE[0]

    status = client.call("get_all_ranks_status")
    assert status["ranks"]["judged"] == {"count": 2, "ranks": "0-1"} and status["ranks"]["clean"] == {"count": 1, "ranks": "0"}

    runs = client.call("list_runs")
    assert {row["run_id"]: row["ranks"] for row in runs["runs"]} == {RUN: [0, 1], TWIN: [0]}

    compared = client.call("compare_runs", runs=[RUN, TWIN])
    assert [run["run_id"] for run in compared["runs"]] == [RUN, TWIN]
    assert "grad_spike" in [entry["kind"] for entry in compared["incidents"]["entries"]]

    # The client closing its side (SSH ends the remote command's input) ends the server cleanly.
    assert client.close() == 0


# --- Text the remote shell prints before the server starts --------------------------------------


def test_a_line_printed_on_stdout_before_the_server_starts_is_the_first_line_the_client_reads(remote, capture):
    """A `.bashrc` that echoes: the client's first line is that text, which is no JSON-RPC message.
    The reply to `initialize` follows it whole, so only a client that drops lines it cannot parse
    gets past it."""
    client = remote(f"echo {shlex.quote(BANNER)}; " + server_line(capture))
    client.send_line(hello())
    first = client.raw()
    assert first == BANNER.encode() + b"\n" and not is_json(first)
    second = client.raw()
    assert json.loads(second)["id"] == 1 and "serverInfo" in json.loads(second)["result"]


def test_text_with_no_newline_before_the_server_starts_joins_the_reply_to_initialize(remote, capture):
    """`printf` with no newline (a prompt, a `module load` message): the text and the reply to
    `initialize` are one line, which is no JSON, so that reply is lost to any client that reads a
    message a line. The server itself is unharmed: the next request gets a whole reply."""
    client = remote(f"printf %s {shlex.quote(BANNER)}; " + server_line(capture))
    client.send_line(hello())
    first = client.raw()
    assert first.startswith(BANNER.encode() + b'{"jsonrpc"') and not is_json(first)
    client.send_line(b'{"jsonrpc":"2.0","id":2,"method":"ping"}')
    assert json.loads(client.raw()) == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_the_same_text_on_stderr_leaves_the_protocol_untouched(remote, capture):
    client = remote(f"echo {shlex.quote(BANNER)} >&2; " + server_line(capture))
    client.initialize()
    assert client.call("list_incidents")["run_state"]["state"] == "healthy"
    assert client.close() == 0
    # It reaches the client's stderr, beside anything the libraries warn about there.
    assert client.log_path.read_text().startswith(BANNER + "\n")
