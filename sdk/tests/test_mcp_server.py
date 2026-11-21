"""The read-only MCP server (`python -m tensorquill.mcp_server PATH`), driven over stdio as a
client drives it: a child process, one JSON-RPC message a line.

What these tests hold it to:

- a scripted client that uses only the four tools recovers, for each of the seven Gate 1
  fixtures, the kind, the first step and the layer, parameter or module of the answer key, and
  finds that step again in the series itself;
- the tool list is exactly the six, all marked read-only, and a session changes no file (four
  until Nov 17, 2026, when `list_runs` and `compare_runs` joined them on purpose);
- `list_runs` and `compare_runs` read the folder the run is kept in, take run IDs and names and
  never a path, and refuse a name that fits several runs with their IDs;
- every reply is strict JSON of at most `MAX_ANSWER_BYTES`, whatever the run's length: a series
  is thinned with the stride reported, and a one-step spike between the points is still named;
- a run folder with two ranks gives the all-ranks status, and a rank whose report is not
  current is unknown, never clean;
- an unknown tool, a bad argument, a missing file and an unknown name each give a clear error,
  and the server answers the next request.

Only the last test needs `pyarrow` and the server binary; it is skipped without them. The
rest need what the SDK needs.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import tensorquill
from tensorquill import compact
from tensorquill import evidence as ev
from tensorquill import mcp_server

SRC = Path(tensorquill.__file__).resolve().parents[1]
SDK = Path(__file__).resolve().parents[1]
BAKEOFF = SDK.parent / "experiments" / "moe-bakeoff"
# The first four answer about the run at PATH; the last two (C4) about the runs kept beside it.
TOOLS = ["list_incidents", "get_incident", "get_series", "get_all_ranks_status", "list_runs", "compare_runs"]
GATE_1 = ["seeded", "unseen-1", "unseen-2", "unseen-3", "unseen-4", "unseen-5", "unseen-6"]
RUN = "two-ranks"


def child_env() -> dict[str, str]:
    """The child imports this checkout's package, as pytest's `pythonpath` makes the tests do."""
    inherited = os.environ.get("PYTHONPATH")
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), *([inherited] if inherited else [])])}


def _no_constants(name: str) -> float:
    raise AssertionError(f"the reply holds {name}, which is not JSON")


class Client:
    """An MCP client on a server's stdio. Replies are read by a thread, so a server that says
    nothing fails the test after `WAIT` seconds and cannot hang it. Every line read is checked
    to be strict JSON and within the size cap."""

    WAIT = 60

    def __init__(self, path: Path, *options: str, cwd: Path, log: Path) -> None:
        self.log = log.open("wb")  # a pipe nobody reads would block the server once it fills
        self.process = subprocess.Popen(
            [sys.executable, "-m", "tensorquill.mcp_server", str(path), *options],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, cwd=cwd, env=child_env(),
        )
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.last_id = 0
        self.largest = 0

    def _pump(self) -> None:
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send_line(self, line: bytes) -> None:
        self.process.stdin.write(line + b"\n")
        self.process.stdin.flush()

    def read(self):
        try:
            line = self.lines.get(timeout=self.WAIT)
        except queue.Empty:
            raise AssertionError(f"no reply in {self.WAIT} s") from None
        assert line is not None, "the server closed its output"
        assert line.endswith(b"\n") and b"\n" not in line[:-1]
        assert len(line) <= mcp_server.MAX_ANSWER_BYTES, f"a reply of {len(line)} bytes"
        self.largest = max(self.largest, len(line))
        return json.loads(line.decode("ascii"), parse_constant=_no_constants)

    def request(self, method: str, params: dict | None = None) -> dict:
        self.last_id += 1
        message = {"jsonrpc": "2.0", "id": self.last_id, "method": method, **({"params": params} if params is not None else {})}
        self.send_line(json.dumps(message).encode())
        reply = self.read()
        assert reply["jsonrpc"] == "2.0" and reply["id"] == self.last_id
        return reply

    def initialize(self, version: str = mcp_server.PROTOCOL_VERSION) -> dict:
        result = self.request("initialize", {
            "protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "test", "version": "0"},
        })["result"]
        self.send_line(b'{"jsonrpc":"2.0","method":"notifications/initialized"}')
        return result

    def result(self, tool: str, **arguments) -> dict:
        result = self.request("tools/call", {"name": tool, "arguments": arguments})["result"]
        assert [block["type"] for block in result["content"]] == ["text"]
        return result

    def call(self, tool: str, **arguments) -> dict:
        """A tool's answer, which must not be an error."""
        result = self.result(tool, **arguments)
        assert result["isError"] is False, result["content"][0]["text"]
        return json.loads(result["content"][0]["text"], parse_constant=_no_constants)

    def error(self, tool: str, **arguments) -> str:
        """The text of a tool's error, which it must be."""
        result = self.result(tool, **arguments)
        assert result["isError"] is True, result["content"][0]["text"][:300]
        return result["content"][0]["text"]

    def close(self) -> int:
        """End the input, as a client that is done does; the server's exit code."""
        self.process.stdin.close()
        try:
            return self.process.wait(20)
        finally:
            self.kill()

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(20)
        for stream in (self.process.stdin, self.process.stdout, self.log):
            if not stream.closed:
                stream.close()


@pytest.fixture
def start(tmp_path):
    """Start a server on a path and say hello: `start(path, *options) -> Client`. Each runs in
    its own empty working directory, and all are gone when the test ends."""
    started: list[Client] = []

    def begin(path: Path, *options: str, hello: bool = True) -> Client:
        cwd = tmp_path / f"cwd{len(started)}"
        cwd.mkdir()
        client = Client(path, *options, cwd=cwd, log=tmp_path / f"server{len(started)}.log")
        client.cwd = cwd
        started.append(client)
        if hello:
            client.initialize()
        return client

    yield begin
    for client in started:
        client.kill()


def gate_1_fixtures() -> dict:
    """The seven Gate 1 fixtures, from the bake-off's own generator: each has `records()` and
    the planted `answer`, which is written out there and never read from the detectors. The
    experiment's `artifacts` folder is not in the repository, so nothing here reads it. Empty
    when the experiment is not in this checkout (an sdist has only the package)."""
    if not (BAKEOFF / "generate.py").exists():
        return {}
    sys.path.insert(0, str(BAKEOFF))
    try:
        import generate
    finally:
        sys.path.remove(str(BAKEOFF))
    return dict(generate.BY_NAME)


FIXTURES = gate_1_fixtures()


@pytest.fixture(scope="module")
def gate1(tmp_path_factory):
    """Each fixture's capture, written as the SDK writes one, and the planted answers:
    `(folder, {name: answer})`, with the capture at `<folder>/<name>/<name>.jsonl`."""
    if not FIXTURES:
        pytest.skip("experiments/moe-bakeoff is not beside the SDK")
    assert sorted(FIXTURES) == GATE_1
    out = tmp_path_factory.mktemp("gate1")
    for name, fixture in FIXTURES.items():
        (out / name).mkdir()
        sink = tensorquill.JsonlSink(out / name / f"{name}.jsonl")
        sink.write(fixture.records())
        sink.close()
    return out, {name: fixture.answer for name, fixture in FIXTURES.items()}


# --- Records and stores made by hand -------------------------------------------------------


def dense(steps: int, *, rank: int = 0, run: str = "dense", spike: tuple[int, str, float] | None = None,
          stats: tuple[str, ...] = ("mean", "std", "abs_max", "zero_frac")) -> list[dict]:
    """Step records of a small dense model. `spike=(step, parameter, factor)` multiplies that
    parameter's gradient norm at that one step."""
    out = []
    for step in range(steps):
        wobble = 1.0 + 0.05 * math.sin(step * 0.7)
        grads = {f"blocks.{block}.weight": 0.2 * wobble * (block + 1) for block in range(3)}
        if spike is not None and step == spike[0]:
            grads[spike[1]] *= spike[2]
        out.append({
            "schema_version": 1, "run_id": run, "rank": rank, "attempt": 1, "step": step, "ts": 1.7e9 + step,
            "layers": {f"blocks.{block}": {stat: wobble * (i + 1) for i, stat in enumerate(stats)} for block in range(3)},
            "grad_norm": math.sqrt(sum(v * v for v in grads.values())), "grad_norms": grads,
            "lr": [1e-3], "scalars": {"loss": 2.0 * wobble},
        })
    return out


def write_capture(path: Path, records: list[dict], tail: str = "") -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records) + tail, encoding="utf-8")
    return path


SPIKE = {
    "kind": "grad_spike", "step": 40, "grad_norm": 9.0, "baseline": 1.0, "ratio": 9.0,
    "parameters": [{"name": "blocks.1.weight", "grad_norm": 8.9, "baseline": 0.4, "ratio": 22.25, "rising_since": 40}],
    "spike_count": 1, "spike_steps": [40],
}
NOT_FINITE = {"kind": "non_finite", "step": 55, "layers": ["blocks.2"], "gradients": []}
RESUMED = {
    "kind": "resume_lr_change", "step": 30, "resumes": 1, "note": "an observation: some restarts change the schedule on purpose",
    "changes": [{"step": 30, "previous_attempt": 1, "basis": "same_step", "expected_lr": [1e-3], "resumed_lr": [5e-4],
                 "first_lr": [1e-3], "equals_first_lr": False}],
}
ATTENTION = {"check": "probe", "kinds": list(ev.ATTN_KINDS), "reason": "1 of 3 probes failed, the first at step 2000: boom"}


def report(rank: int, incidents: list[dict], last_step: int = 59) -> dict:
    """A rank's `incidents.json` as `read_server_report` accepts it: for this run and rank,
    covering every server kind, with this version's thresholds. If a later version of the
    reader asks for more, `store` below fails and names the reason."""
    thresholds = {
        name.lower(): value for name, value in vars(ev).items()
        if name.isupper() and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {
        "schema_version": 1, "source": "tensorquill-server", "run_id": RUN, "rank": rank, "attempt": 1,
        "last_step": last_step, "steps": last_step + 1, "rebuilds": 0, "checked": list(ev.SERVER_KINDS),
        "thresholds": thresholds, "moe_flagged": {"samples": 0, "last_step": None},
        "incidents": incidents, "not_checked": [ATTENTION],
    }


@pytest.fixture
def store(tmp_path) -> Path:
    """A run folder as `tensorquill-server` leaves it, written by hand: two ranks, each with a
    current report and one part (only its name is ever read), and the run's status. Both ranks
    show the same gradient spike and the same resume; rank 1 alone has a non-finite layer."""
    folder = tmp_path / "store" / "runs" / RUN
    for rank, incidents in ((0, [SPIKE, RESUMED]), (1, [{**SPIKE, "grad_norm": 9.5}, RESUMED, NOT_FINITE])):
        parts = folder / f"rank={rank}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-0-59.parquet").write_bytes(b"")
        (folder / f"rank={rank}" / "incidents.json").write_text(json.dumps(report(rank, incidents)), encoding="utf-8")
        look = ev.read_server_report(folder / f"rank={rank}")
        assert look["state"] == "current", f"update `report` in this file: read_server_report now says {look['reason']}"
    (folder / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "summary": "finished", "world_size": 2,
        "ranks": [{"rank": rank, "state": "finished", "last_step": 59} for rank in (0, 1)],
    }), encoding="utf-8")
    return folder


def tree(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


# --- The protocol ---------------------------------------------------------------------------


def test_the_tool_list_is_exactly_the_six_read_only_tools(start, tmp_path):
    client = start(tmp_path / "absent.jsonl", hello=False)
    hello = client.initialize()
    assert hello["protocolVersion"] == "2025-06-18"
    assert hello["capabilities"] == {"tools": {"listChanged": False}}  # no resources, prompts or logging
    assert hello["serverInfo"]["name"] == "tensorquill"
    assert all(name in hello["instructions"] for name in TOOLS)
    tools = client.request("tools/list")["result"]["tools"]
    assert [tool["name"] for tool in tools] == TOOLS
    for tool in tools:
        assert tool["annotations"]["readOnlyHint"] is True and tool["annotations"]["destructiveHint"] is False
        assert tool["inputSchema"]["type"] == "object" and tool["inputSchema"]["additionalProperties"] is False
        assert tool["description"]
    assert client.request("ping")["result"] == {}
    for method in ("resources/list", "prompts/list", "tools/write"):
        assert client.request(method)["error"]["code"] == -32601
    assert client.close() == 0


@pytest.mark.parametrize("asked, given", [
    ("2025-06-18", "2025-06-18"), ("2025-03-26", "2025-03-26"), ("2024-11-05", "2024-11-05"),
    ("2099-01-01", "2025-06-18"), (None, "2025-06-18"),
])
def test_the_protocol_version_is_the_clients_when_known_and_this_servers_otherwise(asked, given):
    params = {"capabilities": {}, **({"protocolVersion": asked} if asked else {})}
    line = json.dumps({"jsonrpc": "2.0", "id": "a", "method": "initialize", "params": params}).encode()
    reply = json.loads(mcp_server.handle_line(mcp_server.Source("absent"), line))
    assert reply["id"] == "a" and reply["result"]["protocolVersion"] == given


def test_the_module_can_only_read():
    """Nothing in the module's text writes a file, starts a process or opens a connection."""
    text = Path(mcp_server.__file__).read_text(encoding="utf-8")
    for word in ("subprocess", "socket", "urllib", "http", "write_text", "write_bytes", "write_evidence",
                 "os.replace", "os.remove", "os.system", "unlink", "rmtree", "mkdir", "Popen", "watch(", "export_rerun",
                 "detectors=", "flight_recorder=str", "open("):
        assert word not in text.replace("os.fdopen(", "").replace(".open(", "").replace("def open(", ""), word
    assert text.count("os.fdopen(") == 1  # the reply stream itself


def test_a_session_changes_no_file(start, store, tmp_path):
    (tmp_path / "data").mkdir()
    capture = write_capture(tmp_path / "data" / "x.jsonl", dense(60, spike=(40, "blocks.1.weight", 60.0)))
    evidence = tmp_path / "data" / "x.evidence.json"
    evidence.write_text(json.dumps(ev.build_evidence(capture)), encoding="utf-8")
    before = tree(tmp_path / "data"), tree(store)
    for path in (capture, evidence, store):
        client = start(path)
        listed = client.call("list_incidents")
        for incident in listed["incidents"]:
            client.call("get_incident", id=incident["id"])
        client.result("get_series", name="blocks.1.weight")
        client.result("get_all_ranks_status")
        listed = client.call("list_runs")
        assert listed["count"] >= 1
        client.result("compare_runs", runs=[listed["runs"][0]["run_id"], "another"])
        assert client.close() == 0
        assert list(client.cwd.iterdir()) == []
    assert (tree(tmp_path / "data"), tree(store)) == before


# --- The Gate 1 fixtures: the plan's done test for the server, without a live agent -----------


def diagnose(client: Client) -> dict:
    """What a client can say about a run's first failure with the four tools alone, in the
    answer key's words. The step is the incident's, and is then found again in the series of
    the router, parameter or module the incident names, read from stride-1 points."""
    with pytest.raises(AssertionError):
        client.call("get_all_ranks_status")  # one rank's capture has none, and says so
    listed = client.call("list_incidents")
    assert listed["run_state"]["state"] == "unhealthy"
    failures = [i for i in listed["incidents"] if i["class"] in ("invariant", "anomaly")]
    first = min(failures, key=lambda incident: incident["step"])
    found = client.call("get_incident", id=first["id"])
    incident, where = found["incident"], found["where"]
    answer = {"kind": found["kind"], "step": found["step"]}
    if "layer" in where:
        answer.update(layer=where["layer"], overloaded=incident["overloaded"], dead=incident["dead"])
        series = client.call("get_series", name=where["router"])
        assert series["kind"] == "router" and series["stride"] == 1
        before = series["values"]["diversity"][0]
        fell = [step for step, value in zip(series["steps"], series["values"]["diversity"]) if value < 0.6 * before]
        assert fell[0] == answer["step"] == series["first_flagged_step"]
        for expert in incident["overloaded"][:1]:
            shares = client.call("get_series", name=where["router"], expert=expert, stat="expert_share")
            at = shares["steps"].index(answer["step"])
            assert shares["values"]["expert_share"][at] > 3 * shares["values"]["expert_share"][at - 1]
    elif "parameters" in where:
        answer.update(parameter=where["parameters"][0], rising_since=incident["parameters"][0]["rising_since"])
        whole = client.call("get_series", name=answer["parameter"])
        assert whole["kind"] == "parameter" and whole["over_range"]["grad_norm"]["max_step"] == answer["step"]
        near = client.call("get_series", name=answer["parameter"], start=answer["step"] - 20, end=answer["step"] + 20)
        assert near["stride"] == 1
        typical = whole["over_range"]["grad_norm"]["median"]
        high = [step for step, value in zip(near["steps"], near["values"]["grad_norm"]) if value > 5 * typical]
        assert high[0] == answer["step"]
    else:
        answer.update(module=where["modules"][0], stat=incident["modules"][0]["stat"])
        whole = client.call("get_series", name=answer["module"], stat=answer["stat"])
        assert whole["kind"] == "layer" and list(whole["values"]) == [answer["stat"]]
        near = client.call("get_series", name=answer["module"], stat=answer["stat"], start=answer["step"] - 20, end=answer["step"] + 20)
        typical = whole["over_range"][answer["stat"]]["median"]
        high = [step for step, value in zip(near["steps"], near["values"][answer["stat"]]) if value > 5 * typical]
        assert high[0] == answer["step"]
    return answer


@pytest.mark.parametrize("name", GATE_1)
def test_a_scripted_client_recovers_each_gate_1_answer_through_the_four_tools(name, gate1, start):
    out, answers = gate1
    client = start(out / name / f"{name}.jsonl")
    assert diagnose(client) == answers[name]
    assert client.close() == 0


def test_an_evidence_file_answers_from_what_is_written_and_reads_series_beside_it(gate1, start, tmp_path):
    out, _ = gate1
    for name in ("unseen-2", "seeded"):  # what `python -m tensorquill <name>.jsonl` writes beside the capture
        evidence = ev.build_evidence(ev.load_records(out / name / f"{name}.jsonl", all_attempts=True))
        (out / name / f"{name}.evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    # Beside its capture: the parameter's series is there.
    client = start(out / "unseen-2" / "unseen-2.evidence.json")
    listed = client.call("list_incidents")
    assert listed["source"]["kind"] == "evidence" and [i["id"] for i in listed["incidents"]] == ["grad_spike@310"]
    assert client.call("get_series", name="blocks.7.mlp.fc2.weight")["over_range"]["grad_norm"]["max_step"] == 310
    # Alone: its routers' samples are a series, and a layer or parameter is said to be missing.
    alone = tmp_path / "alone.evidence.json"
    alone.write_bytes((out / "seeded" / "seeded.evidence.json").read_bytes())
    client = start(alone)
    assert client.call("get_incident", id="moe_imbalance@1850")["sample"]["largest_shares"][0] == {"expert": 42, "share": 0.36}
    series = client.call("get_series", name="layers.31.router", expert=42)
    assert series["first_flagged_step"] == 1850 and series["records"] is None
    assert series["values"]["expert_share"][series["steps"].index(1850)] == 0.36
    assert "only its routers' samples" in client.error("get_series", name="blocks.7.mlp.fc2.weight")
    # With --records it reads the capture named.
    client = start(alone, "--records", str(out / "seeded" / "seeded.jsonl"))
    assert client.call("get_series", name="layers.31.router")["records"].endswith("seeded.jsonl")


def test_a_compact_evidence_file_reads_the_capture_beside_it(tmp_path):
    """CLI-05 (Nov 18, 2026): the trigger hands an agent `<name>.evidence.compact.json`
    (`$TENSORQUILL_EVIDENCE`), and a server started on it found no `<name>.jsonl` beside it,
    as it does for `<name>.evidence.json`, so it had no layer or parameter series."""
    capture = write_capture(tmp_path / "x.jsonl", dense(60, spike=(40, "blocks.1.weight", 60.0)))
    handed = compact.write_compact(ev.build_evidence(capture), tmp_path / "x.evidence.json")
    assert handed.name == "x.evidence.compact.json" and not (tmp_path / "x.evidence.json").exists()
    series = mcp_server.get_series(mcp_server.Source(handed), {"name": "blocks.1.weight"})
    assert series["records"] == str(capture) and series["over_range"]["grad_norm"]["max_step"] == 40
    assert series["recorded"]["steps"] == 60 and "compact" not in series


ROUTER = "blocks.0.router"


def moe(steps: int, collapse_at: int, experts: int = 8) -> list[dict]:
    """Step records of one router that routes evenly, then from `collapse_at` sends every token to
    two experts: a `moe_imbalance` at that step."""
    from tensorquill.capture import _router_stats

    out = []
    for step in range(steps):
        counts = [64 + (step * 7 + e * 13) % 9 for e in range(experts)]
        if step >= collapse_at:
            total = sum(counts)
            counts = [total // 2, total - total // 2] + [0] * (experts - 2)
        out.append({"schema_version": 1, "run_id": "moe", "rank": 0, "attempt": 1, "step": step, "ts": 1.7e9 + step,
                    "grad_norm": 1.0, "moe": {ROUTER: counts}, "moe_top_k": {ROUTER: 2},
                    "moe_stats": {ROUTER: _router_stats(counts)}})
    return out


def test_a_routers_series_from_compact_evidence_is_the_whole_run_when_the_records_are_held(tmp_path):
    """CLI-06 (Nov 18, 2026): with the step records held (`--records`, or the capture beside a
    compact file), a router's series came from the compact file's samples, which cover only a
    window around each MoE incident. It is read from the records, every step; the detectors'
    flags come from the file's own samples (the file may have been written with other thresholds
    or detectors than a rebuild would use), and the answer says they cover the kept steps only.
    With the compact file alone, the series is its window, and the answer says so."""
    (tmp_path / "beside").mkdir()
    capture = write_capture(tmp_path / "beside" / "m.jsonl", moe(80, 60))
    evidence = ev.build_evidence(capture)
    assert [(i["kind"], i["step"]) for i in evidence["incidents"]] == [("moe_imbalance", 60)]
    handed = compact.write_compact(evidence, capture.with_suffix(".evidence.json"), window=2)
    assert json.loads(handed.read_text())["compact"]["kept_steps"] == [[58, 62]]
    alone = tmp_path / "alone.evidence.compact.json"
    alone.write_bytes(handed.read_bytes())
    whole = mcp_server.get_series(mcp_server.Source(capture), {"name": ROUTER, "expert": 1})  # the capture itself
    assert whole["recorded"] == {"first_step": 0, "last_step": 79, "steps": 80} and whole["first_flagged_step"] == 60

    # Alone, with --records naming the capture; and beside the capture, as the trigger leaves it.
    for source in (mcp_server.Source(alone, records=capture), mcp_server.Source(handed)):
        series = mcp_server.get_series(source, {"name": ROUTER, "expert": 1})
        assert series["records"] == str(capture)
        assert series["recorded"] == whole["recorded"] and series["steps"] == whole["steps"]
        for column in ("diversity", "largest_share", "expert_share"):
            assert series["values"][column] == whole["values"][column], column
        # The flags are the file's own, at the steps it keeps, and said to be.
        at = {step: index for index, step in enumerate(series["steps"])}
        assert [series["values"]["overloaded"][at[s]] is not None for s in (57, 58, 62, 63)] == [False, True, True, False]
        assert series["over_range"]["overloaded"]["recorded"] == 5 and series["first_flagged_step"] == 60
        assert "steps 58-62" in series["compact"] and "flags" in series["compact"]

    # Alone: the window, said to be one, here and when a range falls outside it.
    series = mcp_server.get_series(mcp_server.Source(alone), {"name": ROUTER})
    assert series["records"] is None and series["steps"] == [58, 59, 60, 61, 62]
    assert "steps 58-62" in series["compact"] and "80 recorded steps" in series["compact"]
    with pytest.raises(mcp_server.ToolError, match="steps 58-62"):
        mcp_server.get_series(mcp_server.Source(alone), {"name": ROUTER, "start": 0, "end": 40})
    # The full evidence, beside its capture and alone: its own samples, every step, as before.
    full = capture.with_suffix(".evidence.json")
    ev.write_evidence(evidence, full)
    full_alone = tmp_path / "full.evidence.json"
    full_alone.write_bytes(full.read_bytes())
    for source in (mcp_server.Source(full), mcp_server.Source(full_alone)):
        series = mcp_server.get_series(source, {"name": ROUTER, "expert": 1})
        assert {k: series[k] for k in ("recorded", "steps", "values", "first_flagged_step")} == {
            k: whole[k] for k in ("recorded", "steps", "values", "first_flagged_step")} and "compact" not in series


def test_an_older_evidence_file_gets_its_classes_and_says_it_has_no_run_state(start, tmp_path):
    old = tmp_path / "old.json"
    old.write_text('{"schema_version": 1, "incidents": [{"kind": "grad_spike", "step": 5, "ratio": NaN},'
                   ' {"kind": "resume_lr_change", "step": 3}, {"kind": "made_up", "step": 9}], "samples": []}')
    client = start(old)
    listed = client.call("list_incidents")
    assert [(i["id"], i["class"]) for i in listed["incidents"]] == [
        ("grad_spike@5", "anomaly"), ("resume_lr_change@3", "observation"), ("made_up@9", "unclassified"),
    ]
    assert listed["by_class"] == {"anomaly": ["grad_spike"], "observation": ["resume_lr_change"], "unclassified": ["made_up"]}
    assert listed["run_state"] is None and "no run_state" in listed["run_state_note"]
    assert client.call("get_incident", id="grad_spike@5")["incident"]["ratio"] == "nan"  # NaN is not JSON
    assert "claims no failure" in client.call("get_incident", id="resume_lr_change@3")["class_meaning"]


# --- Bounded answers ------------------------------------------------------------------------


def test_a_10_000_step_run_gives_bounded_answers_and_a_spike_between_points_is_still_named(start, tmp_path):
    stats = tuple(f"stat{i}" for i in range(12))
    capture = write_capture(tmp_path / "long.jsonl", dense(10_000, spike=(7_777, "blocks.1.weight", 400.0), stats=stats))
    client = start(capture)
    series = client.call("get_series", name="blocks.1.weight")
    assert series["recorded"] == {"first_step": 0, "last_step": 9_999, "steps": 10_000}
    assert series["range"] == {"start": 0, "end": 9_999, "steps": 10_000}
    assert (series["stride"], series["points"]) == (50, 200) and series["steps"][:3] == [0, 50, 100]
    assert 7_777 not in series["steps"] and max(series["values"]["grad_norm"]) < 1.0
    assert series["over_range"]["grad_norm"]["max_step"] == 7_777 and series["over_range"]["grad_norm"]["max"] > 100
    assert "every 50th recorded step" in series["note"]
    near = client.call("get_series", name="blocks.1.weight", start=7_770, end=7_780)
    assert (near["stride"], near["points"]) == (1, 11) and "note" not in near
    assert near["values"]["grad_norm"][7] == near["over_range"]["grad_norm"]["max"]
    # Twelve columns at 1,000 points would be over the cap: the stride grows until the reply fits.
    wide = client.call("get_series", name="blocks.2", max_points=1000)
    assert len(wide["values"]) == 12 and wide["stride"] > 10 and wide["points"] == len(wide["steps"]) < 1000
    assert all(len(column) == wide["points"] for column in wide["values"].values())
    one = client.call("get_series", name="blocks.2", stat="stat3", max_points=1000)
    assert (one["stride"], one["points"]) == (10, 1000)
    assert client.call("get_series", name="loss", max_points=7)["points"] == 7
    assert client.call("get_series", name="lr")["values"] == {"group0": [0.001] * 200}
    listed = client.call("list_incidents")
    assert listed["source"]["steps"] == {"first": 0, "last": 9_999, "records": 10_000}
    assert 20_000 < client.largest <= mcp_server.MAX_ANSWER_BYTES  # `read` holds every reply to the cap


def test_a_fat_incident_and_a_long_list_are_cut_with_a_note_of_where(start, tmp_path):
    names = [f"encoder.layers.{i}.attention.self.query.weight" for i in range(5_000)]
    drift = {"kind": "replica_drift", "step": 20, "tolerance": 1e-6, "parameters": {
        name: {"norm_spread": 0.25, "sum_spread": 0.5, "projection_spread": 0.125} for name in names[:2_000]}}
    stalled = {"kind": "stalled_weights", "step": 12, "parameters": names, "stalled": len(names), "with_gradients": 5_000,
               "consecutive": 3, "stalled_steps": 3, "stalled_total": len(names), "onsets": [], "onsets_omitted": 0}
    many = [{**SPIKE, "step": step, "parameters": [{**SPIKE["parameters"][0], "name": name} for name in names[:40]]}
            for step in range(3_000)]
    fat = tmp_path / "fat.json"
    fat.write_text(json.dumps({"schema_version": 1, "incidents": [stalled, *many, drift], "samples": []}))
    client = start(fat)
    found = client.call("get_incident", id="stalled_weights@12")
    assert found["where"] == {"parameters": names[:12], "parameters_count": 5_000}
    kept = found["incident"]["parameters"]
    assert kept == names[:len(kept)] and 0 < len(kept) < 5_000 and found["incident"]["stalled"] == 5_000
    assert found["truncated"] == [f"incident.parameters: {len(kept)} of 5000 items shown"]
    # A map of small maps, as replica drift lists its parameters, is cut the same way.
    drifted = client.call("get_incident", id="replica_drift@20")
    spreads = drifted["incident"]["parameters"]
    assert 100 < len(spreads) < 2_000 and list(spreads) == names[:len(spreads)]
    assert spreads[names[0]] == {"norm_spread": 0.25, "sum_spread": 0.5, "projection_spread": 0.125}
    assert drifted["truncated"] == [f"incident.parameters: {len(spreads)} of 2000 entries shown"]
    assert drifted["where"] == {"parameters": names[:12], "parameters_count": 2_000}
    listed = client.call("list_incidents")
    assert listed["incident_count"] == 3_002 and listed["incidents_omitted"] == 2_952
    assert [i["id"] for i in listed["incidents"][:3]] == ["stalled_weights@12", "grad_spike@0", "grad_spike@1"]
    assert any(note.startswith("summary: ") for note in listed["truncated"])
    most = client.call("list_incidents", limit=200)
    assert 4 <= len(most["incidents"]) < 200 and client.largest > 35_000
    assert any(note.startswith(f"incidents: {len(most['incidents'])} of 200 items") for note in most["truncated"])


def test_fit_cuts_the_largest_part_and_leaves_the_original_alone():
    payload = {"small": [1, 2, 3], "big": list(range(4_000)), "text": "x" * 300, "nested": {"rows": [{"n": list(range(900))}]}}
    kept = json.loads(json.dumps(payload))

    def fits(candidate):
        return len(json.dumps(candidate)) <= 2_000

    cut = mcp_server.fit(payload, fits)
    assert payload == kept and fits(cut)
    assert cut["small"] == [1, 2, 3] and cut["big"] == list(range(len(cut["big"])))
    assert f"big: {len(cut['big'])} of 4000 items shown" in cut["truncated"]
    assert f"nested.rows[0].n: {len(cut['nested']['rows'][0]['n'])} of 900 items shown" in cut["truncated"]
    with pytest.raises(mcp_server.ToolError, match="could not be shortened further"):
        mcp_server.fit({"a": 1}, lambda _: False)


# --- A run folder with two ranks ------------------------------------------------------------


def test_the_all_ranks_status_of_a_two_rank_run_folder(start, store):
    client = start(store)
    status = client.call("get_all_ranks_status")
    assert status["source"] == {"path": str(store), "kind": "store", "run_id": RUN}
    assert status["run_state"]["state"] == "unhealthy"
    assert status["ranks"]["judged"] == {"count": 2, "ranks": "0-1"} and status["ranks"]["unknown"] == {"count": 0, "ranks": ""}
    assert status["ranks"]["with_incidents"] == {"count": 2, "ranks": "0-1"} and status["ranks"]["clean"]["count"] == 0
    assert [(r["state"], r["ranks"]) for r in status["reports"]] == [("current", "0-1")]
    assert status["last_step"] == {"min": 59, "max": 59}
    by_id = {i["id"]: i for i in status["incidents"]}
    assert list(by_id) == ["grad_spike@40", "resume_lr_change@30", "non_finite@55"]
    assert by_id["grad_spike@40"]["where"] == {"parameters": ["blocks.1.weight"]}
    assert (by_id["grad_spike@40"]["on_ranks"], by_id["grad_spike@40"]["numbers_differ"]) == ("0-1", True)
    assert (by_id["non_finite@55"]["on_ranks"], by_id["non_finite@55"]["on_all_judged_ranks"]) == ("1", False)
    assert by_id["non_finite@55"]["class"] == "invariant" and by_id["resume_lr_change@30"]["class"] == "observation"
    assert status["not_checked"] == [{**ATTENTION, "ranks": "0-1"}]
    assert [(r["rank"], r["report"], r["judged"], r["incidents"]) for r in status["per_rank"]] == [
        (0, "current", True, ["grad_spike", "resume_lr_change"]),
        (1, "current", True, ["grad_spike", "resume_lr_change", "non_finite"]),
    ]
    assert status["watchdog"]["summary"] == "finished" and "Common to all 2 judged ranks" in status["summary"]
    # These reports are an older server's, without `kind_checks`: what was checked is not counted.
    assert "kinds" not in status and status["kinds_not_stated"]["code"] == "report_without_kind_checks"
    assert status["kinds_not_stated"]["ranks"] == [0, 1] and " kinds checked" not in status["summary"]
    assert "What was checked is not stated" in status["summary"]
    assert client.call("get_all_ranks_status", rank=1)["rank"]["detection"]["report_last_step"] == 59
    assert "rank 7 is not in this run" in client.error("get_all_ranks_status", rank=7)

    listed = client.call("list_incidents")
    assert listed["incidents"] == status["incidents"] and listed["ranks"] == status["ranks"]
    assert "kinds" not in listed and listed["kinds_not_stated"] == status["kinds_not_stated"]
    only = client.call("list_incidents", rank=0)
    assert [i["id"] for i in only["incidents"]] == ["grad_spike@40", "resume_lr_change@30"] and only["rank"]["judged"] is True
    found = client.call("get_incident", id="grad_spike@40")
    assert found["ranks"] == {"on_ranks": "0-1", "rank_count": 2, "on_all_judged_ranks": True, "numbers_differ": True, "numbers_are_from_rank": 0}
    assert "rank 0's, not rank 1's" in client.call("get_incident", id="grad_spike@40", rank=1)["ranks"]["note"]
    assert "rank 0 is not among the ranks" in client.call("get_incident", id="non_finite@55", rank=0)["ranks"]["note"]
    # The follow-up call names the rank the numbers are from (CLI-15, lane V5, Nov 19, 2026).
    assert found["incident"]["grad_norm"] == 9.0 and found["look_next"][0]["arguments"] == {"name": "blocks.1.weight", "kind": "parameter", "rank": 0}
    # The parts made by hand are empty: the error says the parts or pyarrow, and the server goes on.
    assert re.search("stored parts|pyarrow", client.error("get_series", name="blocks.1.weight", rank=1))
    assert client.request("ping")["result"] == {}


def test_the_all_ranks_status_says_what_was_checked_for_the_run_and_never_counts_an_unknown_rank(start, store):
    """`kinds` over the protocol: rank 0 recorded weights and rank 1 did not, so the kind is not
    checked for the run and both ranks are named; then rank 0 falls behind and is in no value."""
    def with_checks(rank: int, **carried: int) -> None:
        path = store / f"rank={rank}" / "incidents.json"
        counts = {**dict.fromkeys(ev.INPUT_NEEDS, 0), "steps": 60, "grad_total": 60, "numbers": 60, **carried}
        document = {**json.loads(path.read_text()), "not_checked": [], "kind_checks": ev.kind_checks(counts, [])}
        path.write_text(json.dumps(document), encoding="utf-8")

    with_checks(0, weight_check=12)
    with_checks(1)
    client = start(store)
    status = client.call("get_all_ranks_status")
    kinds = status["kinds"]
    assert "kinds_not_stated" not in status
    assert kinds["checked"] == ["grad_spike", "repeated_steps", "non_finite", "job_stalled", "rank_silent"]
    assert "stalled_weights" in kinds["not_configured"]
    assert kinds["differs_by_rank"] == {"stalled_weights": {"checked": "0", "not_configured": "1"}}
    assert (kinds["judged_ranks"], kinds["ranks_not_counted"]) == ("0-1", "")
    # Since E3 (lane V3), a kind that reports an incident (resume_lr_change here) is counted as checked.
    assert [row["kinds"] for row in status["per_rank"]] == [
        {"checked": 5, "not_enough_history": 1, "not_configured": 21, "unavailable": 0},
        {"checked": 4, "not_enough_history": 1, "not_configured": 22, "unavailable": 0},
    ]
    assert status["summary"].splitlines()[-1] == (
        "6 of 33 kinds checked on both ranks, 1 of them without enough history yet (resume_lr_change 0 of 1 records"
        " but reported by a faster rule), 27 not configured, 1 of them checked on some ranks only"
        " (stalled_weights on rank 0)"
    )
    assert client.call("list_incidents")["kinds"] == kinds
    assert client.call("get_all_ranks_status", rank=0)["rank"]["kind_counts"]["checked"] == 5
    # Rank 0, the one that recorded weights, stores a part its report does not cover.
    (store / "rank=0" / "attempt=1" / "part-60-79.parquet").write_bytes(b"")
    after = client.call("get_all_ranks_status")["kinds"]
    assert (after["judged_ranks"], after["ranks_not_counted"]) == ("1", "0")
    assert "differs_by_rank" not in after and "stalled_weights" in after["not_configured"]
    described = next(tool["description"] for tool in mcp_server.TOOLS if tool["name"] == "get_all_ranks_status")
    for word in ("kinds.checked", "kinds.differs_by_rank", "kinds.ranks_not_counted", "kinds_not_stated"):
        assert word in described


def test_a_rank_whose_report_is_not_current_is_unknown_and_never_clean(start, store):
    client = start(store)
    assert client.call("get_all_ranks_status")["ranks"]["unknown"]["count"] == 0
    # The path is read again at every call. Rank 1 stores a part its report does not cover, and
    # the status now expects a third rank that never connected.
    (store / "rank=1" / "attempt=1" / "part-60-79.parquet").write_bytes(b"")
    status = json.loads((store / "watchdog.json").read_text())
    (store / "watchdog.json").write_text(json.dumps({**status, "summary": "active", "world_size": 3}))
    after = client.call("get_all_ranks_status")
    assert after["run_state"]["state"] == "unhealthy"
    assert "ranks_unknown" in [reason["code"] for reason in after["run_state"]["reasons"]]
    assert after["ranks"]["judged"] == {"count": 1, "ranks": "0"} and after["ranks"]["unknown"] == {"count": 2, "ranks": "1-2"}
    assert after["ranks"]["clean"]["count"] == 0
    assert {r["state"]: r["ranks"] for r in after["reports"]} == {"current": "0", "behind": "1", "nothing_stored": "2"}
    assert [i["id"] for i in after["incidents"]] == ["grad_spike@40", "resume_lr_change@30"]  # rank 1's are not counted
    behind = client.call("get_all_ranks_status", rank=1)["rank"]
    assert behind["detection"]["judged"] is False and [i["kind"] for i in behind["last_report"]["incidents"]][-1] == "non_finite"
    listed = client.call("list_incidents", rank=1)
    assert listed["incidents"] == [] and listed["rank"]["judged"] is False and "does not mean it has none" in listed["rank"]["note"]


def test_the_one_look_is_the_evidence_the_all_ranks_watcher_writes(store, start, tmp_path):
    """`all_ranks_look` is put together here from the watcher's own parts, because the SDK has
    no call that gives the all-ranks view without writing it. This holds the two together."""
    written = tmp_path / "all.json"
    assert ev.watch(store, written, rank="all", polls=1) == written
    theirs, ours = json.loads(written.read_text()), json.loads(json.dumps(mcp_server.all_ranks_look(store)))
    assert theirs.pop("written_at_ms") <= ours.pop("written_at_ms")
    assert theirs == ours and theirs["view"] == "all_ranks"
    # The written file is a path the server takes too, with the same answers.
    from_file, from_store = start(written), start(store)
    for tool, arguments in (("get_all_ranks_status", {}), ("list_incidents", {}), ("get_incident", {"id": "non_finite@55"})):
        a, b = from_file.call(tool, **arguments), from_store.call(tool, **arguments)
        assert a.pop("source")["kind"] == "all_ranks_evidence" and b.pop("source")["kind"] == "store"
        assert a == b


# --- Errors ---------------------------------------------------------------------------------


def test_bad_requests_get_clear_errors_and_the_server_keeps_answering(start, tmp_path):
    capture = write_capture(tmp_path / "x.jsonl", dense(80, spike=(40, "blocks.1.weight", 60.0)))
    client = start(capture)

    unknown = client.request("tools/call", {"name": "pause_run", "arguments": {}})["error"]
    assert unknown["code"] == -32602 and "Unknown tool: 'pause_run'" in unknown["message"]
    assert all(name in unknown["message"] for name in TOOLS)
    assert client.request("tools/call", {"name": "get_series", "arguments": ["blocks.1"]})["error"]["code"] == -32602
    assert client.request("tools/call", {})["error"]["code"] == -32602
    assert client.request("run/stop")["error"]["code"] == -32601

    assert "`name` is required" in client.error("get_series")
    assert "`id` is required" in client.error("get_incident")
    assert "takes no argument named 'layer'" in client.error("get_series", name="blocks.1", layer=3)
    assert "`max_points` must be between 1 and 1000, not 0" in client.error("get_series", name="loss", max_points=0)
    assert "`max_points` must be a whole number" in client.error("get_series", name="loss", max_points="many")
    assert "`start` (9) is after `end` (3)" in client.error("get_series", name="loss", start=9, end=3)
    assert "`kind` must be one of layer, parameter, router, scalar" in client.error("get_series", name="loss", kind="tensor")
    assert "`limit` must be between 1 and 200" in client.error("list_incidents", limit=10_000)

    missing = client.error("get_series", name="blocks.11.weight")
    assert "no layer, parameter, router or scalar is named 'blocks.11.weight'" in missing
    assert "3 layers, 3 parameters, 0 routers" in missing and "blocks.1.weight (parameter)" in missing
    assert "'blocks.1' is a layer here, not a parameter" in client.error("get_series", name="blocks.1", kind="parameter")
    assert "has no `max`. It has: mean, std, abs_max, zero_frac" in client.error("get_series", name="blocks.1", stat="max")
    assert "`expert` is for a router's series" in client.error("get_series", name="blocks.1", expert=2)
    assert "no recorded step between 500 and 600" in client.error("get_series", name="loss", start=500, end=600)
    assert "rank 3 is not in the capture" in client.error("get_series", name="loss", rank=3)
    assert "rank 3 is not in the capture" in client.error("list_incidents", rank=3)
    listed = client.call("list_incidents")
    wrong = client.error("get_incident", id="grad_spike@41")
    assert "no incident has the id 'grad_spike@41'" in wrong and listed["incidents"][0]["id"] in wrong

    # Lines that are not requests: each gets the JSON-RPC error for it, or silence for a notification.
    for line, code in ((b"{not json", -32700), (b'"text"', -32600), (b"[]", -32600), (b'{"jsonrpc":"2.0","id":null,"method":"ping"}', -32600),
                       (b'{"id":5,"method":"ping"}', -32600), (b'{"jsonrpc":"2.0","id":6,"method":"tools/call","params":[]}', -32602)):
        client.send_line(line)
        assert client.read()["error"]["code"] == code, line
    client.send_line(b'{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":1}}')
    client.send_line(b'{"jsonrpc":"2.0","id":7,"result":{}}')  # a reply to a request this server never sent
    client.send_line(b"")
    client.send_line(b'[{"jsonrpc":"2.0","id":8,"method":"ping"},{"jsonrpc":"2.0","method":"notifications/initialized"},'
                     b'{"jsonrpc":"2.0","id":9,"method":"nothing"}]')
    batch = client.read()  # the first thing said since: nothing answered the three lines before it
    assert [(reply["id"], "result" in reply) for reply in batch] == [(8, True), (9, False)]

    assert client.call("list_incidents")["incident_count"] == listed["incident_count"]
    assert client.process.poll() is None and client.close() == 0


def test_a_path_that_does_not_exist_is_an_error_until_it_does(start, tmp_path):
    path = tmp_path / "later.jsonl"
    client = start(path)
    for tool, arguments in (("list_incidents", {}), ("get_incident", {"id": "x@1"}), ("get_series", {"name": "loss"}), ("get_all_ranks_status", {})):
        assert f"{path} does not exist" in client.error(tool, **arguments)
    # A job starts writing: the last line has no newline yet, and is left out until it has.
    write_capture(path, dense(30), tail='{"schema_version": 1, "run_id": "dense", "step": 30, "gra')
    listed = client.call("list_incidents")
    assert listed["source"]["steps"] == {"first": 0, "last": 29, "records": 30} and listed["source"]["lines_left_out"] == 1
    # No failure, and a local file has no progress status to miss.
    assert listed["incidents"] == [] and listed["run_state"]["state"] == "healthy"
    # What "no incidents" covers: the kinds looked for, the ones never captured, and none that failed.
    kinds = listed["kinds"]
    assert "grad_spike" in kinds["checked"] and "job_stalled" in kinds["not_configured"] and kinds["unavailable"] == {}
    # Thirty steps are enough for a gradient spike and too few for activation drift, which is
    # checked and listed apart: "nothing found" is no verdict for it yet.
    # Activation drift needs 110 records (DETECTOR_RULES_CONTRACT.md, part 2), dead outputs still 60.
    assert kinds["not_enough_history"] == {"activation_drift": {"records": 30, "records_needed": 110},
                                           "dead_activations": {"records": 30, "records_needed": 60}}
    assert not set(kinds["checked"]) & set(kinds["not_enough_history"]) and "history_not_stated" not in kinds
    assert len(kinds["checked"]) + len(kinds["not_enough_history"]) + len(kinds["not_configured"]) == 33
    assert (f"{len(kinds['checked']) + 2} of 33 kinds checked, 2 of them without enough history yet"
            " (activation_drift 30 of 110 records, dead_activations 30 of 60)") in listed["summary"]
    assert client.call("get_series", name="loss")["range"]["end"] == 29

    client = start(tmp_path)  # a folder that is no run folder
    assert "holds no rank=<n> folder and no watchdog.json" in client.error("list_incidents")
    text = tmp_path / "notes.json"
    text.write_text('{"hello": 1}')
    assert "is JSON but not a TensorQuill evidence file" in start(text).error("list_incidents")
    text.write_text("not json at all")
    assert "is not a JSON document" in start(text).error("list_incidents")


def test_a_capture_of_two_ranks_reads_one_and_says_which(start, tmp_path):
    both = [*dense(60, rank=0), *dense(60, rank=1, spike=(40, "blocks.2.weight", 500.0))]
    client = start(write_capture(tmp_path / "both.jsonl", both))
    listed = client.call("list_incidents")
    assert listed["source"]["rank"] == 0 and listed["source"]["ranks_here"] == "0-1" and "pass rank=" in listed["source"]["note"]
    assert client.call("get_series", name="blocks.2.weight")["over_range"]["grad_norm"]["max"] < 1
    other = client.call("get_series", name="blocks.2.weight", rank=1)
    assert other["rank"] == 1 and other["over_range"]["grad_norm"]["max_step"] == 40
    assert client.call("list_incidents", rank=1)["source"]["rank"] == 1


def test_values_that_are_not_finite_are_strings_in_a_series(start, tmp_path):
    records = dense(20)
    records[7]["layers"]["blocks.0"]["mean"] = "nan"  # as the capture writes it
    records[9]["grad_norms"]["blocks.0.weight"] = float("inf")  # as Python reads it back from Parquet
    client = start(write_capture(tmp_path / "nan.jsonl", records))
    layer = client.call("get_series", name="blocks.0", stat="mean")
    assert layer["values"]["mean"][7] == "nan" and layer["over_range"]["mean"]["first_not_finite_step"] == 7
    grads = client.call("get_series", name="blocks.0.weight")
    assert grads["values"]["grad_norm"][9] == "inf" and grads["over_range"]["grad_norm"]["not_finite"] == 1
    assert client.call("list_incidents")["incidents"][0]["kind"] == "non_finite"


# --- The instructions ---------------------------------------------------------------------


def test_the_agent_instructions_name_the_six_tools_and_offer_no_action_on_the_run():
    text = (SDK / "AGENT_INSTRUCTIONS.md").read_text(encoding="utf-8")
    # `compare` joined the verbs looked for on Nov 17, 2026, with the tool `compare_runs`.
    named = set(re.findall(r"`((?:list|get|compare|set|put|write|pause|stop|kill|cancel|delete|update|create|restart)_[a-z_]+)", text))
    assert named == set(TOOLS)
    # What a comparison must not be read as, in the instructions and in the tool's own words.
    compared = next(tool["description"] for tool in mcp_server.TOOLS if tool["name"] == "compare_runs")
    for word in ("absent_from", "other_ranks_not_compared", "divergence"):
        assert word in text and word in compared, word
    for word in ("not_configured", "not_enough_history", "ended_before_step", "not_said", "--runs FOLDER"):
        assert word in text, word
    assert [tool["name"] for tool in mcp_server.TOOLS] == TOOLS
    for word in ("invariant", "anomaly", "observation", "not_checked", "unknown", "run_state", "python -m tensorquill.mcp_server"):
        assert word in text, word
    # "Checked" is told from "checked, not enough history yet", in the instructions and in the tool's own words.
    described = next(tool["description"] for tool in mcp_server.TOOLS if tool["name"] == "list_incidents")
    for word in ("kinds.checked", "kinds.not_enough_history", "records_needed", "kinds.history_not_stated"):
        assert word in text and word in described, word
    assert "not enough history yet" in text and "still warming up" in text and "still warming up" in described
    # The document sends the agent to each incident's `class` and copies no table of kinds. The
    # few kinds it gives as examples sit on a line that names their class, as the SDK has it.
    examples = [(kind, line) for line in text.splitlines() for kind in ev.INCIDENT_CLASSES if f"`{kind}`" in line]
    assert 1 <= len({kind for kind, _ in examples}) <= 6
    for kind, line in examples:
        assert ev.INCIDENT_CLASSES[kind] in line, f"{kind} is an {ev.INCIDENT_CLASSES[kind]}: {line}"


# --- The runs kept beside this one: list_runs and compare_runs ---------------------------------


def labelled(folder: Path, run_id: str, records: list[dict], evidence: bool = True, **manifest) -> Path:
    """A run as `tensorquill.init` leaves it: records, the manifest, and the evidence file."""
    folder.mkdir(parents=True, exist_ok=True)
    capture = write_capture(folder / f"{run_id}.rank0.jsonl", records)
    if manifest:
        (folder / f"{run_id}.run.json").write_text(json.dumps({"schema_version": 1, "run_id": run_id, "config": {}, **manifest}))
    if evidence:
        ev.write_evidence(ev.build_evidence(capture), capture.with_suffix(".evidence.json"))
    return capture


@pytest.fixture
def kept(tmp_path) -> Path:
    """A folder of four runs: two of one project, two that share a name, one with no manifest."""
    folder = tmp_path / "tensorquill-out"
    labelled(folder, "run-1", dense(60, run="run-1"), project="granite", name="baseline", config={"lr": 0.001})
    labelled(folder, "run-2", dense(60, run="run-2", spike=(40, "blocks.1.weight", 60.0)), project="granite",
             name="after-the-fix", config={"lr": 0.01})
    labelled(folder, "run-3", dense(60, run="run-3"), evidence=False, project="llama", name="baseline")
    labelled(folder, "old-run", dense(5, run="old-run"))
    return folder


def test_list_runs_gives_the_runs_beside_this_one_by_project(start, kept):
    client = start(kept / "run-2.rank0.jsonl")
    listed = client.call("list_runs")
    assert listed["folder"] == str(kept) and listed["count"] == 4
    assert listed["projects"] == {"granite": 2, "llama": 1, "uncategorized": 1}
    rows = {row["run_id"]: row for row in listed["runs"]}
    assert (rows["run-2"]["name"], rows["run-2"]["run_state"], rows["run-2"]["first_incident"]) == (
        "after-the-fix", "unhealthy", {"kind": "grad_spike", "step": 40})
    assert rows["run-1"]["incidents"] == {} and rows["run-1"]["kinds_checked"]["of"] == 33
    # No evidence file was written for run-3: nothing is said of it, and that is not "clean".
    assert (rows["run-3"]["run_state"], rows["run-3"]["incidents"], rows["run-3"]["evidence"]["state"]) == (None, None, "missing")
    assert (rows["old-run"]["project"], rows["old-run"]["name"], rows["old-run"]["has_manifest"]) == ("uncategorized", "old-run", False)
    only = client.call("list_runs", project="granite", limit=1)
    assert (only["count"], only["shown"], [row["run_id"] for row in only["runs"]]) == (2, 1, ["run-1"])
    assert "no run of project 'nobody'" in client.error("list_runs", project="nobody")
    assert "granite (2)" in client.error("list_runs", project="nobody")
    assert "takes no argument named 'path'" in client.error("list_runs", path=str(kept))


def test_list_runs_does_not_count_a_file_of_several_ranks_as_one(start, tmp_path):
    folder = tmp_path / "tensorquill-out"
    labelled(folder, "run-1", dense(60, run="run-1"), project="granite", name="baseline")
    both = [r for pair in zip(dense(60, run="both"), dense(60, run="both", rank=1)) for r in pair]
    write_capture(folder / "both.jsonl", both)  # a file with a name of its own, holding ranks 0 and 1
    client = start(folder / "run-1.rank0.jsonl")
    rows = {row["run_id"]: row for row in client.call("list_runs")["runs"]}
    assert (rows["both"]["ranks"], rows["both"]["rank_count"]) == ([0, 1], None)
    assert "both.jsonl has a name of its own" in rows["both"]["rank_count_not_known"]
    assert rows["run-1"]["rank_count"] == 1 and "rank_count_not_known" not in rows["run-1"]


def test_compare_runs_takes_ids_and_names_and_says_what_was_checked_before_the_incidents(start, kept):
    client = start(kept / "run-2.rank0.evidence.json")
    result = client.call("compare_runs", runs=["run-1", "after-the-fix"])
    assert [(run["label"], run["run_id"], run["name"]) for run in result["runs"]] == [
        ("A", "run-1", "baseline"), ("B", "run-2", "after-the-fix")]
    keys = list(result)
    assert keys.index("setup") < keys.index("checked") < keys.index("incidents") < keys.index("divergence")
    assert result["setup"]["differs"] == [{"field": "config.lr", "values": {"A": 0.001, "B": 0.01}}]
    (entry,) = result["incidents"]["entries"]
    assert (entry["kind"], entry["in"], entry["first_step"], entry["absent_from"]) == (
        "grad_spike", ["B"], {"B": 40}, {"A": "checked"})
    assert entry["where"]["B"] == {"parameters": ["blocks.1.weight"]}
    assert result["divergence"]["state"] in ("no_answer", "no_divergence", "diverged", "no_records") and result["divergence"]["detail"]

    # A short run's silence is not a "none", and the tool says so.
    short = client.call("compare_runs", runs=["old-run", "run-2"])
    assert short["incidents"]["entries"][0]["absent_from"] == {"A": "not_enough_history"}
    assert short["setup"]["no_manifest"] == ["A"]

    # A name that fits two runs is refused with their IDs; a project narrows it.
    refused = client.error("compare_runs", runs=["baseline", "run-2"])
    assert "fits 2 runs" in refused and "run-1" in refused and "run-3" in refused and "give the run ID" in refused
    narrowed = client.call("compare_runs", runs=["baseline", "run-2"], project="granite")
    assert [run["run_id"] for run in narrowed["runs"]] == ["run-1", "run-2"]
    assert "no run with the ID or name 'nobody'" in client.error("compare_runs", runs=["run-1", "nobody"])
    for bad in (["run-1"], "run-1", ["run-1", 2], [], ["run-1", ""], ["run-1"] * 9):
        assert "`runs` must be a list of 2 to 8" in client.error("compare_runs", runs=bad)
    assert "same run" in client.error("compare_runs", runs=["run-1", "baseline"], project="granite")


def test_list_runs_and_compare_runs_read_every_rank_of_a_run_of_several(start, kept):
    # run-4 has two ranks; the spike is on rank 1 only, and both ranks have their evidence file.
    labelled(kept, "run-4", dense(60, run="run-4"), project="granite", name="two-ranks")
    rank1 = write_capture(kept / "run-4.rank1.jsonl", dense(60, run="run-4", rank=1, spike=(40, "blocks.1.weight", 60.0)))
    ev.write_evidence(ev.build_evidence(rank1), rank1.with_suffix(".evidence.json"))
    client = start(kept / "run-1.rank0.jsonl")
    rows = {row["run_id"]: row for row in client.call("list_runs")["runs"]}
    assert (rows["run-4"]["rank_count"], rows["run-4"]["ranks_with_incidents"], rows["run-4"]["first_incident"]) == (
        2, [1], {"kind": "grad_spike", "step": 40})
    assert rows["run-1"]["rank_count"] == 1 and rows["run-4"]["run_state"] == "unhealthy"
    result = client.call("compare_runs", runs=["run-1", "two-ranks"])
    (entry,) = result["incidents"]["entries"]
    assert (entry["kind"], entry["in"], entry["ranks"], entry["absent_from"]) == ("grad_spike", ["B"], {"B": [1]}, {"A": "checked"})
    assert result["runs"][1]["read_from"]["judged_ranks"] == [0, 1] and result["divergence"]["state"] == "ranks_differ"
    on_zero = client.call("compare_runs", runs=["run-1", "two-ranks"], rank=0)
    assert on_zero["incidents"]["entries"] == [] and on_zero["incidents"]["other_ranks_not_compared"] == {"B": [1]}


def test_a_tool_argument_is_never_taken_as_a_path(start, kept, tmp_path):
    outside = labelled(tmp_path / "elsewhere", "secret", dense(60, run="secret"), project="other", name="secret")
    client = start(kept / "run-1.rank0.jsonl")
    for wanted in (str(outside), str(outside.parent), "../elsewhere/secret.rank0.jsonl", str(kept / "run-2.rank0.jsonl")):
        refused = client.error("compare_runs", runs=["run-1", wanted])
        assert "no run with the ID or name" in refused and "secret" not in refused.replace(wanted, "")
    assert "secret" not in json.dumps(client.call("list_runs"))
    # Started with --runs, the same server reads that folder and nothing else.
    elsewhere = start(kept / "run-1.rank0.jsonl", "--runs", str(outside.parent))
    assert [row["run_id"] for row in elsewhere.call("list_runs")["runs"]] == ["secret"]
    nowhere = start(kept / "run-1.rank0.jsonl", "--runs", str(tmp_path / "absent"))
    assert "is not a folder" in nowhere.error("list_runs") and "--runs FOLDER" in nowhere.error("compare_runs", runs=["a", "b"])
    assert nowhere.call("list_incidents")["run_state"]["state"] == "healthy"  # the run itself still answers


def test_a_run_folder_lists_and_compares_the_runs_of_its_store(start, store):
    """`store` is `runs/two-ranks` with two ranks; a second run is put beside it."""
    other = store.parent / "one-rank"
    (other / "rank=0" / "attempt=1").mkdir(parents=True)
    (other / "rank=0" / "attempt=1" / "part-0-59.parquet").write_bytes(b"")
    (other / "rank=0" / "incidents.json").write_text(json.dumps({**report(0, []), "run_id": "one-rank"}), encoding="utf-8")
    (other / "run.json").write_text(json.dumps({"run_id": "one-rank", "project": "granite", "name": "clean"}), encoding="utf-8")
    (other / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "summary": "finished", "world_size": 1, "ranks": [{"rank": 0, "state": "finished", "last_step": 59}],
    }), encoding="utf-8")
    for path in (store, store / "rank=1"):
        client = start(path)
        listed = client.call("list_runs")
        assert listed["folder"] == str(store.parent.resolve())
        assert {row["run_id"]: (row["project"], row["source"], row["ranks"]) for row in listed["runs"]} == {
            "one-rank": ("granite", "store", [0]), RUN: ("uncategorized", "store", [0, 1])}
        result = client.call("compare_runs", runs=["clean", RUN])
        # Every rank of each (Nov 17, 2026): rank 1's non-finite layer is compared, and named as rank 1's.
        assert [(e["kind"], e.get("ranks")) for e in result["incidents"]["entries"]] == [
            ("resume_lr_change", {"B": [0, 1]}), ("grad_spike", {"B": [0, 1]}), ("non_finite", {"B": [1]})]
        assert "other_ranks_not_compared" not in result["incidents"]
        assert all("no part was read" in run["read_from"]["judged_by"] for run in result["runs"])
        # Rank 0 of each, asked for: rank 1's non-finite layer is not compared, and the answer says so.
        on_rank_0 = client.call("compare_runs", runs=["clean", RUN], rank=0)
        assert [e["kind"] for e in on_rank_0["incidents"]["entries"]] == ["resume_lr_change", "grad_spike"]
        assert on_rank_0["incidents"]["other_ranks_not_compared"] == {"B": [1]}
        on_rank_1 = client.call("compare_runs", runs=["clean", RUN], rank=1)
        assert "non_finite" in [e["kind"] for e in on_rank_1["incidents"]["entries"]]
        assert on_rank_1["incidents"]["other_ranks_not_compared"] == {"B": [0]}
    # A report that is behind its store is not current. The tool opens no part and runs no
    # detector on a run folder, so it says so and compares nothing.
    (other / "rank=0" / "attempt=1" / "part-60-99.parquet").write_bytes(b"")
    refused = client.error("compare_runs", runs=["clean", RUN])
    assert "one-rank rank 0 has no current report from its server: it is behind" in refused
    assert client.call("list_runs")["count"] == 2


def test_a_listing_of_many_runs_and_a_comparison_of_fat_runs_stay_within_the_size_cap(start, tmp_path):
    folder = tmp_path / "many"
    for index in range(230):
        labelled(folder, f"run-{index:03d}", dense(2, run=f"run-{index:03d}"), evidence=False, project="sweep",
                 name=f"trial-{index}", tags=["t" * 40] * 5)
    wide = [f"blocks.{block}.sub.{part}.weight" for block in range(40) for part in range(10)]
    for run_id in ("fat-a", "fat-b"):
        records = dense(60, run=run_id)
        for record in records:
            record["grad_norms"] = {name: 0.1 for name in wide}
            record["layers"] = {name: {"mean": 0.0, "std": 1.0, "abs_max": 2.0, "zero_frac": 0.0} for name in wide}
        if run_id == "fat-b":
            for name in wide:
                records[40]["layers"][name]["abs_max"] = float("nan")
        labelled(folder, run_id, records, evidence=False, project="fat", name=run_id, config={f"key{i}": "v" * 100 for i in range(300)})
    client = start(folder / "fat-a.rank0.jsonl")
    listed = client.call("list_runs", limit=200)
    assert listed["count"] == 232 and len(listed["runs"]) <= 200
    result = client.call("compare_runs", runs=["fat-a", "fat-b"])
    # 300 config keys of 100 characters differ in nothing, and an incident naming 400 layers lists eight.
    (entry,) = result["incidents"]["entries"]
    assert (entry["kind"], entry["in"], len(entry["where"]["B"]["layers"]), entry["where"]["B"]["layers_count"]) == (
        "non_finite", ["B"], 8, 400)
    assert len(result["setup"]["config_keys_same"]) == 300 and "truncated" not in result
    # Every key differing is more than a reply holds: it is cut, with a note of where.
    labelled(folder, "fat-c", dense(60, run="fat-c"), evidence=False, project="fat", name="fat-c",
             config={f"key{i}": "w" * 100 for i in range(300)})
    cut = client.call("compare_runs", runs=["fat-a", "fat-c"])
    assert "truncated" in cut and cut["incidents"]["entries"] == []
    assert client.largest <= mcp_server.MAX_ANSWER_BYTES


# --- The same through the real server ---------------------------------------------------------


def test_a_store_written_by_the_real_server(start, tmp_path):
    """Two ranks streamed to `tensorquill-server`: the all-ranks tool reads its reports, and
    `get_series` reads its Parquet parts. Needs pyarrow and the server binary."""
    pytest.importorskip("pyarrow.flight")
    from tensorquill.flight import send_file
    from test_flight_e2e import BINARY, Server

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    server = Server(tmp_path / "store", 0, "--flush-steps", "50", "--flush-seconds", "0.2")
    try:
        folder = tmp_path / "store" / "runs" / RUN
        for rank, spike in ((0, None), (1, (70, "blocks.2.weight", 400.0))):
            capture = write_capture(tmp_path / f"{RUN}.rank{rank}.jsonl", dense(120, rank=rank, run=RUN, spike=spike))
            assert send_file(capture, server.endpoint, RUN, rank, timeout=30.0)
        deadline = time.monotonic() + 20
        while any(ev.read_server_report(folder / f"rank={rank}")["state"] != "current" for rank in (0, 1)):
            assert time.monotonic() < deadline, [ev.read_server_report(folder / f"rank={rank}")["reason"] for rank in (0, 1)]
            time.sleep(0.05)
        assert server.terminate() == 0
    finally:
        if server.process.poll() is None:
            server.process.kill()
            server.process.wait(20)
    expected = ev.build_evidence(dense(120, rank=1, run=RUN, spike=(70, "blocks.2.weight", 400.0)))["incidents"]
    assert [i["kind"] for i in expected] == ["grad_spike"]

    written = tmp_path / "all.json"
    ev.watch(folder, written, rank="all", polls=1)
    theirs, ours = json.loads(written.read_text()), json.loads(json.dumps(mcp_server.all_ranks_look(folder)))
    assert theirs.pop("written_at_ms") <= ours.pop("written_at_ms")
    for document in (theirs, ours):  # the status's age is taken at each look
        (document.get("watchdog") or {}).pop("not_written_for_seconds", None)
        for reason in document["run_state"]["reasons"]:
            reason.pop("detail")
    assert theirs == ours

    client = start(folder)
    status = client.call("get_all_ranks_status")
    assert status["ranks"]["judged"] == {"count": 2, "ranks": "0-1"} and status["ranks"]["clean"] == {"count": 1, "ranks": "0"}
    spike = next(i for i in status["incidents"] if i["kind"] == "grad_spike")
    assert (spike["id"], spike["on_ranks"], spike["where"]) == (f"grad_spike@{expected[0]['step']}", "1", {"parameters": ["blocks.2.weight"]})
    series = client.call("get_series", name="blocks.2.weight", rank=1)
    assert series["recorded"]["steps"] == 120 and series["over_range"]["grad_norm"]["max_step"] == 70
    assert client.call("get_series", name="blocks.2.weight")["rank"] == 0  # the lowest, when none is named
    assert client.call("get_series", name="blocks.2.weight")["over_range"]["grad_norm"]["max"] < 1
    assert client.close() == 0
