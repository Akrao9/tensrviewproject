"""Wave AP, lane AP-4: the MCP server on audit finding AO-05.

1. A router's counts hold a point on a record by one rule, the rule the evidence is judged by
   (`evidence._router_total`: a list of signed 64-bit integers, no bool among them, whose sum is
   above 0). A series read from the records, and the check of whether the records still are what
   an evidence file judged (`_router_steps_changed`, wave AO), both use it. So for every kind of
   counts the detectors do not read, at any step, `get_series` beside the evidence file built from
   those records never fails and never says the records differ, and its steps are the steps the
   evidence holds a sample at.
2. One rule says what a JSON-RPC id is (a string, or a finite number that is not a bool), so a
   reply of any kind echoes exactly the ids a normal reply carries, and null otherwise.

They need what the SDK needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import compact as _compact
from tensorquill import evidence as ev
from tensorquill import mcp_server

ROUTER = "router0"
STEPS = 12
GOOD = [10, 10]

# Counts the detectors read at no step: `build_evidence` writes a missing sample there. The
# strings "nan", "inf" and "-inf" are how the capture writes a non-finite float; the bare `NaN`
# and `Infinity` are what Python's JSON reader takes.
REFUSED = {
    "a string": ["bad", "counts"],
    "a string beside a number": ["bad", 3],
    "None": [None, 1],
    "True": [True, 1],
    "False": [False, 3],
    "every count a bool": [True, True],
    "a float": [1.5, 2],
    "a whole float": [2.0, 3.0],
    "nan as the capture writes it": ["nan", 1],
    "inf as the capture writes it": ["inf", 1],
    "-inf as the capture writes it": ["-inf", 1],
    "a NaN float": [float("nan"), 1],
    "an infinite float": [float("inf"), 1],
    "an integer past 64 bits": [2**63, 1],
    "an integer too large for a float": [10**400, 1],
    "an empty list": [],
    "every count 0": [0, 0],
    "every count negative": [-1, -1],
    "a mapping": {"a": 1, "b": 2},
    "a number": 5,
    "a string alone": "10,10",
    "null": None,
    "a nested list": [[1], [2]],
}
POSITIONS = {"first": (0,), "middle": (STEPS // 2,), "last": (STEPS - 1,), "every": tuple(range(STEPS))}


def _records(bad: object, at: tuple[int, ...]) -> list[dict]:
    return [{"schema_version": 1, "run_id": "r", "rank": 0, "step": step, "loss": 1.0,
             "moe": {ROUTER: bad if step in at else list(GOOD)}} for step in range(STEPS)]


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _ask(source: mcp_server.Source, **arguments) -> dict:
    """`get_series` for the router through `handle_line`, as a client asks it: the reply's result."""
    line = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                       "params": {"name": "get_series", "arguments": {"kind": "router", "name": ROUTER, **arguments}}})
    reply = json.loads(mcp_server.handle_line(source, line.encode()))
    assert reply["id"] == 7 and "error" not in reply, reply
    return reply["result"]


def _judged_steps(document: dict) -> list[int]:
    """The steps the evidence holds a sample of the router at, as the detectors judged them."""
    return sorted(int(s["step"]) for s in document.get("samples") or ()
                  if s.get("router") == ROUTER and not s.get("missing"))


def _check(result: dict, judged: list[int], *, every: bool) -> None:
    text = result["content"][0]["text"]
    assert "failed inside the server" not in text, text
    if every:
        # No step holds a point: the tool error every router with no read counts gives (a mapping
        # at every step gave it before wave AP), never a failure.
        assert result["isError"] is True and text.startswith(f"no step carries values for the router {ROUTER!r}"), text
        return
    assert result["isError"] is False, text
    answer = json.loads(text)
    assert answer["steps"] == judged, answer["steps"]
    assert "differ" not in (answer.get("evidence_file") or ""), answer["evidence_file"]


CASES = [pytest.param(bad, at, id=f"{label}-{where}") for label, bad in REFUSED.items() for where, at in POSITIONS.items()]


@pytest.mark.parametrize("bad, at", CASES)
def test_counts_the_detectors_do_not_read_never_fail_and_never_make_the_records_differ(tmp_path, bad, at):
    """`build_evidence` on the records, then `get_series` on the untouched records beside that
    fresh file: no failure, no "differ", and the steps are the evidence's own. The capture alone
    answers the same steps."""
    capture = _write(tmp_path / "r.jsonl", _records(bad, at))
    document = ev.build_evidence(str(capture), detectors="python")
    judged = _judged_steps(document)
    every = len(at) == STEPS
    assert judged == ([] if every else sorted(set(range(STEPS)) - set(at)))
    report = tmp_path / "r.evidence.json"
    ev.write_evidence(document, report)

    _check(_ask(mcp_server.Source(report, records=capture)), judged, every=every)
    _check(_ask(mcp_server.Source(capture)), judged, every=every)


@pytest.mark.parametrize("bad, at", [pytest.param(bad, POSITIONS["middle"], id=label) for label, bad in REFUSED.items()]
                         + [pytest.param(bad, POSITIONS["every"], id=f"{label}-every") for label, bad in REFUSED.items()])
def test_a_compact_file_beside_the_records_reads_them_by_the_same_rule(tmp_path, bad, at):
    """Compact evidence that left samples out reads the routing from the step records (CLI-06):
    the same rule decides which of them hold a point."""
    capture = _write(tmp_path / "r.jsonl", _records(bad, at))
    document = ev.build_evidence(str(capture), detectors="python")
    small = _compact.compact_evidence(document, 1)
    assert _compact._omits(small) is not None
    report = tmp_path / "r.evidence.compact.json"
    ev.write_evidence(small, report)

    _check(_ask(mcp_server.Source(report, records=capture)), _judged_steps(document), every=len(at) == STEPS)


@pytest.mark.parametrize("bad, at", [pytest.param(bad, POSITIONS["first"], id=label) for label, bad in REFUSED.items()])
def test_records_nothing_judges_here_read_by_the_same_rule(tmp_path, monkeypatch, bad, at):
    """A run folder of a store given as --records, beside another run's file: nothing judges the
    records here, so the series is read from them with no flags (wave AO), by the same rule."""
    capture = _write(tmp_path / "r.jsonl", _records(bad, at))
    judged = _judged_steps(ev.build_evidence(str(capture), detectors="python"))
    held = mcp_server.Source(capture).rank_records(None)
    folder = tmp_path / "store" / "runs" / "r"
    folder.mkdir(parents=True)
    records, names, origin, scalar_only = held
    report = tmp_path / "elsewhere" / "other.evidence.json"
    report.parent.mkdir()
    other = [{**record, "run_id": "other", "moe": {ROUTER: list(GOOD)}} for record in _records(GOOD, ())]
    ev.write_evidence(ev.build_evidence(other), report)
    source = mcp_server.Source(report, records=folder)
    monkeypatch.setattr(source, "rank_records", lambda rank: (records, names, {**origin, "records": str(folder)}, scalar_only))

    result = _ask(source)
    _check(result, judged, every=False)
    assert "with no flags" in json.loads(result["content"][0]["text"])["evidence_file"]


def test_a_negative_count_beside_a_larger_one_is_a_point_on_both_sides(tmp_path):
    """Pinned: the detectors read `[-5, 10]` (signed 64-bit integers, sum above 0), so it is a point
    in the evidence and in the records alike, and the file still stands for them."""
    capture = _write(tmp_path / "r.jsonl", _records([-5, 10], (3,)))
    document = ev.build_evidence(str(capture), detectors="python")
    assert _judged_steps(document) == list(range(STEPS))
    report = tmp_path / "r.evidence.json"
    ev.write_evidence(document, report)
    answer = json.loads(_ask(mcp_server.Source(report, records=capture))["content"][0]["text"])
    assert answer["steps"] == list(range(STEPS)) and "evidence_file" not in answer


def test_records_that_changed_after_the_file_are_still_said_to_differ(tmp_path):
    """The reverse holds: a step whose counts became unreadable after the file was written is a
    change (the file judged a point there, the records hold none now)."""
    capture = _write(tmp_path / "r.jsonl", _records(GOOD, ()))
    report = tmp_path / "r.evidence.json"
    ev.write_evidence(ev.build_evidence(str(capture), detectors="python"), report)
    _write(capture, _records(["bad", 1], (STEPS - 1,)))
    answer = json.loads(_ask(mcp_server.Source(report, records=capture))["content"][0]["text"])
    assert answer["steps"] == list(range(STEPS - 1))
    assert "differ from what it judged" in answer["evidence_file"]


# --- 2. One rule for a JSON-RPC id ----------------------------------------------------------

IDS = {
    "a string": ("s", "s"), "an empty string": ("", ""), "an integer": (1, 1), "zero": (0, 0), "a negative": (-3, -3),
    "a fraction": (1.5, 1.5), "an integer too large for a float": (10**400, 10**400),
    "true": (True, None), "false": (False, None), "null": (None, None), "a list": ([1], None), "an object": ({"a": 1}, None),
}


def _reply(source: mcp_server.Source, message: dict) -> dict:
    return json.loads(mcp_server.handle_line(source, json.dumps(message).encode()))


@pytest.fixture
def source(tmp_path):
    return mcp_server.Source(_write(tmp_path / "r.jsonl", _records(GOOD, ())))


@pytest.mark.parametrize("given, echoed", list(IDS.values()), ids=list(IDS))
def test_every_reply_echoes_the_ids_a_normal_reply_carries(source, monkeypatch, given, echoed):
    """A ping (the main check in `handle`), a request with no method, a failure inside `handle`,
    and a reply that is not strict JSON: all four carry the same id, null for one that is not a
    string or a finite number that is not a bool."""
    ping = _reply(source, {"jsonrpc": "2.0", "id": given, "method": "ping"})
    assert ping["id"] == echoed and ("result" in ping) == (echoed is not None)
    assert _reply(source, {"jsonrpc": "2.0", "id": given})["id"] == echoed

    def broken(source, message):
        raise RuntimeError("inside")

    monkeypatch.setattr(mcp_server, "handle", broken)
    failed = _reply(source, {"jsonrpc": "2.0", "id": given, "method": "ping"})
    assert failed["id"] == echoed and failed["error"]["code"] == -32603
    monkeypatch.setattr(mcp_server, "handle", lambda source, message: {"jsonrpc": "2.0", "id": message.get("id"),
                                                                        "result": {"x": float("nan")}})
    unstrict = _reply(source, {"jsonrpc": "2.0", "id": given, "method": "ping"})
    assert unstrict["id"] == echoed and unstrict["error"]["code"] == -32603


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
def test_an_id_that_is_no_json_number_is_null_in_every_reply(source, monkeypatch, text):
    """Python's JSON reader takes NaN and Infinity, which no reply can carry."""
    for line in (f'{{"jsonrpc": "2.0", "id": {text}, "method": "ping"}}', f'{{"jsonrpc": "2.0", "id": {text}}}'):
        assert json.loads(mcp_server.handle_line(source, line.encode()))["id"] is None
    monkeypatch.setattr(mcp_server, "handle", lambda source, message: 1 / 0)
    line = f'{{"jsonrpc": "2.0", "id": {text}, "method": "ping"}}'
    assert json.loads(mcp_server.handle_line(source, line.encode()))["id"] is None


def test_the_rule_is_the_one_helper():
    assert all(mcp_server._is_request_id(given) == (echoed is not None) for given, echoed in IDS.values())
    assert not any(mcp_server._is_request_id(v) for v in (float("nan"), float("inf"), float("-inf")))
