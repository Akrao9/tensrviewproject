"""Lane C, wave AG: a router recorded with no tokens (WAVE_AF_CONTRACT.md, section 8), in the Python readers.

A router the records name whose counts are 0 on every record was not read. When that is every
router, `moe_imbalance` and `starved_experts` are `unavailable`, and one `not_checked` entry
`{"check": "moe_routing", "kinds", "routers", "reason"}` names them sorted. When it is some
routers, the kinds are judged on the others as on the base, and the entry names only those not
read. A run where every router was read is unchanged. Under the Rust detectors the server's report
decides, and the Python side shows what it says.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from tensorquill import mcp_server

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases_ag.json").read_text())["cases"]
NAMES = sorted(CASES)
KINDS = ["moe_imbalance", "starved_experts"]


def reason(names: list[str]) -> str:
    return "; ".join(f"router {name} recorded no tokens on any step: its routing was not read" for name in names)


def answer(evidence: dict) -> dict:
    """What a case's `expected` and `today` say, read from one evidence."""
    checks = evidence["kind_checks"]
    entries = [i for i in evidence.get("not_checked") or () if i.get("check") == "moe_routing"]
    return {
        "moe_imbalance": checks["moe_imbalance"]["value"], "starved_experts": checks["starved_experts"]["value"],
        "incident_kinds": sorted({i["kind"] for i in evidence["incidents"]}),
        "not_checked_checks": [i["check"] for i in evidence.get("not_checked") or ()],
        "not_checked_routers": entries[0]["routers"] if entries else None,
        "run_state": evidence["run_state"]["state"],
    }


def assert_expected(found: dict, expected: dict) -> None:
    for key, value in expected.items():
        if key == "not_checked_has":
            assert value in found["not_checked_checks"], found
        elif key == "run_state_is_not":
            assert found["run_state"] != value, found
        else:
            assert found[key] == value, (key, found)


@pytest.mark.parametrize("name", NAMES)
def test_each_case_with_the_python_detectors(name):
    case = CASES[name]
    assert_expected(answer(ev.build_evidence(case["given"]["records"], detectors="python")), case["expected"])


def test_every_router_with_no_tokens_is_said_and_its_kinds_are_unavailable():
    evidence = ev.build_evidence(CASES["every_router_recorded_with_no_tokens"]["given"]["records"])
    routers = ["blocks.0.router", "blocks.1.router"]
    entry = {"check": "moe_routing", "kinds": KINDS, "routers": routers, "reason": reason(routers)}
    assert [i for i in evidence["not_checked"] if i["check"] == "moe_routing"] == [entry]
    assert list(next(i for i in evidence["not_checked"] if i["check"] == "moe_routing")) == ["check", "kinds", "routers",
                                                                                            "reason"]
    for kind in KINDS:
        check = evidence["kind_checks"][kind]
        assert check["value"] == "unavailable" and check["check"] == "moe_routing" and check["reason"] == reason(routers)
    assert evidence["run_state"]["state"] == "degraded"
    assert "Not checked (moe_routing): " + reason(routers) in ev.incident_summary(evidence)


def test_one_router_not_read_beside_one_that_was_is_named_and_the_kinds_are_judged_as_before():
    case = CASES["one_router_with_no_tokens_beside_one_that_was_read"]
    evidence = ev.build_evidence(case["given"]["records"])
    assert [i for i in evidence["not_checked"] if i["check"] == "moe_routing"] == [
        {"check": "moe_routing", "kinds": KINDS, "routers": ["blocks.0.router"], "reason": reason(["blocks.0.router"])}]
    # The kinds and the incidents are the base's (`today`), judged on the router that was read.
    today = case["today"]
    found = answer(evidence)
    for key in ("moe_imbalance", "starved_experts", "incident_kinds", "run_state"):
        assert found[key] == today[key], key


def test_a_router_read_on_some_records_is_read():
    records = [dict(r) for r in CASES["routers_that_were_read"]["given"]["records"]]
    for record in records[:40]:
        record["moe"] = {name: [0] * len(counts) for name, counts in record["moe"].items()}
    evidence = ev.build_evidence(records)
    assert not [i for i in evidence.get("not_checked") or () if i["check"] == "moe_routing"]


def test_the_entry_comes_after_refused_routers_and_both_make_the_kinds_unavailable():
    records = [dict(r) for r in CASES["one_router_with_no_tokens_beside_one_that_was_read"]["given"]["records"]]
    records[0] = {**records[0], "moe_refused": {"blocks.2.router": "not indices"}}
    evidence = ev.build_evidence(records)
    checks = [i["check"] for i in evidence["not_checked"]]
    assert checks.index("router") + 1 == checks.index("moe_routing")
    # The refused router's entry makes the kinds unavailable whatever was read, as on the base.
    assert evidence["kind_checks"]["moe_imbalance"]["value"] == "unavailable"
    assert evidence["kind_checks"]["moe_imbalance"]["check"] == "router"


# ---- Under the Rust detectors: the server's report decides ---------------------------------------


def _rust_or_skip() -> None:
    from tensorquill import server_detect

    if server_detect.find_binary() is None:
        pytest.skip("no tensorquill-server binary")
    pytest.importorskip("pyarrow")


def test_under_the_rust_detectors_the_python_rule_is_not_run(monkeypatch):
    _rust_or_skip()
    calls = []
    real = ev.routers_not_read_not_checked
    monkeypatch.setattr(ev, "routers_not_read_not_checked", lambda records: calls.append(1) or real(records))
    ev.build_evidence(CASES["every_router_recorded_with_no_tokens"]["given"]["records"], detectors="rust")
    assert calls == []
    ev.build_evidence(CASES["every_router_recorded_with_no_tokens"]["given"]["records"], detectors="python")
    assert calls == [1]


def test_under_the_rust_detectors_the_servers_entry_and_values_are_shown(monkeypatch):
    """What the server's report says is what the evidence says (a report with the entry, as lane D's
    server writes it, stood in for here)."""
    _rust_or_skip()
    from tensorquill import server_detect

    real = server_detect.server_report
    routers = ["blocks.0.router", "blocks.1.router"]
    entry = {"check": "moe_routing", "kinds": KINDS, "routers": routers, "reason": reason(routers)}

    def with_entry(*args, **kwargs):
        report = real(*args, **kwargs)
        report["not_checked"] = [*(report.get("not_checked") or ()), entry]
        for kind in KINDS:
            report["kind_checks"][kind] = {**report["kind_checks"][kind], "value": "unavailable",
                                           "check": "moe_routing", "reason": entry["reason"]}
        return report

    monkeypatch.setattr(server_detect, "server_report", with_entry)
    evidence = ev.build_evidence(CASES["every_router_recorded_with_no_tokens"]["given"]["records"], detectors="rust")
    assert entry in evidence["not_checked"]
    assert all(evidence["kind_checks"][kind]["value"] == "unavailable" for kind in KINDS)
    assert evidence["run_state"]["state"] != "healthy"


def _entries_text(evidence: dict) -> str:
    """The `moe_routing` entries as text with sorted keys. The server writes every object's keys
    sorted (`serde_json` without `preserve_order`: its refused routers' entry is `check, kinds,
    reason, routers, step` where Python's is `check, kinds, routers, reason, step`), so the bytes
    compared are each value's, the key order being each engine's for every entry."""
    return json.dumps([i for i in evidence.get("not_checked") or () if i.get("check") == "moe_routing"], sort_keys=True)


@pytest.mark.parametrize("name", NAMES)
def test_each_case_with_the_rust_detectors(name):
    """The server's report (lane D, wave AG) gives each case's expected values, and its entry is the
    Python one byte for byte, in the same place among the entries; the two kinds' values agree."""
    _rust_or_skip()
    case = CASES[name]
    rust = ev.build_evidence(case["given"]["records"], detectors="rust")
    python = ev.build_evidence(case["given"]["records"], detectors="python")
    assert_expected(answer(rust), case["expected"])
    assert _entries_text(rust) == _entries_text(python)
    assert answer(rust)["not_checked_checks"] == answer(python)["not_checked_checks"]
    for kind in KINDS:
        assert {k: rust["kind_checks"][kind].get(k) for k in ("value", "check", "reason")} == \
            {k: python["kind_checks"][kind].get(k) for k in ("value", "check", "reason")}, kind


# ---- The all-ranks views and the MCP tools, by section 2's rule ----------------------------------


def _write(path: Path, records: list[dict], rank: int) -> None:
    path.write_text("".join(json.dumps({**r, "rank": rank}) + "\n" for r in records))


def test_the_all_ranks_view_of_local_files_names_the_rank(tmp_path):
    from tensorquill.compare import open_run

    read = CASES["routers_that_were_read"]["given"]["records"]
    unread = CASES["every_router_recorded_with_no_tokens"]["given"]["records"]
    run_id = read[0]["run_id"]
    _write(tmp_path / f"{run_id}.rank0.jsonl", read, 0)
    _write(tmp_path / f"{run_id}.rank1.jsonl", [{**r, "run_id": run_id} for r in unread], 1)
    view = open_run(run_id, source=tmp_path)["evidence"]
    said = [(item, ranks) for item, ranks in ev.rank_not_checked(view) if item["check"] == "moe_routing"]
    routers = ["blocks.0.router", "blocks.1.router"]
    assert said == [({"check": "moe_routing", "kinds": KINDS, "routers": routers, "reason": reason(routers)}, [1])]
    for kind in KINDS:
        assert view["kind_checks"][kind]["value"] == "unavailable"
        assert view["kind_checks"][kind]["ranks"]["unavailable"] == [1]


def test_mcp_list_incidents_on_a_capture_says_it(tmp_path):
    path = tmp_path / "r.jsonl"
    _write(path, CASES["every_router_recorded_with_no_tokens"]["given"]["records"], 0)
    answer_ = mcp_server.list_incidents(mcp_server.Source(path), {})
    assert [i["routers"] for i in answer_["not_checked"] if i["check"] == "moe_routing"] == [
        ["blocks.0.router", "blocks.1.router"]]
    assert "moe_imbalance" in answer_["kinds"]["unavailable"] and "starved_experts" in answer_["kinds"]["unavailable"]
    assert answer_["run_state"]["state"] == "degraded"


def test_the_all_ranks_view_of_a_store_and_the_mcp_tools_carry_the_servers_entry(tmp_path):
    """A store's reports come from the server (lane D, wave AG): its entry, byte for byte the Python
    one, reaches the all-ranks view and the MCP tools with the rank it came from."""
    _rust_or_skip()
    from test_flight_e2e import BINARY, Server
    from tensorquill.flight import send_file

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    read = CASES["routers_that_were_read"]["given"]["records"]
    unread = CASES["every_router_recorded_with_no_tokens"]["given"]["records"]
    store = tmp_path / "store"
    server = Server(store, 0, "--flush-steps", "50", "--flush-seconds", "0.2")
    try:
        for rank, rows in ((0, read), (1, unread)):
            path = tmp_path / f"send.rank{rank}.jsonl"
            _write(path, [{**r, "run_id": "ag"} for r in rows], rank)
            assert send_file(path, server.endpoint, "ag", rank, timeout=30.0)
        folder = store / "runs" / "ag"
        deadline = time.monotonic() + 20
        while not all(ev.read_server_report(folder / f"rank={n}")["state"] == "current" for n in (0, 1)):
            assert time.monotonic() < deadline, "the server did not report on both ranks"
            time.sleep(0.05)
    finally:
        server.terminate()
    report = json.loads((folder / "rank=1" / "incidents.json").read_text())
    python = ev.build_evidence(unread, detectors="python")
    assert _entries_text(report) == _entries_text(python)
    assert not [i for i in json.loads((folder / "rank=0" / "incidents.json").read_text()).get("not_checked") or ()
                if i.get("check") == "moe_routing"]
    view = ev.all_ranks_evidence(folder)
    assert [(json.dumps(item, sort_keys=True), ranks) for item, ranks in ev.rank_not_checked(view)
            if item["check"] == "moe_routing"] == [(_entries_text(python)[1:-1], [1])]
    for kind in KINDS:
        assert view["kind_checks"][kind]["value"] == "unavailable" and view["kind_checks"][kind]["ranks"]["unavailable"] == [1]
    for answer_ in (mcp_server.get_all_ranks_status(mcp_server.Source(folder), {}),
                    mcp_server.list_incidents(mcp_server.Source(folder), {})):
        assert [i["ranks"] for i in answer_["not_checked"] if i["check"] == "moe_routing"] == ["1"]


class _CountedCounts(list):
    """A router's counts that count how often they are read through."""

    reads = 0

    def __iter__(self):
        type(self).reads += 1
        return super().__iter__()


def test_a_router_found_read_is_not_scanned_again():
    """The rule scans a router's counts only until it finds a count above 0 (wave AF-W: it summed
    every count of every record at every look, 2.2 s of a 3.6 s watcher update at 10,000 steps)."""
    records = [dict(r) for r in CASES["one_router_with_no_tokens_beside_one_that_was_read"]["given"]["records"]]
    _CountedCounts.reads = 0
    for record in records:
        record["moe"] = {name: _CountedCounts(counts) for name, counts in record["moe"].items()}
    found = ev.routers_not_read_not_checked(records)
    assert found["routers"] == ["blocks.0.router"]
    # The router read is read through on its first record only; the one never read, on every record.
    assert _CountedCounts.reads == 1 + len(records)
