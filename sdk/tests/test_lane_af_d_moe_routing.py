"""Lane D, wave AG (Nov 21, 2026): section 8 of `WAVE_AF_CONTRACT.md` in the Rust detectors, offline
(`tensorquill-server detect`) and in the live report (`incidents.json`, and the run state of
`watchdog.json`). A router the records name whose counts are 0 on every record was not read.

The three cases are `fixtures/wave_af/cases_ag.json`'s. The `not_checked` entry is lane C's, matched
value for value: `{"check": "moe_routing", "kinds": ["moe_imbalance", "starved_experts"],
"routers": [...], "reason": ...}`, the routers sorted by name, the reason one clause per router joined
by "; ". The server writes JSON objects with their keys in sorted order, as every entry of its reports;
the values are the same.

Fails before the fix: no such entry, and the kinds `not_configured`, as each case's `today` says.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
fl = pytest.importorskip("pyarrow.flight")

from tensorquill.flight import records_to_batch  # noqa: E402
from tensorquill.server_detect import records_stream  # noqa: E402

from test_flight_e2e import BINARY, Server  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_af" / "cases_ag.json").read_text())["cases"]
KINDS = ("moe_imbalance", "starved_experts")


def _reason(routers):
    return "; ".join(f"router {name} recorded no tokens on any step: its routing was not read" for name in routers)


def _offline(records) -> dict:
    done = subprocess.run([str(BINARY), "detect"], input=records_stream(records), capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _live(tmp_path: Path, records) -> tuple[dict, dict]:
    store = tmp_path / "store"
    run, last = records[0]["run_id"], records[-1]["step"]
    server = Server(store, 0, "--flush-steps", "1", "--watchdog-seconds", "0.2")
    incidents = store / "runs" / run / "rank=0" / "incidents.json"
    status_file = store / "runs" / run / "watchdog.json"
    try:
        client = fl.FlightClient(server.endpoint)
        batch = records_to_batch(records)
        writer, reader = client.do_put(fl.FlightDescriptor.for_path("runs", run, "rank=0"), batch.schema)
        metadata = {"schema_version": 1, "run_id": run, "rank": 0, "min_step": records[0]["step"], "max_step": last,
                    "end_byte_offset": 100, "world_size": 1}
        writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
        assert json.loads(reader.read().to_pybytes())["end_byte_offset"] == 100
        deadline = time.monotonic() + 30
        while True:
            try:
                report = json.loads(incidents.read_text())
                status = json.loads(status_file.read_text())
                if report["last_step"] == last and status.get("run_state", {}).get("state") not in (None, "unknown"):
                    break
            except (OSError, ValueError):
                pass
            assert time.monotonic() < deadline, "no report and run state for the last step"
            time.sleep(0.05)
        writer.done_writing(); writer.close(); client.close()
    finally:
        server.terminate()
    return report, status


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_router_recorded_with_no_tokens_offline_and_live(name, tmp_path):
    case = CASES[name]
    records, expected = case["given"]["records"], case["expected"]
    offline = _offline(records)
    live, status = _live(tmp_path, records)
    for where, report in (("detect", offline), ("live", live)):
        entries = [item for item in report["not_checked"] if item["check"] == "moe_routing"]
        routers = expected.get("not_checked_routers")
        if routers is None:                                      # the pin: as on the base
            assert entries == [], (where, entries)
            for kind in KINDS:
                assert report["kind_checks"][kind]["value"] == expected[kind], (where, kind)
            assert sorted({i["kind"] for i in report["incidents"]}) == expected["incident_kinds"], where
            continue
        assert entries == [{"check": "moe_routing", "kinds": list(KINDS), "routers": routers, "reason": _reason(routers)}], (where, entries)
        for kind in KINDS:
            wanted = expected.get(kind, case["today"][kind])     # the partial case: judged, as on the base
            assert report["kind_checks"][kind]["value"] == wanted, (where, kind, report["kind_checks"][kind])
            if wanted == "unavailable":
                assert (report["kind_checks"][kind]["check"], report["kind_checks"][kind]["reason"]) == ("moe_routing", _reason(routers))
        if "incident_kinds" in case["today"] and expected.get(KINDS[0]) is None:
            assert sorted({i["kind"] for i in report["incidents"]}) == case["today"]["incident_kinds"], where
    # The live report is detect's, but for the key only a live server has.
    assert {k: v for k, v in live.items() if k != "rebuilds"} == {k: v for k, v in offline.items() if k != "rebuilds"}
    state = status["run_state"]["state"]
    if expected.get("run_state_is_not"):
        assert state != expected["run_state_is_not"], status["run_state"]
    print(name, "live run state:", state, [r.get("code") for r in status["run_state"].get("reasons", [])])
