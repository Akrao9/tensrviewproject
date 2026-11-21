"""Lane AB2 of wave AB (Nov 20, 2026): the server's side of the two contract pages, through the
real server, with records sent as the SDK sends them (`flight.send_file`, `record_to_row`).

- `STEP_AND_EVIDENCE_CONTRACT.md`, sections 1 and 2: each case that holds records is one rank's
  file, sent to a server, and the rank's `incidents.json` is held to the case's expectation:
  `step_skipped_by` and `grads_not_read` travel in `extra_json`, and the server reads them there.
- `REPLICA_OBSERVATION_CONTRACT.md`, section 4a (D1): cases of `attribution_cases.json` sent as one
  file per rank, and the run's `watchdog.json` holds the case's attribution under `replica`, and
  keeps it over a restart of the server.

The Rust tests (`server/src/detect/lane_ab2_tests.rs`) run every case on stored batches; this file
checks that the SDK's own encoding reaches the same rules.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill.flight import send_file  # noqa: E402
from test_flight_e2e import BINARY, Server  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

FIXTURES = Path(__file__).parent / "fixtures"
BEHAVIOUR = json.loads((FIXTURES / "behaviour" / "cases.json").read_text())["cases"]
ATTRIBUTION = json.loads((FIXTURES / "replica" / "attribution_cases.json").read_text())
OVER_RECORDS = [
    (name, body) for name, body in BEHAVIOUR.items() if "records" in body or "records" in body.get("given", {})
]
DETECTORS = {
    "detect_grad_spikes": "grad_spike",
    "detect_non_finite": "non_finite",
    "detect_loss_scale_trouble": "loss_scale_trouble",
    "detect_stalled_weights": "stalled_weights",
}
FAST = ("--flush-steps", "50", "--flush-seconds", "0.2", "--watchdog-seconds", "0.5")


def _wait(what: str, read, seconds: float = 20.0):
    deadline = time.monotonic() + seconds
    while True:
        try:
            got = read()
            if got is not None:
                return got
        except (FileNotFoundError, ValueError, KeyError):
            pass
        assert time.monotonic() < deadline, f"the server did not write {what}"
        time.sleep(0.05)


def _report(tmp_path: Path, records: list[dict]) -> dict:
    """One rank's records sent to a server, and the rank's report once it covers them."""
    run = "ab2"
    store = tmp_path / "store"
    file = tmp_path / f"{run}.rank0.jsonl"
    rows = [{"schema_version": 1, "run_id": run, "rank": 0, "attempt": 1, "ts": float(r["step"]), **r} for r in records]
    file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    server = Server(store, 0, *FAST)
    try:
        assert send_file(file, server.endpoint, run, timeout=30.0)
        path = store / "runs" / run / "rank=0" / "incidents.json"

        def covered():
            document = json.loads(path.read_text(encoding="utf-8"))
            return document if document["last_step"] == rows[-1]["step"] else None

        return _wait("the rank's report", covered)
    finally:
        assert server.terminate() == 0


def _incident(report: dict, kind: str) -> dict | None:
    return next((i for i in report["incidents"] if i["kind"] == kind), None)


@pytest.mark.parametrize("name,body", OVER_RECORDS, ids=[n for n, _ in OVER_RECORDS])
def test_each_case_over_records_holds_on_the_real_server(tmp_path, name, body):
    records = body.get("records") or body["given"]["records"]
    report = _report(tmp_path, records)
    expected = body["expected"]
    if "detector" in body:
        kind = DETECTORS[body["detector"]]
        found = _incident(report, kind)
        wanted = {k: v for k, v in expected.items() if k != "fires"}
        if expected["fires"]:
            assert found is not None, report["incidents"]
            assert {k: found.get(k) for k in wanted} == wanted
        else:
            assert found is None, found
    for kind in expected.get("incident_kinds_present", ()):
        found = _incident(report, kind)
        assert found is not None, report["incidents"]
        carries = expected.get("incident_carries", {})
        assert {k: found.get(k) for k in carries} == carries
    for kind in expected.get("incident_kinds_absent", ()):
        assert _incident(report, kind) is None, report["incidents"]
    for kind, value in expected.get("kind_values", {}).items():
        assert report["kind_checks"][kind]["value"] == value, report["kind_checks"][kind]
    for wanted in expected.get("not_checked", ()):
        assert any(all(entry.get(k) == v for k, v in wanted.items()) for entry in report["not_checked"]), \
            report["not_checked"]
    for key, value in expected.get("evidence_carries", {}).items():
        assert report.get(key) == value
    for key in expected.get("evidence_lacks", ()):
        assert key not in report


def test_every_case_over_records_is_run_here():
    assert len(OVER_RECORDS) == 30


# --- D1 ------------------------------------------------------------------------------------------

D1_CASES = ["three_ranks_two_to_one", "a_set_aside_later_attempt_does_not_vote", "one_clean_step_and_one_attributed"]


def _send_ranks(tmp_path: Path, server: Server, records: list[dict]) -> None:
    """One file per rank, each rank's records in the case's order (contract 1's), sent in turn."""
    for rank in sorted({r["rank"] for r in records}):
        file = tmp_path / f"r1.rank{rank}.jsonl"
        file.write_text("".join(json.dumps(r) + "\n" for r in records if r["rank"] == rank))
        assert send_file(file, server.endpoint, "r1", rank, timeout=30.0)


def _replica(store: Path, steps: int) -> dict | None:
    status = json.loads((store / "runs" / "r1" / "watchdog.json").read_text(encoding="utf-8"))
    replica = status.get("replica")
    complete = [e for e in (replica or {}).get("steps", ()) if e["coverage"] == "complete"]
    return replica if replica is not None and len(complete) >= steps else None


@pytest.mark.parametrize("case", D1_CASES)
def test_the_live_server_writes_attribution_and_keeps_it_over_a_restart(tmp_path, case):
    body = ATTRIBUTION["cases"][case]
    expected = body["expected"]
    complete = sum(1 for e in expected["coverage"] if e["coverage"] == "complete")
    store = tmp_path / "store"
    server = Server(store, 0, *FAST)
    try:
        _send_ranks(tmp_path, server, body["records"])
        replica = _wait("the replica result", lambda: _replica(store, complete))
        assert replica.get("attribution", []) == expected["attribution"]
        assert len(replica["replica_drift"]) == expected["incidents"]
    finally:
        assert server.terminate() == 0
    written = json.loads((store / "runs" / "r1" / "watchdog.json").read_text(encoding="utf-8"))["replica"]
    again = Server(store, 0, *FAST)
    try:
        # The restarted server takes up what it wrote, and writes it again unchanged. (The stopped
        # server's last status says it is not running; the restarted one's first says it is.)
        def rewritten():
            status = json.loads((store / "runs" / "r1" / "watchdog.json").read_text(encoding="utf-8"))
            return status if status.get("server_running") is True else None

        status = _wait("the status after the restart", rewritten)
        assert status["replica"].get("attribution", []) == written.get("attribution", []) == expected["attribution"]
    finally:
        assert again.terminate() == 0
