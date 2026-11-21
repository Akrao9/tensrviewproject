"""Lane AD8 of wave AD (Nov 21, 2026): the server's side of `DETECTOR_RULES_CONTRACT.md`, parts 1 to
5 and 7, through the real `tensorquill-server`, with records sent as the SDK sends them
(`flight.record_to_row`).

Each case of `fixtures/rules/cases.json` over records is one rank's records, given to
`tensorquill-server detect` through `server_detect.server_report`; the server's own report is held to
the case's `expected`. It is the server's document that is read, not `build_evidence`: the Python
reader of the report (`evidence.py`) is lane AD7's and does not read these keys in this tree.
Numbers agree to one part in 1e12, everything else exactly.

What a case states that is a reader's is left out by name: the `moe_source` `words`, and the
all-ranks union of `parameters_not_judged`; for that case each rank's own list is checked. The eight
cases of the score's arithmetic have no records and are held in the Rust tests
(`server/src/detect/lane_ad8_score_tests.rs`). The three `capture_stopped_*` cases are the live
status: each case's heartbeat is sent to a running server, and the rank's `capture_stopped` in
`watchdog.json` is held to the case, before and after a restart.

The Rust tests (`server/src/detect/lane_ad8_tests.rs`, `server/src/watchdog/lane_ad8_tests.rs`) run
the same cases on stored batches; this file checks that the SDK's own encoding reaches the same rules.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

from tensorquill.server_detect import find_binary, server_report  # noqa: E402

pytestmark = pytest.mark.skipif(find_binary() is None, reason="tensorquill-server is not built")

CASES = json.loads((Path(__file__).parent / "fixtures" / "rules" / "cases.json").read_text())["cases"]
LIVE = ("capture_stopped_at_step_0", "capture_stopped_before_any_step", "capture_stopped_from_an_older_sender")
STANDING = ("flight_recorder", "traincheck")  # the files every server report says it never receives


def of(*parts: str) -> list[tuple[str, dict[str, Any]]]:
    return [(name, body) for name, body in CASES.items() if body["part"] in parts and name not in LIVE]


def differs(at: str, got: Any, expected: Any, out: list[str]) -> None:
    """Where `got` differs from `expected`: numbers to one part in 1e12, all else exactly."""
    number = (int, float)
    if isinstance(got, number) and isinstance(expected, number) and not isinstance(got, bool) \
            and not isinstance(expected, bool):
        if not math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12):
            out.append(f"{at}: {got} expected {expected}")
    elif isinstance(got, dict) and isinstance(expected, dict):
        for key in expected.keys() | got.keys():
            if key not in got:
                out.append(f"{at}.{key}: absent, expected {expected[key]}")
            elif key not in expected:
                out.append(f"{at}.{key}: {got[key]} not expected")
            else:
                differs(f"{at}.{key}", got[key], expected[key], out)
    elif isinstance(got, list) and isinstance(expected, list) and len(got) == len(expected):
        for index, (a, b) in enumerate(zip(got, expected)):
            differs(f"{at}[{index}]", a, b, out)
    elif got != expected:
        out.append(f"{at}: {got} expected {expected}")


def judged(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    run = name.replace("_", "-")
    return server_report([{"schema_version": 1, "run_id": run, **r} for r in records])


def incident(report: dict[str, Any], kind: str) -> dict[str, Any] | None:
    """The report's incident of `kind` without `class`, which every incident of a report has, and
    `scalars`, what was logged on its step, which a report adds (as `build_evidence` does) and no
    case states: the cases' incidents are the detector functions' with the contract's keys added."""
    found = next((i for i in report["incidents"] if i["kind"] == kind), None)
    if found is None:
        return None
    assert "class" in found, found
    return {k: v for k, v in found.items() if k not in ("class", "scalars")}


def entry_differs(at: str, got: dict[str, Any], expected: dict[str, Any], absent: str, out: list[str]) -> None:
    for key, value in expected.items():
        differs(f"{at}.{key}", got.get(key), value, out)
    if absent not in expected and absent in got:
        out.append(f"{at}.{absent}: {got[absent]} not expected")


def case_differs(name: str, body: dict[str, Any]) -> list[str]:
    expected, part, wrong = body["expected"], body["part"], []
    if "records_by_rank" in body:
        # The union is the Python reader's (`combine_rank_evidence`); each rank names its own.
        by_rank = expected["kind_checks"]["stalled_weights"]["parameters_not_judged_by_rank"]
        for rank, records in body["records_by_rank"].items():
            report = judged(name, [{**r, "rank": int(rank)} for r in records])
            differs(f"rank {rank} parameters_not_judged",
                    report["kind_checks"]["stalled_weights"].get("parameters_not_judged"), by_rank[rank], wrong)
        return wrong
    report = judged(name, body["records"])
    checks = report["kind_checks"]
    if part == "score_records":
        differs("incident", incident(report, expected["incident"]["kind"]), expected["incident"], wrong)
    elif part == "activation_drift":
        differs("incident", incident(report, "activation_drift"), expected["incident"], wrong)
        differs("records_needed", checks["activation_drift"]["records_needed"], expected["records_needed"], wrong)
        differs("enough_history", checks["activation_drift"]["enough_history"], expected["enough_history"], wrong)
    elif part == "moe_imbalance":
        check = checks["moe_imbalance"]
        differs("incident", incident(report, "moe_imbalance"), expected["incident"], wrong)
        differs("starting_state", check.get("starting_state", []), expected["starting_state"], wrong)
        if check.get("starting_state") == []:
            wrong.append("starting_state is written only when a router is in it")
        if check.get("starting_state") and check["enough_history"] is not False:
            wrong.append(f"a router in its starting state, and enough_history {check['enough_history']}")
    elif part == "evidence":
        if "incident" in expected:
            differs("incident", incident(report, "update_spike"), expected["incident"], wrong)
        for kind, want in expected.get("kind_checks", {}).items():
            entry_differs(f"kind_checks.{kind}", checks[kind], want, "parameters_not_judged", wrong)
    elif name.startswith("moe_source"):
        differs("moe_source", report.get("moe_source"), expected["moe_source"], wrong)  # `words` is a reader's
    else:
        standing = [i for i in report["not_checked"] if i["check"] in STANDING]
        assert len(standing) == 2, report["not_checked"]
        differs("not_checked", [i for i in report["not_checked"] if i["check"] not in STANDING],
                expected["not_checked"], wrong)
        for kind, want in expected["kind_checks"].items():
            entry_differs(f"kind_checks.{kind}", checks[kind], want, "modules_not_judged", wrong)
    return wrong


@pytest.mark.parametrize("part,count", [("score_records", 4), ("activation_drift", 9), ("moe_imbalance", 11),
                                        ("evidence", 4), ("wire", 7)])
def test_every_case_over_records_holds_on_the_real_server(part, count):
    cases = of(part)
    assert len(cases) == count
    wrong = {name: why for name, body in cases if (why := case_differs(name, body))}
    assert wrong == {}, json.dumps(wrong, indent=1)


# --- the live status: the step capture stopped at ------------------------------------------------

def _stopped(status: dict[str, Any]) -> dict[str, Any]:
    from test_lane_w2_attempt_order import rank0

    entry = dict(rank0(status)["capture_stopped"])
    assert isinstance(entry.pop("seconds"), (int, float)), entry  # how long ago: no case states it
    return entry


def _live_case(name: str, store: Path) -> list[str]:
    """One case's heartbeat sent to a running server, as written, after a beat and a few steps of
    its attempt; what differs in the rank's `capture_stopped` before and after a restart."""
    flight = pytest.importorskip("pyarrow.flight")
    from test_lane_w2_attempt_order import Server, Stream, rank0, rec, wait_status

    body = CASES[name]
    heartbeat, expected = body["heartbeat"], body["expected"]["status"]["capture_stopped"]
    run_id, attempt = heartbeat["run_id"], heartbeat["attempt"]

    def send(server: Server, payload: dict[str, Any]) -> None:
        client = flight.FlightClient(server.endpoint)
        try:
            action = flight.Action("heartbeat", json.dumps({**payload, "interval_seconds": 0.2}).encode())
            for _ in client.do_action(action, flight.FlightCallOptions(timeout=10.0)):
                pass
        finally:
            client.close()

    def ready(status: dict[str, Any]) -> bool:
        return rank0(status).get("capture_stopped") is not None

    wrong: list[str] = []
    server = Server(store)
    try:
        send(server, {k: v for k, v in heartbeat.items() if not k.startswith("capture_stopped")} | {"work": 0})
        stream = Stream(server.endpoint, attempt, run_id=run_id)
        stream.send([rec(attempt, step, run_id=run_id) for step in range(3)])
        send(server, heartbeat | {"work": 3})
        status = wait_status(store, ready, run_id=run_id)
        differs(f"{name}", _stopped(status), expected, wrong)
        stream.close()
    finally:
        server.kill()
    again = Server(store)
    try:
        status = wait_status(store, lambda s: s["written_at_ms"] >= again.started_ms and ready(s), run_id=run_id)
        assert status["written_at_ms"] >= again.started_ms, status
        differs(f"{name} after a restart", _stopped(status), expected, wrong)
    finally:
        again.stop()
    return wrong


def test_the_step_capture_stopped_at_reaches_the_status_and_survives_a_restart(tmp_path):
    """The three `capture_stopped_*` cases: step 0, null (none committed) and no `step` key (a sender
    older than the key) stay three different things in the status, and over a restart."""
    from test_lane_w2_attempt_order import BINARY

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    wrong = [why for name in LIVE for why in _live_case(name, tmp_path / name)]
    assert wrong == [], "\n".join(wrong)
