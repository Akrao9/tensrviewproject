"""Lane AC2 of wave AC (Nov 21, 2026): the server's side of `CHECKPOINT_TENSOR_CONTRACT.md`,
section 4 (TrainCheck's B14), through the real server, with records sent as the SDK sends them
(`flight.send_file`, `record_to_row`).

Each of the 17 cases with `"group": "records"` in `fixtures/checkpoint/cases.json` is one rank's
file, both attempts in it, sent to a server; the rank's `incidents.json` is held to the case's
`expected`: whether `resume_incomplete` is raised, the first change's findings exactly, and the
rank's `checkpoint_tensors` exactly (null: no such key). The block is read from the server's own
report, not from `build_evidence`: `evidence.py` is lane AC1's and does not carry it yet.

The Rust tests (`server/src/detect/lane_ac2_tests.rs`) run the same cases on stored batches; this
file checks that the SDK's own encoding reaches the same rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from test_flight_e2e import BINARY, Server  # noqa: E402
from test_server_detectors import FAST, Stream  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

CASES = json.loads((Path(__file__).parent / "fixtures" / "checkpoint" / "cases.json").read_text())["cases"]
OVER_RECORDS = [(name, body) for name, body in CASES.items() if body["group"] == "records"]


def _differs(report: dict, expected: dict) -> list[str]:
    wrong = []
    found = next((i for i in report["incidents"] if i["kind"] == "resume_incomplete"), None)
    if (found is not None) != expected["fires"]:
        wrong.append(f"fires {found is not None} expected {expected['fires']}")
    if found is not None and "findings" in expected and found["changes"][0]["findings"] != expected["findings"]:
        wrong.append(f"findings {found['changes'][0]['findings']} expected {expected['findings']}")
    if report.get("checkpoint_tensors") != expected["checkpoint_tensors"]:
        wrong.append(f"checkpoint_tensors {report.get('checkpoint_tensors')} expected {expected['checkpoint_tensors']}")
    return wrong


def test_every_records_case_holds_on_the_real_server(tmp_path):
    assert len(OVER_RECORDS) == 17
    server = Server(tmp_path / "store", 0, *FAST)
    wrong = {}
    try:
        for name, body in OVER_RECORDS:
            run = name.replace("_", "-")
            rows = [{"schema_version": 1, "run_id": run, "ts": float(r["step"]), **r} for r in body["records"]]
            assert len({r["attempt"] for r in rows}) == 2, name  # both attempts in one file: judged as a resume
            report = Stream(tmp_path, run=run, server=server).send(rows)
            assert report["steps"] == len(rows), name
            if found := _differs(report, body["expected"]):
                wrong[name] = found
    finally:
        assert server.terminate() == 0
    assert wrong == {}, json.dumps(wrong, indent=1)
