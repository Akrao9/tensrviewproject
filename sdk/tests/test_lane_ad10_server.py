"""Lane AD10 of wave AD (Nov 21, 2026): the server's side of `DETECTOR_RULES_CONTRACT.md`, part 6, the
four measurements `gradient_trend`, `attention_trends`, `update_trends` and `weight_jumps`, through the
real `tensorquill-server`, with records sent as the SDK sends them (`flight.record_to_row`).

Each of the ten `measurements` cases of `fixtures/rules/cases.json` is one rank's records. They are
given to `tensorquill-server detect` (`server_detect.server_report`), and streamed to a running
server, whose `incidents.json` is read after a first stretch, and again after a restart and the last
record, which the server judges by rebuilding the rank from its store. An `expected` of null is a
block that must be absent from the report: a missing key, never null. The live server's detectors
keep no trend series, so its report has `update_trends` and `weight_jumps` and neither trend. Numbers agree to one part in
1e12, everything else exactly.

It is the server's document that is read, not `build_evidence`: the Python reader of the report is
lane AD7's. The Rust tests (`server/src/detect/lane_ad10_tests.rs`) run the same cases on stored
batches, and pin what no case states.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill.server_detect import find_binary, server_report  # noqa: E402
from test_flight_e2e import Server  # noqa: E402
from test_server_detectors import FAST, Stream  # noqa: E402

pytestmark = pytest.mark.skipif(find_binary() is None, reason="tensorquill-server is not built")

CASES = {name: body for name, body in json.loads(
    (Path(__file__).parent / "fixtures" / "rules" / "cases.json").read_text())["cases"].items()
    if body["part"] == "measurements"}
BLOCKS = ("gradient_trend", "attention_trends", "update_trends", "weight_jumps")


def differs(at: str, got: Any, expected: Any, out: list[str]) -> None:
    """Where `got` differs from `expected`: numbers to one part in 1e12, all else exactly, and an
    integer stays an integer."""
    number = (int, float)
    if isinstance(got, number) and isinstance(expected, number) and not isinstance(got, bool) \
            and not isinstance(expected, bool):
        if not math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12):
            out.append(f"{at}: {got} expected {expected}")
        if isinstance(got, int) != isinstance(expected, int):
            out.append(f"{at}: {got!r} expected {expected!r} (integer or not)")
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


def case_differs(name: str, report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for block, expected in CASES[name]["expected"].items():
        if expected is None:
            if block in report:
                out.append(f"{name}: {block} is {report[block]}, expected absent")
        elif block not in report:
            out.append(f"{name}: {block} absent, expected {expected}")
        else:
            differs(f"{name}: {block}", report[block], expected, out)
    return out


@pytest.mark.parametrize("block,count", [("gradient_trend", 4), ("attention_trends", 2),
                                         ("update_trends", 2), ("weight_jumps", 2)])
def test_every_measurement_case_holds_in_the_report_of_tensorquill_server_detect(block, count):
    names = [name for name, body in CASES.items() if block in body["expected"]]
    assert len(names) == count
    out: list[str] = []
    for name in names:
        run = name.replace("_", "-")
        out += case_differs(name, server_report([{"schema_version": 1, "run_id": run, **r} for r in CASES[name]["records"]]))
    assert out == []


TRENDS = ("gradient_trend", "attention_trends")  # the series grow with the run: `detect` only
BOUNDED = ("update_trends", "weight_jumps")


def live_differs(name: str, document: dict[str, Any], detect: dict[str, Any], *, whole: bool) -> list[str]:
    """A live report has neither trend, and the bounded blocks as `detect` gives them on the same
    records and, once it covers the `whole` case, as the case gives them."""
    out = [f"{name}: the live report has {block}" for block in TRENDS if block in document]
    if whole and any(block in CASES[name]["expected"] for block in BOUNDED):
        out += case_differs(name, {block: document[block] for block in BOUNDED if block in document})
    out += [f"{name}: {block} differs from detect's" for block in BOUNDED if document.get(block) != detect.get(block)]
    return out


def test_a_live_servers_report_has_the_bounded_blocks_and_no_trend_across_a_restart(tmp_path):
    """The live path appends each stored part to the rank's detectors, and rebuilds them from the
    store after a restart. Those detectors keep no trend series (their memory would grow with the
    run), so the report has no `gradient_trend` and no `attention_trends`, never a shorter one; its
    `update_trends` and `weight_jumps` are the case's and `detect`'s, either way."""
    assert len(CASES) == 10
    server = Server(tmp_path / "store", 0, *FAST)
    streams: dict[str, Stream] = {}
    out: list[str] = []
    for name, body in CASES.items():
        run = name.replace("_", "-")
        records = [{"schema_version": 1, "run_id": run, "rank": 0, "attempt": 1, **r} for r in body["records"]]
        (tmp_path / name).mkdir()
        stream = streams[name] = Stream(tmp_path / name, *FAST, run=run, server=server)
        stream.store = tmp_path / "store"
        if len(records) > 1:
            # The first stretch, appended part by part.
            out += live_differs(name, stream.send(records[:-1]), server_report(stream.sent), whole=False)
    assert server.terminate() == 0
    server = Server(tmp_path / "store", 0, *FAST)
    present = 0
    for name, body in CASES.items():
        stream = streams[name]
        stream.server = server
        run = name.replace("_", "-")
        last = {"schema_version": 1, "run_id": run, "rank": 0, "attempt": 1, **body["records"][-1]}
        document = stream.send([last])
        # A part after the restart rebuilds the rank from the store; a one-record case has no first part.
        assert document["rebuilds"] == 1, (name, document["rebuilds"])
        out += live_differs(name, document, server_report(stream.sent), whole=True)
        present += sum(block in document for block in BOUNDED)
    assert server.terminate() == 0
    assert present == 2  # the two cases that expect a bounded block
    assert out == []
