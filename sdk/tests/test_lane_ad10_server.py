"""Lane AD10 of wave AD (Nov 21, 2026): the server's side of `DETECTOR_RULES_CONTRACT.md`, part 6, the
four measurements `gradient_trend`, `attention_trends`, `update_trends` and `weight_jumps`, through the
real `tensorquill-server`, with records sent as the SDK sends them (`flight.record_to_row`).

Each of the ten `measurements` cases of `fixtures/rules/cases.json` is one rank's records. They are
given to `tensorquill-server detect` (`server_detect.server_report`), and streamed to a running
server, whose `incidents.json` is read after a first stretch, and again after a restart and the last
record, which the server judges by rebuilding the rank from its store. An `expected` of null is a
block that must be absent from the report: a missing key, never null. The live server keeps every
series, so its report's four blocks are `detect`'s on the same records. A run streamed with a
restart and a resumed attempt is held to `detect` and to the page's reference functions. Numbers agree to one part in
1e12, everything else exactly.

It is the server's document that is read, not `build_evidence`: the Python reader of the report is
lane AD7's. The Rust tests (`server/src/detect/lane_ad10_tests.rs`) run the same cases on stored
batches, and pin what no case states.
"""

from __future__ import annotations

import importlib.util
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


def test_every_measurement_case_holds_in_a_live_servers_report_across_a_restart(tmp_path):
    """The live path appends each stored part to the rank's detectors, and rebuilds them from the
    store after a restart. Both keep every series (the author's decision: option A), so the live
    report's four blocks are `detect`'s on the same records, with the same absence rules, and the
    case's once it holds the whole case."""
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
            first = stream.send(records[:-1])
            detect = server_report(stream.sent)
            out += [f"{name}: the first stretch differs from detect on {block}" for block in BLOCKS
                    if first.get(block) != detect.get(block)]
    assert server.terminate() == 0
    server = Server(tmp_path / "store", 0, *FAST)
    for name, body in CASES.items():
        stream = streams[name]
        stream.server = server
        run = name.replace("_", "-")
        last = {"schema_version": 1, "run_id": run, "rank": 0, "attempt": 1, **body["records"][-1]}
        document = stream.send([last])
        # A part after the restart rebuilds the rank from the store; a one-record case has no first part.
        assert document["rebuilds"] == 1, (name, document["rebuilds"])
        out += case_differs(name, document)
        detect = server_report(stream.sent)
        out += [f"{name}: detect differs from the live report on {block}" for block in BLOCKS
                if document.get(block) != detect.get(block)]
    assert server.terminate() == 0
    assert out == []


def _reference():
    spec = importlib.util.spec_from_file_location("make_cases", Path(__file__).parent / "fixtures" / "rules" / "make_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workload(run: str, attempt: int, steps: range, scale: float) -> list[dict[str, Any]]:
    """A run that holds every block: a total norm on every step, update sizes, and a weight check
    and a probe of two attention modules every ten steps. `scale` changes every value."""
    out = []
    for step in steps:
        wobble = 1.0 + (step * 37 % 21) / 100
        record = {"schema_version": 1, "run_id": run, "rank": 0, "attempt": attempt, "step": step, "ts": float(step),
                  "grad_norm": scale * (1.0 + step / 50) * wobble,
                  "update_rms": {"w": scale * 1e-3 * wobble, "v": scale * 1e-3 / (1.0 + step / 40)}}
        if step % 10 == 0:
            record["weight_norms"] = {"w": scale * (5.0 + step / 10), "v": 3.0 + step % 30}
            record["probe"] = {"attention": {"a.0": {"entropy_share": [0.5], "max_logit": [scale * (2.0 + step / 20)]},
                                             "a.1": {"entropy_share": [0.5], "max_logit": [scale * 3.0 * wobble]}}}
        out.append(record)
    return out


def test_a_live_servers_blocks_after_a_restart_and_a_resume_are_those_of_the_resolved_run(tmp_path):
    """A run sent, the server stopped and started again on the same store: the rebuilt report's four
    blocks are `detect`'s on the same records. Then a second attempt restarts at step 80, below the
    last step 119, with other values: the blocks are those of the resolved run, as `detect` gives
    them and as the page's reference functions give them on the records `build_evidence` judges."""
    from tensorquill.evidence import resolve_attempts

    run = "ad10-resume"
    stream = Stream(tmp_path, *FAST, run=run)
    stream.send(workload(run, 1, range(0, 60), 1.0))
    stream.restart()
    after_restart = stream.send(workload(run, 1, range(60, 120), 1.0))
    assert after_restart["rebuilds"] == 1
    assert all(block in after_restart for block in BLOCKS), sorted(after_restart)
    detect = server_report(stream.sent)
    assert {b: after_restart[b] for b in BLOCKS} == {b: detect[b] for b in BLOCKS}

    resumed = stream.send(workload(run, 2, range(80, 150), 1.5))
    assert resumed["attempt"] == 2 and resumed["rebuilds"] == 2  # a new attempt rebuilds from the store
    detect = server_report(stream.sent)
    assert {b: resumed.get(b) for b in BLOCKS} == {b: detect.get(b) for b in BLOCKS}
    judged = sorted(resolve_attempts(stream.sent), key=lambda r: int(r["step"]))
    assert [r["step"] for r in judged] == list(range(0, 80)) + list(range(80, 150))
    reference = _reference()
    out: list[str] = []
    for block in BLOCKS:
        differs(block, resumed[block], getattr(reference, block)(judged), out)
    assert out == []
    assert resumed["gradient_trend"]["points"] == 150 and resumed["gradient_trend"]["last_step"] == 149
    assert resumed["gradient_trend"] != after_restart["gradient_trend"]
    assert stream.server.terminate() == 0
