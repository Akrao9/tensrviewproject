"""Seams of wave AD: where two lanes meet and neither could test the join in its own tree.

1. The step capture stopped at: lane AD4 gave the heartbeat `stop_capture(reason, step=)`, and the
   one call in `capture.py`, lane AD3's file, is made here.
2. `divergence.py` (lane AD5) reads `evidence.run_of`, the public name lane AD1 added.
3. `runs._judged_to_step` (lane AD5) reads the step from the keys lane AD1's watchers write.
4. `moe_source` from an actual server report: lane AD8 writes it in the report and lane AD7's reader
   did not pass it on.
5. Rows of distinct steps sent out of order inside one part: the live report's four measurement
   blocks against offline Rust and Python on the records the server holds (the author's
   requirement for the trial).

4 and 5 need the built server.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import divergence, evidence as ev, runs

from test_flight_e2e import BINARY, Server, stored_records

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
BLOCKS = ev.MEASUREMENT_BLOCKS


# --- 1 ------------------------------------------------------------------------------------------

class Beat:
    def __init__(self) -> None:
        self.told: list[tuple[str, Any]] = []

    def stop_capture(self, reason, step="not given"):
        self.told.append((reason, step))


@pytest.mark.parametrize("steps,expected", [(0, None), (1, 0), (3, 2)])
def test_a_stopped_session_tells_its_heartbeat_the_last_step_it_committed(tmp_path, steps, expected):
    """Fail before the seam: the call gave no step, so the beat carried null even after steps."""
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tensorquill.init(model, run_id="seam-stop", every=1, sink=tensorquill.JsonlSink(tmp_path / "seam.rank0.jsonl"))
    try:
        for _ in range(steps):
            opt.zero_grad()
            model(torch.ones(2, 4)).sum().backward()
            opt.step()
        beat = session._sink.heartbeat = Beat()
        session._stop(RuntimeError("the sink failed"), reason="sink_error")
        assert beat.told == [("sink_error", expected)]  # step 0 is a step; none committed is None
    finally:
        session.close()


# --- 2 ------------------------------------------------------------------------------------------

def test_divergence_reads_the_public_name():
    source = inspect.getsource(divergence)
    assert "_run_of" not in source and "run_of(" in source
    assert ev.run_of({"run_id": "a"}) == ev._run_of({"run_id": "a"})


# --- 3 ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("document,expected", [
    ({"freshness": {"current": False, "judged_to_step": 40, "records_to_step": 60, "why": "records_newer"},
      "detection": {"source": "server", "state": "current", "report_last_step": 99}}, 40),
    ({"freshness": {"current": False, "judged_to_step": None, "records_to_step": 60, "why": "nothing_judged"}}, None),
    ({"detection": {"source": "server", "state": "current", "report_last_step": 12}}, 12),
    ({"detection": {"source": "server", "state": "waiting", "report_last_step": 12}}, 12),  # the last current report
    ({"detection": {"source": "python", "report_last_step": 5, "stored_last_step": 30}}, 30),  # judged here, from the parts
    ({"detection": {"source": "python"}}, None),
    ({}, None),
    ({"detection": {"source": "server", "report_last_step": True}}, None),
])
def test_runs_takes_the_judged_step_from_what_the_watchers_write(document, expected):
    """Fail before the seam on the rows with `freshness`, a waiting server report and the Python
    fallback: `runs` read only a current server report's step."""
    assert runs._judged_to_step(document) == expected


# --- 4 and 5: the real server ----------------------------------------------------------------------

RUN = "seam-live"


def _send(endpoint: str, records: list[dict]) -> None:
    """One `DoPut` stream holding one batch: the records in the order given, as one part."""
    import pyarrow as pa
    import pyarrow.flight as flight
    from tensorquill.flight import records_to_batch

    client = flight.FlightClient(endpoint)
    try:
        batch = records_to_batch(records)
        writer, reader = client.do_put(flight.FlightDescriptor.for_path("runs", RUN, "rank=0"), batch.schema,
                                       flight.FlightCallOptions(timeout=60.0))
        steps = [r["step"] for r in records]
        metadata = {"schema_version": 2, "run_id": RUN, "rank": 0, "end_byte_offset": 100 * len(records),
                    "min_step": min(steps), "max_step": max(steps), "sender_attempt": 1}
        writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
        writer.done_writing()
        while reader.read() is not None:
            pass
        writer.close()
    finally:
        client.close()


def _live(tmp_path: Path, records: list[dict]) -> tuple[dict, list[dict], Path]:
    """The live server's report for rank 0 after it was sent `records` as one part and stopped, the
    records it stored, and the rank's folder."""
    store = tmp_path / "store"
    server = Server(store)
    try:
        _send(server.endpoint, records)
    finally:
        server.terminate()  # writes what it holds, and the report
    folder = store / "runs" / RUN / "rank=0"
    return json.loads((folder / "incidents.json").read_text()), stored_records(store, RUN), folder


def _row(step: int, **more) -> dict:
    return {"schema_version": 1, "run_id": RUN, "rank": 0, "attempt": 1, "step": step, "ts": 1.0 + step, **more}


@needs_server
def test_moe_source_reaches_a_reader_of_an_actual_server_report(tmp_path):
    """Fail before the seam: the server's report held `moe_source` and the evidence built from it did not."""
    source = {"r": {"form": "indices", "point": "unknown"}}
    records = [_row(i, moe={"r": [32] * 8}, moe_top_k={"r": 1}, **({"moe_source": source} if i == 0 else {})) for i in range(12)]
    report, stored, folder = _live(tmp_path, records)
    assert len(stored) == 12 and report.get("moe_source") == source  # the server's own document
    evidence = ev._report_evidence(folder, ev.read_server_report(folder), ev.read_watchdog(folder.parent / "watchdog.json"))
    assert evidence.get("moe_source") == source
    assert ev.moe_source_words(evidence) == ev.moe_source_words({"moe_source": source}) != {}


LATE = [step for step in range(60) if step % 5 == 2]  # sent three places after their own


def _out_of_order() -> list[dict]:
    """60 distinct steps in one part. Each has a gradient norm and an update size, and every second
    step a weight check and an attention probe, all rising gently with the step. Every fifth row is
    sent three places late, and carries values far from its neighbours' (a gradient norm of a
    million, an update a billion times smaller, a weight norm of a billion, a logit of a million):
    whoever reads those rows, or reads the rows in another order, gives other blocks."""
    rows = []
    for step in range(60):
        late = step in LATE
        level = 1.0 + 0.05 * step
        row = _row(step, grad_norm=1e6 if late else level,
                   update_rms={"w": 1e-3 * level, "v": 1e-12 if late else 1e-3 / level})
        if step % 2 == 0:
            row["weight_norms"] = {"w": 1e9 if late else 5.0 + 0.01 * step, "v": 2.0 + 0.5 * step}
            row["probe"] = {"attention": {"attn.0": {"entropy_share": [0.5, 0.5], "max_logit": [1e6 if late else 4.0 * 1.05 ** step, 1.0]},
                                          "attn.1": {"entropy_share": [0.5, 0.5], "max_logit": [9.0, 1.0]}}}
        rows.append(row)
    order = [step for step in range(60) if step not in LATE]
    for step in LATE:
        order.insert(order.index(step + 1) + 3 if step + 1 in order else len(order), step)
    by_step = {row["step"]: row for row in rows}
    return [by_step[step] for step in order]


def _blocks(document: dict) -> dict:
    return {name: document.get(name) for name in BLOCKS}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


@needs_server
def test_steps_sent_out_of_order_in_one_part_give_the_live_report_the_offline_blocks(tmp_path):
    """The author's requirement for the trial. The server takes the part, and its live report's
    four measurement blocks are those of offline Rust and of Python on the records it holds.

    What the server does with such a part, pinned here because the comparison rests on it: a row
    whose step is not above the last step it kept for the attempt is not stored (the rule that
    drops a replayed row). So the rows that arrive late are left out, and what it holds is in
    step order."""
    from tensorquill.server_detect import server_report

    sent = _out_of_order()
    arrival = [row["step"] for row in sent]
    assert sorted(arrival) == list(range(60)) and arrival != sorted(arrival)       # distinct steps, out of order
    assert all(arrival.index(step) > arrival.index(step + 1) for step in LATE)     # each late row after its successor
    report, stored, _ = _live(tmp_path, sent)
    kept = [int(r["step"]) for r in stored]
    assert kept == [step for step in range(60) if step not in LATE], kept            # the late rows were not stored
    live = _blocks(report)
    offline = _blocks(server_report(stored))            # `tensorquill-server detect` on what the server holds
    python = _blocks(ev.build_evidence(stored, detectors="python"))
    assert all(python[name] is not None for name in BLOCKS), python                 # every block is exercised
    assert _same(offline, python), {"offline": offline, "python": python}
    assert _same(live, offline), {"live": live, "offline": offline}
    # And the choice of values does bite: the blocks of every row sent, in step order or as they
    # arrived, are other ones, so a live path that had read a late row would not have agreed.
    everything = _blocks(ev.build_evidence(sorted(sent, key=lambda r: r["step"]), detectors="python"))
    as_sent = _blocks(ev.build_evidence([dict(row, step=at) for at, row in enumerate(sent)], detectors="python"))
    for name in BLOCKS:
        assert not _same(everything[name], python[name]), name
        assert not _same(as_sent[name], python[name]), name
