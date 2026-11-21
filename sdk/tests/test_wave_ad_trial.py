"""Joined checks of wave AD's trial tree: what no lane could run in its own tree, and what the
author asked the trial to show.

1. Both engines, through `build_evidence`, on every rule case that gives records.
2. RS10 joined: the real server delivers the refusal and the production `FlightSender` resumes at
   the right byte. The sender builds every batch from one schema, so the refused batch is made by
   giving it, for that one batch, a schema whose `step` column is text; everything else is the
   production path on both sides.
3. A Rust-backed `--watch` on a file gives, at every look, what a fresh judgment of the file gives,
   through a step appended, a new attempt, late rows of an old attempt and an attempt that replaces
   judged steps, and again under another threshold.
4. A long-lived all-ranks watcher on a live server's store gives, after each change, what a watcher
   that has seen nothing before gives.

All need the built server.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import evidence as ev  # noqa: E402
from tensorquill import flight  # noqa: E402
from tensorquill.flight import FlightSender, offset_path, read_binding, send_file  # noqa: E402

from test_flight import append, record as sender_record, wait_for  # noqa: E402
from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402
from test_lane_ad11_costs import _append, _lines, _record, _Script  # noqa: E402
from test_server_detectors import FAST, RUN, record, wobble  # noqa: E402
from test_watch_all_ranks import Run, look, steady  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
CASES = json.loads((Path(__file__).parent / "fixtures" / "rules" / "cases.json").read_text())["cases"]
WITH_RECORDS = sorted(name for name, body in CASES.items() if "records" in body)
JUDGED = ("incidents", "not_checked", "kind_checks", "run_state", "moe_source", *ev.MEASUREMENT_BLOCKS)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)) or isinstance(b, float) and isinstance(a, (int, float)):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


# --- 1 ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", WITH_RECORDS)
def test_both_engines_give_the_same_evidence_on_a_rule_case(name):
    records = CASES[name]["records"]
    python = ev.build_evidence(records, detectors="python")
    rust = ev.build_evidence(records, detectors="rust")
    for key in JUDGED:
        assert _same(python.get(key), rust.get(key)), f"{key}: python {python.get(key)!r}\nrust {rust.get(key)!r}"[:1500]


# --- 2 ------------------------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="found at wave AD's trial: a sender that returns with a saved offset first sends an empty "
                   "batch to ask for the store, the real server refuses that batch for its columns too, and the refusal "
                   "names no range of the sender's, so the sender holds its offset and the refused rows are never sent")
def test_rs10_joined_the_real_server_refuses_and_the_production_sender_resumes_at_the_right_byte(tmp_path, monkeypatch):
    """The author's run with the production sender on both ends of each disconnect: A is stored;
    the sender returns and sends B, whose `step` column is text; it returns again and sends C.
    Intended: only A and C are stored, B is counted once on each side, and the sender's offset
    ends at the end of C.

    What happens on the trial tree (Nov 21, 2026): the second sender's first batch is the empty one
    that asks the server for its store. Its schema is the stream's, so the server refuses it; the
    refusal does not name bytes the sender sent, so by its own rule the sender keeps its offset and
    tries again. B's rows never leave, the server counts one refused batch (the empty one), and when
    the sender's rows are readable again B and C are both stored."""
    import pyarrow as pa

    store, path = tmp_path / "store", tmp_path / "run-a.rank0.jsonl"
    server = Server(store, 0, *FAST)   # parts written at once, so an acknowledgement does not wait for the flush interval
    good_schema, good_rows = flight.flight_schema, flight.rows_to_send
    text = pa.schema([pa.field("step", pa.string(), nullable=False) if f.name == "step" else f for f in good_schema()])

    def text_rows(records, schema, *more):
        batch, steps, *rest = good_rows(records, good_schema(), *more)
        at = batch.schema.get_field_index("step")
        table = pa.Table.from_batches([batch]).set_column(at, text.field("step"), pa.array([str(s) for s in steps], pa.string()))
        return (table.to_batches()[0], steps, *rest)

    def sender():
        return FlightSender(path, server.endpoint, "run-a", 0, batch_seconds=0.05, poll_seconds=0.02, backoff_cap=0.2)

    saved = lambda: (read_binding(offset_path(path)) or {}).get("offset")
    try:
        append(path, [sender_record(step) for step in range(0, 3)])                       # A
        assert sender().close(timeout=10.0) is True and saved() == path.stat().st_size
        monkeypatch.setattr(flight, "flight_schema", lambda: text)                         # B, from a sender that returned
        monkeypatch.setattr(flight, "rows_to_send", text_rows)
        append(path, [sender_record(step) for step in range(3, 5)])
        after_b = path.stat().st_size
        second = sender()
        try:
            wait_for(lambda: second.refused_batches == 1, timeout=8.0)
        finally:
            second.close(timeout=1.0)
        assert saved() == after_b                                                          # resumed past the refused bytes
        monkeypatch.setattr(flight, "flight_schema", good_schema)                          # C, from a sender that returned again
        monkeypatch.setattr(flight, "rows_to_send", good_rows)
        append(path, [sender_record(step) for step in range(5, 8)])
        assert sender().close(timeout=10.0) is True and saved() == path.stat().st_size
    finally:
        server.terminate()
    assert sorted(int(r["step"]) for r in stored_records(store, "run-a")) == [0, 1, 2, 5, 6, 7]   # only A and C
    status = json.loads((store / "runs" / "run-a" / "watchdog.json").read_text())
    entry = next(r for r in status["ranks"] if r["rank"] == 0)
    assert (entry.get("refused_batches"), entry.get("refused_batch_keys")) == (1, {"step": 1})


# --- 3 ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("options", [{}, {"starve_window": 30}], ids=["default_thresholds", "another_threshold"])
def test_a_rust_backed_watch_gives_a_fresh_judgment_at_every_look(tmp_path, options):
    path, out = tmp_path / "run.jsonl", tmp_path / "out.json"
    changes = [
        lambda: path.write_text(_lines(_record(s) for s in range(140))),
        lambda: _append(path, [_record(s) for s in range(140, 150)]),                    # steps appended
        lambda: _append(path, [_record(s, 2) for s in range(150, 160)]),                 # a new attempt after the newest step
        lambda: _append(path, [_record(s, 1) for s in range(150, 155)]),                 # late rows of the old attempt
        lambda: _append(path, [_record(s, 3) for s in range(145, 150)]),                 # an attempt replacing judged steps
        lambda: _append(path, [_record(s, 3) for s in range(150, 171)] + [_record(160, 2)]),
    ]
    script = _Script(path, out, changes, detectors="rust", **options)   # each look is held to build_evidence(path, detectors="rust")
    ev.watch(path, out, interval=0.0, polls=len(changes), trigger=script, detectors="rust", **options)
    assert script.checked == len(changes), "a look wrote nothing"


# --- 4 ------------------------------------------------------------------------------------------

def _more(start: int, stop: int, attempt: int = 1, spike: int | None = None) -> list[dict]:
    return [record(step, attempt, grad_norm=wobble(step) * (8.0 if step == spike else 1.0), lr=[1e-3],
                   scalars={"train/loss": wobble(step, 3)}) for step in range(start, stop)]


def _shape(evidence: dict) -> dict:
    """What an all-ranks view says, without the parts that name the moment it was made."""
    keys = ("judged_ranks", "unknown_ranks", "clean_ranks", "ranks_with_incidents", "incident_groups", "last_step", "kind_checks")
    ranks = [{k: r.get(k) for k in ("rank", "state", "incidents", "attempt", "last_step")} for r in evidence.get("ranks", [])]
    return {**{k: evidence.get(k) for k in keys}, "ranks": ranks}


def test_a_long_lived_all_ranks_watcher_gives_what_a_fresh_one_gives_after_each_change(tmp_path):
    run = Run(tmp_path, *FAST)
    try:
        for rank in range(3):
            run.send(rank, steady(60))

        def late_rows():
            with run.file(2).open("a", encoding="utf-8") as file:   # rows of attempt 1 at steps attempt 2 replaced
                for item in _more(90, 95):
                    file.write(json.dumps({**item, "rank": 2}) + "\n")
            assert send_file(run.file(2), run.server.endpoint, RUN, 2, timeout=30.0)
            time.sleep(1.0)

        changes = [
            lambda: None,
            lambda: [run.send(rank, _more(60, 90, spike=75 if rank == 1 else None)) for rank in range(3)],   # steps appended, one incident
            lambda: run.send(2, _more(90, 110, attempt=2)),                                                 # a new attempt
            late_rows,                                                                                      # late rows of the old one
            lambda: [run.send(rank, _more(90 if rank < 2 else 110, 130, attempt=1 if rank < 2 else 2)) for rank in range(3)],
        ]

        class Fresh:
            looks = checked = 0

            def tick(self) -> None:
                if self.looks < len(changes):
                    changes[self.looks]()
                self.looks += 1

            def look(self, evidence, written, rank=None) -> None:
                fresh, _ = look(run, tmp_path / f"fresh-{self.looks}")
                assert _shape(json.loads(written.read_text())) == _shape(fresh), f"look {self.looks}"
                self.checked += 1

        script = Fresh()
        ev.watch(run.folder, tmp_path / "all.json", interval=0.0, polls=len(changes), rank="all", trigger=script)
        assert script.checked >= 4, f"the watcher wrote on {script.checked} looks"   # a look with nothing new writes nothing
    finally:
        run.server.terminate()
