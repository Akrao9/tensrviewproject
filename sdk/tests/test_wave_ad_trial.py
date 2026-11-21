"""Joined checks of wave AD's trial tree: what no lane could run in its own tree, and what the
author asked the trial to show.

1. Both engines, through `build_evidence`, on every rule case that gives records.
2. RS10 joined: the real server delivers the refusal and the production `FlightSender` resumes at
   the right byte. The sender builds every batch from one schema, so the refused batch is made by
   giving it, for that one batch, a schema whose `step` column is text; everything else is the
   production path on both sides. Reconnecting to a running server passes. The same run across a
   restart of the server itself is here too, as the regression for the follow-up: its safe half
   passes, and the half the follow-up must fix is an expected failure.
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


def test_a_store_query_is_answered_whatever_its_columns_and_is_not_a_refused_batch(tmp_path):
    """The narrow fix of wave AD's trial. A batch of no rows with an end offset of 0 is the sender's
    question for the store. It is answered before any column is looked at, counts no refusal and
    moves no offset; a batch with rows and the same columns is refused as before.
    Fail before the fix: the query's stream ended with the refusal, and it was counted."""
    import pyarrow as pa
    import pyarrow.flight as pa_flight

    store = tmp_path / "store"
    server = Server(store, 0, *FAST)
    text = pa.schema([pa.field("step", pa.string(), nullable=False) if f.name == "step" else f for f in flight.flight_schema()])
    client = pa_flight.FlightClient(server.endpoint)
    try:
        writer, reader = client.do_put(pa_flight.FlightDescriptor.for_path("runs", "query", "rank=0"), text,
                                       pa_flight.FlightCallOptions(timeout=30.0))
        metadata = {"schema_version": 1, "run_id": "query", "rank": 0, "end_byte_offset": 0, "min_step": None, "max_step": None}
        writer.write_with_metadata(pa.RecordBatch.from_pylist([], schema=text), pa.py_buffer(json.dumps(metadata).encode()))
        answer = json.loads(reader.read().to_pybytes())
        assert answer["end_byte_offset"] == 0 and "refused" not in answer and answer.get("store_id")
        writer.done_writing()
        while reader.read() is not None:
            pass
        writer.close()
    finally:
        client.close()
        server.terminate()
    status_file = store / "runs" / "query" / "watchdog.json"
    if status_file.exists():
        assert all("refused_batches" not in rank for rank in json.loads(status_file.read_text()).get("ranks", []))


def test_rs10_joined_the_real_server_refuses_and_the_production_sender_resumes_at_the_right_byte(tmp_path, monkeypatch):
    """The author's run with the production sender on both sides of each disconnect, against the
    real server. A is stored. The sender returns with its saved offset: its store query is answered,
    and it sends B, whose `step` column is text; the server confirms the refusal of exactly B's
    bytes, and the sender moves its offset across those bytes and no further. It returns again and
    sends C, which is stored. Only A and C are stored, and B is counted once on each side.

    The sender builds every batch from one schema, so B is made by giving it, for the life of the
    second sender, a schema whose `step` is text; everything else is the production path."""
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
        after_a = saved()
        second = sender()                                                                   # starts with a saved offset: it asks first
        try:
            wait_for(lambda: second.refused_batches == 1, timeout=15.0)
            assert second.last_refused == {**second.last_refused, "start_byte_offset": after_a, "end_byte_offset": after_b,
                                           "column": "step"}
            assert saved() == after_b                                                      # across the refused bytes, and no further
        finally:
            second.close(timeout=2.0)
        assert second.refused_batches == 1 and saved() == after_b
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


@pytest.fixture(scope="module")
def across_a_server_restart(tmp_path_factory):
    """RS10's open part, as one run (the regression for the follow-up). A is stored and the sender's
    offset saved. The server stops and a new one starts on the same store. The sender returns and
    sends B, whose `step` column is text, which the server refuses. Then a sender returns once more
    with C. What each side said is returned; the two tests below read it.

    B is made as in the joined test above, by a schema given to the second sender alone, so the
    third sender, which has the real schema, can send B's bytes again if its offset still stands
    before them: that is how a sender that did not move shows in what is stored."""
    import pyarrow as pa

    tmp = tmp_path_factory.mktemp("rs10-restart")
    store, path = tmp / "store", tmp / "run-a.rank0.jsonl"
    good_schema, good_rows, refusal = flight.flight_schema, flight.rows_to_send, FlightSender._refusal
    text = pa.schema([pa.field("step", pa.string(), nullable=False) if f.name == "step" else f for f in good_schema()])
    named: list[Any] = []   # every refusal the server told the sender, as it came

    def text_rows(records, schema, *more):
        batch, steps, *rest = good_rows(records, good_schema(), *more)
        at = batch.schema.get_field_index("step")
        table = pa.Table.from_batches([batch]).set_column(at, text.field("step"), pa.array([str(s) for s in steps], pa.string()))
        return (table.to_batches()[0], steps, *rest)

    def heard(self, refused):
        named.append(refused)
        return refusal(self, refused)

    def sender(server):
        return FlightSender(path, server.endpoint, "run-a", 0, batch_seconds=0.05, poll_seconds=0.02, backoff_cap=0.2)

    saved = lambda: (read_binding(offset_path(path)) or {}).get("offset")
    seen: dict[str, Any] = {}
    server = Server(store, 0, *FAST)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(FlightSender, "_refusal", heard)
            append(path, [sender_record(step) for step in range(0, 3)])                    # A
            assert sender(server).close(timeout=10.0) is True
            seen["after_a"] = saved()
            assert seen["after_a"] == path.stat().st_size
            server.terminate()                                                              # the server itself restarts
            seen["stored_at_restart"] = sorted(int(r["step"]) for r in stored_records(store, "run-a"))
            server = Server(store, 0, *FAST)
            patch.setattr(flight, "flight_schema", lambda: text)                            # B, from a sender that returned
            patch.setattr(flight, "rows_to_send", text_rows)
            append(path, [sender_record(step) for step in range(3, 5)])
            seen["after_b"] = path.stat().st_size
            second = sender(server)
            try:
                wait_for(lambda: bool(named), timeout=15.0)                                 # its query was answered, and B refused
            finally:
                second.close(timeout=2.0)
            seen.update(named=list(named), counted_by_sender=second.refused_batches, saved_after_b=saved())
            patch.setattr(flight, "flight_schema", good_schema)                             # C, from a sender that returned again
            patch.setattr(flight, "rows_to_send", good_rows)
            append(path, [sender_record(step) for step in range(5, 8)])
            seen["c_sent"] = sender(server).close(timeout=10.0)
            seen.update(saved_at_end=saved(), size=path.stat().st_size)
    finally:
        server.terminate()
    seen["stored"] = sorted(int(r["step"]) for r in stored_records(store, "run-a"))
    status = json.loads((store / "runs" / "run-a" / "watchdog.json").read_text())
    seen["counted_by_server"] = next(r for r in status["ranks"] if r["rank"] == 0).get("refused_batches")
    return seen


def test_rs10_across_a_server_restart_the_refusal_is_told_and_nothing_is_lost(across_a_server_restart):
    """What holds today across a restart of the server, and must go on holding: A survived the
    restart, the returning sender's query was answered and B's refusal reached it naming B's end
    and its column, the saved offset is at one of the two places it may be (before B, or across
    it), and A and C are both stored in the end."""
    seen = across_a_server_restart
    assert seen["stored_at_restart"] == [0, 1, 2]
    assert seen["named"] and all(isinstance(r, dict) and r.get("end_byte_offset") == seen["after_b"]
                                 and r.get("column") == "step" for r in seen["named"]), seen["named"]
    assert seen["saved_after_b"] in (seen["after_a"], seen["after_b"])
    assert seen["c_sent"] is True and seen["saved_at_end"] == seen["size"]
    assert {0, 1, 2, 5, 6, 7} <= set(seen["stored"])
    assert seen["counted_by_server"] == 1


@pytest.mark.xfail(strict=True, reason="RS10, open part (the follow-up to wave AD): a restarted server has forgotten the rank's "
                   "last acknowledged offset, so its refusal names a range from byte 0; the sender, whose range begins at its "
                   "saved offset, holds there and sends the refused bytes again")
def test_rs10_across_a_server_restart_the_sender_moves_across_the_refused_bytes(across_a_server_restart):
    """What the follow-up must make true. The refusal names exactly B's bytes, from the offset the
    sender saved before the restart; the sender counts it and moves across those bytes and no
    further; and only A and C are stored.
    Today: the refusal names byte 0 as its start, the sender counts nothing and stays before B, and
    B's bytes go again (here they are then stored, since the third sender has the real schema; a
    batch the server can never take would be sent without end, with nothing behind it stored)."""
    seen = across_a_server_restart
    assert [r.get("start_byte_offset") for r in seen["named"]] == [seen["after_a"]]
    assert seen["counted_by_sender"] == 1 and seen["saved_after_b"] == seen["after_b"]
    assert seen["stored"] == [0, 1, 2, 5, 6, 7]


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
