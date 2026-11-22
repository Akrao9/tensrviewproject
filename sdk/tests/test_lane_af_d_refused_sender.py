"""Lane D of wave AF (Nov 21, 2026), section 7 of `WAVE_AF_CONTRACT.md` (D5): a sender whose
batches are refused, a test before any limit is chosen.

The production sender (`flight.FlightSender`, `flight.py` unedited) against the real server. A, a
valid batch, is stored. Then batches the server refuses for good, one after another on the same
sender (B1, then B2 and B3 in the second test). Then C, a valid batch, sent while the server is down,
so nothing confirms it; the server comes back on the same store and port, the sender reconnects,
and C is stored.

The sender builds every batch from one schema, so a refused batch is made as in
`test_wave_ad_trial.py`: for the sender of the B batches alone, its schema has a `step` column of
text, which the server refuses naming the column and the bytes. Everything else is the production
path on both sides.

What is asserted, for D5:
- C is eventually stored, and only A and C are: no refused bytes are stored and none are lost.
- The sender's saved offset moves only across bytes the server confirmed: acknowledged as stored, or
  refused for good naming exactly the range the sender sent. Every offset the sender writes is
  recorded against what the server had said by then, and checked on the test's thread.
- A range neither confirmed nor refused (C while the server is down) stays pending: the offset stays
  where it was, and C is sent again once the server is back.

What it showed (Nov 21, 2026, wave AF): with one refused batch all of it held; with several in a row
the second was never confirmed, since the server named it from the rank's last acknowledgement, not
from the end of the refusal it had confirmed. The sender then sent that batch for the rest of the run
and C behind it was never stored; it saved no unconfirmed offset. Fixed in the server (wave AG, section
9 of `WAVE_AF_CONTRACT.md`): it keeps the end of the last refusal it confirmed for the rank and names a
refused batch that claims to begin there from it. The test of three refusals in a row passes since.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import flight  # noqa: E402
from tensorquill.flight import FlightSender, offset_path, read_binding  # noqa: E402

from test_flight import append, record, wait_for  # noqa: E402
from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402
from test_server_detectors import FAST  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
RUN = "run-a"   # what `test_flight.record` writes
# Parts stored at once, and the status written at every change within a fifth of a second.
OPTIONS = (*FAST, "--watchdog-seconds", "0.2")


def _the_run(tmp_path: Path, monkeypatch, refusals: int) -> dict[str, Any]:
    """A, then `refusals` refused batches, then C while the server is down, then a reconnect.
    Returns what each side said; raises AssertionError, with what was heard, where a wait fails."""
    import pyarrow as pa

    store, path = tmp_path / "store", tmp_path / "run-a.rank0.jsonl"
    good_schema, good_rows = flight.flight_schema, flight.rows_to_send
    good_ack, good_refusal, good_write = FlightSender._ack, FlightSender._refusal, flight.write_offset
    text = pa.schema([pa.field("step", pa.string(), nullable=False) if f.name == "step" else f for f in good_schema()])

    def text_rows(records, schema, *more):
        batch, steps, *rest = good_rows(records, good_schema(), *more)
        at = batch.schema.get_field_index("step")
        table = pa.Table.from_batches([batch]).set_column(at, text.field("step"), pa.array([str(s) for s in steps], pa.string()))
        return (table.to_batches()[0], steps, *rest)

    # What the server said, as the sender heard it, and every offset the sender saved.
    acked: set[int] = set()
    refused: list[tuple[int, int]] = []
    written: list[tuple[int, int, str]] = []   # (from, to, why it was allowed), as each was written
    batches: set[tuple[int, int]] = set()      # the byte ranges of the batches appended, as the test made them

    def heard_ack(self, offset, store_id=None):
        acked.add(offset)
        return good_ack(self, offset, store_id)

    def heard_refusal(self, answer):
        try:
            refused.append((answer["start_byte_offset"], answer["end_byte_offset"]))
        except Exception:  # noqa: BLE001  the sender reads what it can; so does the record
            pass
        return good_refusal(self, answer)

    def recorded_write(where, offset, binding=None):
        # Runs on the sender's thread: recorded here, asserted by the test.
        before = (read_binding(where) or {}).get("offset", 0)
        if offset in acked:
            why = "acknowledged"
        elif (before, offset) in refused and (before, offset) in batches:
            why = "refused exactly"
        else:
            why = "UNCONFIRMED"
        written.append((before, offset, why))
        return good_write(where, offset, binding)

    monkeypatch.setattr(FlightSender, "_ack", heard_ack)
    monkeypatch.setattr(FlightSender, "_refusal", heard_refusal)
    monkeypatch.setattr(flight, "write_offset", recorded_write)

    def add(steps: range) -> tuple[int, int]:
        start = path.stat().st_size if path.exists() else 0
        append(path, [record(step) for step in steps])
        batches.add((start, path.stat().st_size))
        return start, path.stat().st_size

    def sender(endpoint: str) -> FlightSender:
        return FlightSender(path, endpoint, RUN, 0, batch_seconds=0.05, poll_seconds=0.02, backoff_cap=0.2)

    saved = lambda: (read_binding(offset_path(path)) or {}).get("offset")
    seen: dict[str, Any] = {"written": written, "refused": refused}
    server = Server(store, 0, *OPTIONS)
    port = server.port
    try:
        _, seen["after_a"] = add(range(0, 3))                                               # A
        assert sender(server.endpoint).close(timeout=10.0) is True and saved() == seen["after_a"]

        monkeypatch.setattr(flight, "flight_schema", lambda: text)                          # B1 ...: refused
        monkeypatch.setattr(flight, "rows_to_send", text_rows)
        refusing = sender(server.endpoint)
        seen["ends"] = []
        try:
            for number in range(1, refusals + 1):
                start, end = add(range(1 + 2 * number, 3 + 2 * number))
                try:
                    wait_for(lambda: refusing.refused_batches == number, timeout=20.0)
                except AssertionError:
                    raise AssertionError(
                        f"refused batch {number} ({start} to {end}) was never confirmed: the server named "
                        f"{sorted(set(refused))} ({len(refused)} refusals heard); saved {saved()}; written {written}"
                    ) from None
                seen.setdefault("last_refused", []).append(dict(refusing.last_refused))
                seen.setdefault("saved_after", []).append(saved())
                seen["ends"].append((start, end))
        finally:
            refusing.close(timeout=2.0)
        seen["counted_by_sender"] = refusing.refused_batches
        status_file = store / "runs" / RUN / "watchdog.json"

        def counted() -> Any:
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            entry = next((r for r in status.get("ranks", []) if r.get("rank") == 0), {})
            return entry.get("refused_batches"), entry.get("refused_batch_keys")

        wait_for(lambda: counted() == (refusals, {"step": refusals}), timeout=20.0)         # counted once each by the server
        server.terminate()                                                                  # the server goes down

        monkeypatch.setattr(flight, "flight_schema", good_schema)                           # C: valid, and nothing confirms it yet
        monkeypatch.setattr(flight, "rows_to_send", good_rows)
        _, seen["after_c"] = add(range(1 + 2 * (refusals + 1), 4 + 2 * (refusals + 1)))
        returning = sender(f"grpc://127.0.0.1:{port}")
        try:
            wait_for(lambda: returning.last_error is not None, timeout=20.0)               # it tried, and could not send
            seen["pending"] = saved()
            server = Server(store, port, *OPTIONS)                                          # back on the same store and port
            wait_for(lambda: saved() == seen["after_c"], timeout=30.0)                      # the reconnect stores C
        finally:
            seen["closed"] = returning.close(timeout=10.0)
    finally:
        try:
            server.terminate()
        except Exception:  # noqa: BLE001  already stopped
            pass
    seen["stored"] = sorted(int(r["step"]) for r in stored_records(store, RUN))
    return seen


def _check(seen: dict[str, Any], refusals: int) -> None:
    # Only bytes the server confirmed moved the offset, at every write.
    assert all(why != "UNCONFIRMED" for _, _, why in seen["written"]), seen["written"]
    assert [why for _, _, why in seen["written"]].count("refused exactly") == refusals
    # Each refusal named exactly the batch's bytes, and the sender saved their end.
    for (start, end), told, after in zip(seen["ends"], seen["last_refused"], seen["saved_after"]):
        assert (told["start_byte_offset"], told["end_byte_offset"], told["column"]) == (start, end, "step")
        assert after == end
    assert seen["counted_by_sender"] == refusals
    # C stayed pending while nothing confirmed it, and the reconnect stored it.
    assert seen["pending"] == seen["ends"][-1][1]
    assert seen["closed"] is True and seen["written"][-1] == (seen["ends"][-1][1], seen["after_c"], "acknowledged")
    a = [0, 1, 2]
    c = list(range(1 + 2 * (refusals + 1), 4 + 2 * (refusals + 1)))
    assert seen["stored"] == a + c                                                          # only A and C


def test_one_refused_batch_then_a_valid_batch_then_a_reconnect(tmp_path, monkeypatch):
    _check(_the_run(tmp_path, monkeypatch, 1), 1)


def test_several_refused_batches_then_a_valid_batch_then_a_reconnect(tmp_path, monkeypatch):
    _check(_the_run(tmp_path, monkeypatch, 3), 3)
