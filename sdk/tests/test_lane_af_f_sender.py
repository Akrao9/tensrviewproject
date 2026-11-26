"""Lane AF-F, wave AG: the sender and a line whose `attempt`, `rank` or `step` is negative
(`WAVE_AF_CONTRACT.md` section 10, "The sender (lane F)").

On the base and the trial tree the server refused the whole batch holding such a line, with no
range, and the sender sent that batch for the rest of the run: nothing after it was stored and
`close()` returned False. Now `rows_to_send` skips and counts the line, by `identity.check`'s rules,
and says the first skipped line once, as it does a line whose key cannot be stored."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from tensorquill import flight  # noqa: E402
from tensorquill.flight import FlightSender, flight_schema, parse_lines, rows_to_send  # noqa: E402
from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402

needs_server = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")


def line(at: int, run_id: str, **fields) -> str:
    record = {"schema_version": 1, "run_id": run_id, "rank": 0, "attempt": 0, "step": at, "ts": float(at),
              "grad_norm": 1.0 + at}
    record.update(fields)
    return json.dumps(record) + "\n"


def probe_file(path: Path, run_id: str, field: str, value) -> None:
    """Lane AF-F's probe: ten lines, steps 0 to 9, the sixth (step 5) the bad one."""
    path.write_text("".join(line(s, run_id, **({field: value} if s == 5 else {})) for s in range(10)))


@needs_server
@pytest.mark.parametrize("batch_steps", [50, 1])
@pytest.mark.parametrize("field", ["attempt", "rank", "step"])
def test_a_negative_key_line_is_skipped_and_every_valid_line_is_stored(tmp_path, field, batch_steps, capfd):
    run_id = f"ag-{field}-{batch_steps}"
    path = tmp_path / "rank0.jsonl"
    probe_file(path, run_id, field, -3)
    server = Server(tmp_path / "store")
    try:
        sender = FlightSender(path, server.endpoint, run_id, 0, batch_steps=batch_steps,
                              batch_seconds=0.05, poll_seconds=0.02, backoff_cap=0.2)
        closed = sender.close(20.0)
    finally:
        assert server.terminate() == 0
    # Wave AO (BUG-24 c, lane AO-2, accepted Nov 26): `close()` says False while a line of the run
    # was left out, counted in `left_out_lines`, and says so in one more line on stderr.
    assert closed is False and sender.left_out_lines == 1
    assert sorted(r["step"] for r in stored_records(tmp_path / "store", run_id)) == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    assert sender.skipped_lines == 1
    assert sender.acked_offset == path.stat().st_size
    said = [s for s in capfd.readouterr().err.splitlines() if s.startswith("tensorquill:")]
    offset = len("".join(line(s, run_id) for s in range(5)))
    assert len(said) == 2 and f"1 line of run {run_id}" in said[1], said
    assert f"the line at byte {offset} of {path}" in said[0] and f"{field} is -3" in said[0]


def test_rows_to_send_names_each_line_it_leaves_out_for_a_key():
    good = json.loads(line(0, "r"))
    records = [good, {**good, "step": 1, "rank": -1}, {**good, "step": 2, "attempt": None},
               {**good, "step": 3, "schema_version": 1.0}, {**good, "step": 2**70}, {**good, "step": 4, "schema_version": 99}]
    refused: list = []
    batch, steps, skipped = rows_to_send(records, flight_schema(), refused=refused)
    assert (steps, skipped) == ([0, 4], 4)
    assert [(index, field, value) for index, field, value in refused] == [
        (1, "rank", -1), (2, "attempt", None), (3, "schema_version", 1.0), (4, "step", 2**70)]


def test_a_step_that_is_not_an_integer_is_named_by_rows_to_send_not_skipped_silently():
    """The base's `parse_lines` left out a line of this run whose `step` is not an integer, with
    nothing said; that line now reaches `rows_to_send`, which skips, counts and names it. The total
    `skipped_lines` adds up the same."""
    lines = [line(0, "r").encode().rstrip(b"\n"), json.dumps({"run_id": "r", "step": "1"}).encode(),
             json.dumps({"run_id": "r"}).encode(), b"not json", line(9, "other").encode().rstrip(b"\n")]
    places: list = []
    records, skipped = parse_lines(lines, "r", places)
    assert skipped == 2 and places == [0, 1, 2]
    refused: list = []
    _, steps, unstorable = rows_to_send(records, flight_schema(), refused=refused)
    assert (steps, unstorable) == ([0], 2)
    assert [(i, f) for i, f, _ in refused] == [(1, "step"), (2, "step")]


def test_the_first_skipped_line_is_said_once_and_the_rest_counted(tmp_path, capfd):
    sender = FlightSender.__new__(FlightSender)
    sender.path, sender.endpoint, sender._said_skipped = tmp_path / "f.jsonl", "grpc://x", False
    lines = [b'{"a":1}', b'{"b":2}', b'{"c":3}']
    sender._say_skipped(100, lines, [0, 2], [(1, "attempt", -2)], [5, 7])
    sender._say_skipped(200, lines, [0, 1], [(0, "rank", True)], [8, 9])
    said = [s for s in capfd.readouterr().err.splitlines() if s]
    assert len(said) == 1
    assert said[0].startswith(f"tensorquill: the line at byte {100 + 8 + 8} of {tmp_path / 'f.jsonl'} (step 7) is not "
                              "sent to grpc://x: its attempt is -2, and tensorquill reads an integer of 0 or more")


def test_saying_never_raises_with_stderr_closed(tmp_path, monkeypatch):
    import sys

    class Closed:
        def write(self, *_):
            raise ValueError("I/O operation on closed file")

        def flush(self):
            raise ValueError("closed")

    monkeypatch.setattr(sys, "stderr", Closed())
    sender = FlightSender.__new__(FlightSender)
    sender.path, sender.endpoint, sender._said_skipped = tmp_path / "f.jsonl", "grpc://x", False
    sender._say_skipped(0, [b"{}"], [0], [(0, "step", flight.MalformedRecordError.ABSENT)], [None])
