"""Wave AO, lane AO-2: the sender (`flight.py`, `sink.py`, `identity.py`).

BUG-02, the sender's side: a line of record version 2 that is not a whole scalar-only record (no
`"record": "scalars"`) was sent as a step row (`record_version` 2, a null `seq`), and every Python
reader then refused the stored part. Such a line is now skipped and counted like any other line the
sender cannot send (`identity.check`'s rule for version 2, the one `is_scalar_record` mirrors).

Each test that needs pyarrow or the server binary (`TENSORQUILL_SERVER_BIN`, or one beside the
interpreter) is skipped without it, one by one.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from tensorquill import flight
from tensorquill.identity import is_scalar_record
from tensorquill.server_detect import find_binary

try:  # each test is collected and skipped without pyarrow
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

BINARY = find_binary()
needs_arrow = pytest.mark.skipif(pyarrow is None, reason="pyarrow is not installed")


def skip_without_server() -> None:
    if pyarrow is None:
        pytest.skip("pyarrow is not installed")
    if not BINARY or not os.path.isfile(BINARY):
        pytest.skip("tensorquill-server is not built")


class Server:
    """A server on a loopback port with its own store; `close()` stops it and returns its stderr."""

    def __init__(self, store: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("TENSORQUILL_TOKEN", "TENSORQUILL_ENDPOINT")}
        self.process = subprocess.Popen(
            [BINARY, "--listen", "127.0.0.1:0", "--store", str(store), "--flush-steps", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.process.stdout.readline()
        assert "listening on" in line, line + self.process.stderr.read()
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"
        self.store = store

    def close(self) -> str:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        self.process.wait(30)
        err = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        return err


def write_lines(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def step_record(run_id: str, step: int, **more) -> dict:
    return {"schema_version": 1, "run_id": run_id, "rank": 0, "step": step, "ts": float(step), "grad_norm": 1.0,
            **more}


def scalar_record(run_id: str, step: int, seq: int, scalars: dict) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": run_id, "rank": 0, "step": step, "seq": seq,
            "ts": float(step), "scalars": scalars}


# The line BUG-02 names: version 2 with no `"record": "scalars"`.
def bare_v2(run_id: str, step: int) -> dict:
    return {"schema_version": 2, "run_id": run_id, "rank": 0, "step": step, "loss": 1.0}


# --- BUG-02: a version 2 line that is not a whole scalar-only record ------------------------


@needs_arrow
def test_bug02_a_version_2_line_without_the_record_marker_is_left_out_and_counted():
    refused: list = []
    batch, steps, skipped = flight.rows_to_send([bare_v2("p", 4)], flight.flight_schema(), refused=refused)
    assert batch.num_rows == 0, batch.to_pylist()
    assert steps == [] and skipped == 1
    assert [(index, field, value) for index, field, value in refused] == [(0, "schema_version", 2)]


@needs_arrow
def test_bug02_the_audit_record_with_a_seq_but_no_marker_is_left_out_too():
    record = {"schema_version": 2, "run_id": "local", "rank": 0, "attempt": 1, "step": 4, "seq": 0,
              "scalars": {"loss": 1.0}}
    assert not is_scalar_record(record)
    batch, steps, skipped = flight.rows_to_send([record], flight.flight_schema())
    assert (batch.num_rows, steps, skipped) == (0, [], 1)


@needs_arrow
def test_bug02_pins_what_is_still_sent():
    """A whole scalar-only record is sent with its seq; a step record of version 0 or 1 as a step row;
    a version this package does not read (3) is sent for the server to count (RS10); a scalar-only
    record with no seq is left out naming seq, as before."""
    records = [
        scalar_record("p", 7, 0, {"eval/loss": 1.5}),
        step_record("p", 0),
        {"run_id": "p", "rank": 0, "step": 1},
        {**step_record("p", 2), "schema_version": 3},
        {"schema_version": 2, "record": "scalars", "run_id": "p", "rank": 0, "step": 9, "scalars": {"x": 1.0}},
    ]
    refused: list = []
    batch, steps, skipped = flight.rows_to_send(records, flight.flight_schema(), refused=refused)
    rows = batch.to_pylist()
    assert [(r["step"], r["seq"], r["record_version"], r["schema_version"]) for r in rows] == [
        (7, 0, 2, 2), (0, None, 1, 2), (1, None, None, 2), (2, None, 3, 3)]
    assert steps == [0, 1, 2] and skipped == 1
    assert [(i, f) for i, f, _ in refused] == [(4, "seq")]


def test_bug02_through_a_real_server_the_store_stays_readable(tmp_path, capsys):
    skip_without_server()
    from tensorquill.evidence import load_records

    running = Server(tmp_path / "store")
    try:
        path = write_lines(tmp_path / "p.jsonl", [step_record("p", 0), bare_v2("p", 1), step_record("p", 2)])
        sender = flight.FlightSender(path, running.endpoint, "p", 0)
        sender.close(20.0)
        assert sender.skipped_lines == 1
    finally:
        running.close()
    stored = load_records(tmp_path / "store" / "runs" / "p")
    assert [r["step"] for r in stored] == [0, 2]
    said = capsys.readouterr().err
    assert "(step 1) is not sent" in said and '"record": "scalars"' in said, said


# --- BUG-24 (a): an integer scalar a float64 cannot hold exactly ----------------------------
#
# Capture keeps a logged Python integer as it is (`capture.Session._scalar_values`), and the
# `.jsonl` file holds it exactly. The `scalars` column is Float64: the column keeps the nearest
# float, which is what the Rust detectors judge, and the exact integer travels beside it in
# `extra_json`, so a record read back from the store is the file's.

EXACT_EDGE = 2**53
PAST_EDGE = [2**53 + 1, -(2**53) - 1, 2**60 + 1, 2**63 + 1]


def _rows(records: list[dict], sender: bool) -> list[dict]:
    if sender:
        batch, _, skipped = flight.rows_to_send(records, flight.flight_schema())
        assert skipped == 0
        return batch.to_pylist()
    return flight.records_to_batch(records).to_pylist()


@needs_arrow
@pytest.mark.parametrize("sender", [False, True], ids=["record_to_row", "rows_to_send"])
def test_bug24a_an_integer_scalar_past_2_53_comes_back_exact(sender):
    scalars = {"loss": 1.5, **{f"n{i}": v for i, v in enumerate(PAST_EDGE)}}
    record = step_record("r", 0, scalars=scalars)
    row = _rows([record], sender)[0]
    assert dict(row["scalars"]) == {name: float(v) for name, v in scalars.items()}  # the column: the Rust view
    back = flight.row_to_record(row)
    assert back["scalars"] == scalars
    assert list(back["scalars"]) == list(scalars)  # in the record's order
    assert all(type(back["scalars"][f"n{i}"]) is int for i in range(len(PAST_EDGE)))


@needs_arrow
def test_bug24a_a_scalar_only_record_keeps_its_exact_integer_too():
    record = scalar_record("r", 3, 0, {"counter": 2**53 + 1})
    row = _rows([record], True)[0]
    assert row["seq"] == 0
    back = flight.row_to_record(row)
    assert back["scalars"] == {"counter": 2**53 + 1} and is_scalar_record(back)


@needs_arrow
def test_bug24a_pins_an_integer_a_float_holds_and_a_float():
    """2**53, a small integer and a float stay in the column alone, with no `extra_json`, as before."""
    scalars = {"edge": EXACT_EDGE, "small": 3, "loss": 0.25, "neg": -(2**53)}
    row = _rows([step_record("r", 0, scalars=scalars)], True)[0]
    assert row["extra_json"] is None
    assert flight.row_to_record(row)["scalars"] == scalars


def test_bug24a_through_a_real_server_the_stored_series_is_the_files(tmp_path):
    skip_without_server()
    import torch

    import tensorquill as tq
    from tensorquill.evidence import load_records, scalar_series

    path = tmp_path / "metrics.jsonl"
    with tq.init(torch.nn.Linear(1, 1), sink=tq.JsonlSink(path), run_id="metrics", mirror_scalars=False) as session:
        session.log(counter=2**53 + 1, step=0)
        session.log(eval_loss=1.5, step=1)
    running = Server(tmp_path / "store")
    try:
        assert flight.FlightSender(path, running.endpoint, "metrics", 0).close(20.0)
    finally:
        running.close()
    stored = tmp_path / "store" / "runs" / "metrics"
    local = scalar_series(path)
    assert local["counter"] == [[0, 2**53 + 1]]
    assert scalar_series(stored) == local
    assert [r["scalars"] for r in load_records(stored, scalars=True) if "scalars" in r] == \
        [r["scalars"] for r in load_records(path, scalars=True) if "scalars" in r]

