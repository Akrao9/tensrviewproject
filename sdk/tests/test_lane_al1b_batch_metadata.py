"""Wave AL step 2, the owner's turn of lane AL-1''s decision 6 (Nov 24): a batch's `min_step` and `max_step`
count training rows only.

A batch holding training rows and scalar-only rows gives the training rows' `min_step` and `max_step` alone,
also when a scalar-only row's step is far above or below them. A batch of scalar-only rows alone sends null for
both, as a batch with no steps does (the stream's first, empty batch that asks for the store: both keys
present, null). A batch of training rows alone is byte for byte what it was before.

The batches are read by a small Flight server in this process that records each batch's metadata and
acknowledges it. Tests skip one by one without pyarrow; none needs the server binary or git history."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from tensorquill import flight

RUN = "meta"


def _training(step: int) -> dict:
    return {"schema_version": 1, "run_id": RUN, "rank": 0, "attempt": 1, "step": step, "ts": float(step),
            "grad_norm": 1.0}


def _scalar(seq: int, step: int) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": RUN, "rank": 0, "attempt": 1, "step": step,
            "seq": seq, "ts": 100.0 + seq, "scalars": {"eval/loss": 1.0 + seq}}


def _sent(tmp_path: Path, records: list[dict]) -> list[bytes]:
    """`records` as the rank's `.jsonl`, sent whole with `send_file`; each batch's metadata as received."""
    pa = pytest.importorskip("pyarrow")
    pa_flight = pytest.importorskip("pyarrow.flight")

    class Recorder(pa_flight.FlightServerBase):
        def __init__(self) -> None:
            super().__init__("grpc://127.0.0.1:0")
            self.metadata: list[bytes] = []

        def do_put(self, context, descriptor, reader, writer):
            while True:
                try:
                    chunk = reader.read_chunk()
                except StopIteration:
                    return
                data = chunk.app_metadata.to_pybytes()
                self.metadata.append(data)
                end = json.loads(data)["end_byte_offset"]
                writer.write(pa.py_buffer(json.dumps({"end_byte_offset": end}).encode()))

    path = tmp_path / f"{RUN}.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    server = Recorder()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        assert flight.send_file(path, f"grpc://127.0.0.1:{server.port}", RUN, 0, timeout=30) is True
    finally:
        server.shutdown()
        thread.join(10)
    return server.metadata


def test_a_batch_of_training_rows_alone_is_byte_for_byte_what_it_was(tmp_path):
    [metadata] = _sent(tmp_path, [_training(step) for step in range(10, 15)])
    assert metadata == (b'{"schema_version": 2, "run_id": "meta", "rank": 0, "min_step": 10, "max_step": 14, '
                        b'"end_byte_offset": 535, "start_byte_offset": 0}')


@pytest.mark.parametrize("scalar_steps", [(1000,), (0,), (0, 1000), (12,)], ids=repr)
def test_scalar_only_rows_never_widen_a_batch_s_steps(tmp_path, scalar_steps):
    records = [_training(step) for step in range(10, 15)]
    for seq, step in enumerate(scalar_steps):
        records.insert(2 + seq, _scalar(seq, step))
    [metadata] = _sent(tmp_path, records)
    document = json.loads(metadata)
    assert (document["min_step"], document["max_step"]) == (10, 14)
    assert list(document) == ["schema_version", "run_id", "rank", "min_step", "max_step", "end_byte_offset",
                              "start_byte_offset"]


def test_a_batch_of_scalar_only_rows_alone_sends_null_for_both_as_a_batch_with_no_steps_does(tmp_path):
    records = [_scalar(0, 1000), _scalar(1, 3)]
    [metadata] = _sent(tmp_path, records)
    end = sum(len(json.dumps(r)) + 1 for r in records)
    assert json.loads(metadata) == {"schema_version": 2, "run_id": RUN, "rank": 0, "min_step": None, "max_step": None,
                                    "end_byte_offset": end, "start_byte_offset": 0}
    assert b'"min_step": null, "max_step": null' in metadata


def test_rows_to_send_gives_the_training_rows_steps_and_every_row():
    pytest.importorskip("pyarrow")
    records = [_training(10), _scalar(0, 1000), _training(11), _scalar(1, 0)]
    batch, steps, skipped = flight.rows_to_send(records, flight.flight_schema())
    assert (steps, skipped) == ([10, 11], 0)
    assert batch.column("step").to_pylist() == [10, 1000, 11, 0]
    assert batch.column("seq").to_pylist() == [None, 0, None, 1]
