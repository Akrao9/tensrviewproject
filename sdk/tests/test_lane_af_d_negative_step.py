"""Lane D, wave AG (Nov 21, 2026): section 10 of `WAVE_AF_CONTRACT.md` at live ingest. A batch holding
a negative `step` is refused whole with `invalid_argument`, as a batch holding a negative `attempt` is:
nothing of it is stored or acknowledged, and the next valid batch is stored.

The case is lane D's probe of wave AF (`live_negative.py` in its notes), which on the trial tree had
the batch of steps -3 and -2 acknowledged and stored as `part--0000003--0000002.parquet`, then judged.
Fails before the fix: the batch was acknowledged and its part stored.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow.flight")

import pyarrow as pa  # noqa: E402
import pyarrow.flight as fl  # noqa: E402

from tensorquill import flight  # noqa: E402

from test_flight_e2e import BINARY, Server, stored_records  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
RUN = "neg-step"


def _put(endpoint: str, steps: list[int], end: int) -> dict:
    rows = []
    for step in steps:
        row = flight.record_to_row({"schema_version": 1, "run_id": RUN, "rank": 0, "attempt": 0, "step": 0, "grad_norm": 1.0})
        row["step"] = step
        rows.append(row)
    batch = pa.RecordBatch.from_pylist(rows, schema=flight.flight_schema())
    client = fl.FlightClient(endpoint)
    out: dict = {"acks": []}
    try:
        writer, reader = client.do_put(fl.FlightDescriptor.for_path("runs", RUN, "rank=0"), batch.schema)
        metadata = {"schema_version": 2, "run_id": RUN, "rank": 0, "min_step": min(steps), "max_step": max(steps),
                    "end_byte_offset": end}
        writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
        writer.done_writing()
        while (answer := reader.read()) is not None:
            out["acks"].append(json.loads(answer.to_pybytes()))
        writer.close()
    except pa.ArrowInvalid as error:
        out["refused"] = str(error).splitlines()[0]
    finally:
        client.close()
    return out


def test_a_batch_with_a_negative_step_is_refused_whole_and_the_next_valid_batch_is_stored(tmp_path):
    store = tmp_path / "store"
    server = Server(store, 0, "--flush-steps", "1")
    try:
        negative = _put(server.endpoint, [-3, -2], 100)
        valid = _put(server.endpoint, [0, 1], 200)
    finally:
        server.terminate()
    assert "invalid argument" in negative.get("refused", "") and "negative" in negative["refused"], negative
    assert all(ack.get("end_byte_offset") != 100 for ack in negative["acks"]), negative   # nothing acknowledged
    assert valid == {"acks": [{**valid["acks"][0], "end_byte_offset": 200}]}, valid
    assert not list((store / "runs" / RUN).glob("rank=0/attempt=*/part--*.parquet"))   # no part of negative steps
    assert sorted(int(record["step"]) for record in stored_records(store, RUN)) == [0, 1]
    report = json.loads((store / "runs" / RUN / "rank=0" / "incidents.json").read_text())
    assert (report["steps"], report["last_step"]) == (2, 1)
