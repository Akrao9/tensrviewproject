"""A stand-in for `tensorlens-server`, run as a subprocess by `test_flight.py` so a test can
kill it mid-stream. Usage: `python flight_fake_server.py <store_dir> <port>`.

It follows the server's `DoPut` contract: rows whose step is not above the last one kept for
`(run_id, rank, attempt)` are dropped, and each batch is answered with its `end_byte_offset`. Kept
rows go to `<store_dir>/rows.jsonl` and one line per received batch to `batches.jsonl`, so
the store outlives a restart. While `<store_dir>/full` exists every batch is refused.
"""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.flight as flight


class FakeServer(flight.FlightServerBase):
    def __init__(self, store: Path, port: int) -> None:
        super().__init__(f"grpc://127.0.0.1:{port}")
        self.store = store
        self.watermark: dict[tuple[str, int, int], int] = {}
        rows = store / "rows.jsonl"
        if rows.exists():
            for line in rows.read_text().splitlines():
                row = json.loads(line)
                key = (row["run_id"], row["rank"], row.get("attempt", 0))
                self.watermark[key] = max(row["step"], self.watermark.get(key, -1))

    def do_put(self, context, descriptor, reader, writer):
        _, run_id, rank = (part.decode() for part in descriptor.path)
        stream = (run_id, int(rank.removeprefix("rank=")))
        while True:
            try:
                chunk = reader.read_chunk()
            except StopIteration:
                return
            if (self.store / "full").exists():
                raise flight.FlightServerError("ingest buffer full")
            metadata = json.loads(chunk.app_metadata.to_pybytes())
            rows = chunk.data.to_pylist()
            kept = []
            for row in rows:
                key = (*stream, row.get("attempt", 0))
                if row["step"] > self.watermark.get(key, -1):
                    self.watermark[key] = row["step"]
                    kept.append(row)
            with (self.store / "rows.jsonl").open("a") as file:
                file.writelines(json.dumps(row) + "\n" for row in kept)
            with (self.store / "batches.jsonl").open("a") as file:
                file.write(json.dumps({**metadata, "rows": len(rows)}) + "\n")
            writer.write(pa.py_buffer(json.dumps({"end_byte_offset": metadata["end_byte_offset"]}).encode()))


if __name__ == "__main__":
    server = FakeServer(Path(sys.argv[1]), int(sys.argv[2]))
    print(server.port, flush=True)
    server.serve()
