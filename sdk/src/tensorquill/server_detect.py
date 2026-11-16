"""The Rust detectors on records in hand, through `tensorquill-server detect`.

The server binary judges one rank's rows read from an Arrow stream and prints the report it
would write as that rank's `incidents.json`. `server_report` is the Python side: records in,
that report out. `build_evidence(..., detectors="rust")` takes its incidents from it, and
passes its own threshold options on as the server's one `--thresholds` JSON object.

Needs `pyarrow` (the `flight` extra) and the binary: on `PATH`, or named by
`TENSORQUILL_SERVER_BIN`.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

BINARY_ENV = "TENSORQUILL_SERVER_BIN"
_BATCH_ROWS = 500


def server_binary(binary: str | os.PathLike[str] | None = None) -> str:
    """The `tensorquill-server` to run: `binary`, else `TENSORQUILL_SERVER_BIN`, else the one
    on `PATH`. Raises `RuntimeError` when there is none."""
    chosen = binary or os.environ.get(BINARY_ENV) or shutil.which("tensorquill-server")
    if not chosen:
        raise RuntimeError(
            "the Rust detectors need the tensorquill-server binary: build it (cargo build --release in server/) "
            f"and put it on PATH, or set {BINARY_ENV} to its path"
        )
    if not Path(chosen).is_file():
        raise RuntimeError(f"{chosen} is not a file: {BINARY_ENV} or binary= must name the tensorquill-server binary")
    return str(chosen)


def records_stream(records: Iterable[Mapping[str, Any]]) -> bytes:
    """One rank's records as the Arrow IPC stream `tensorquill-server detect` reads: grouped
    by attempt, oldest first, and otherwise in the order given."""
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("the Rust detectors read Arrow: pip install 'tensorquill[flight]'") from error
    from .flight import flight_schema, record_to_row

    def attempt(record: Mapping[str, Any]) -> int:
        value = record.get("attempt", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    # A stable sort: within an attempt the rows stay in recorded order, as the server stores them.
    rows = [
        # Records written without a run ID still have to travel in the column.
        record_to_row(record if isinstance(record.get("run_id"), str) else {**record, "run_id": ""})
        for record in sorted(records, key=attempt)
    ]
    schema = flight_schema()
    out = io.BytesIO()
    with pa.ipc.new_stream(out, schema) as writer:
        for start in range(0, len(rows), _BATCH_ROWS):
            writer.write_batch(pa.RecordBatch.from_pylist(rows[start:start + _BATCH_ROWS], schema=schema))
    return out.getvalue()


def server_report(
    records: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any] | None = None,
    binary: str | os.PathLike[str] | None = None,
    timeout: float | None = 300.0,
) -> dict[str, Any]:
    """The server's detectors over one rank's records: the report `tensorquill-server` writes
    as `rank=<rank>/incidents.json`, made here from records in hand.

    Pass every attempt of the rank (`load_records(..., all_attempts=True)`): the detectors
    compare a resume with what it replaced. Records of a second rank or run are refused.

    `thresholds` are the detector options, sent as the one JSON object `--thresholds` takes:
    `overload_threshold`, `drift_tolerance`, `starve_window`, `starve_ratio` and `top_k` (None,
    a number, or a number per router name). A key left out keeps the server's default, and
    the report's own `thresholds` lists what was applied. An unknown key or a value the
    server refuses raises `ValueError`.

    Raises `RuntimeError` when the binary or `pyarrow` is missing, or the binary fails.
    """
    records = list(records)
    if not records:
        raise ValueError("no records to judge")
    chosen = server_binary(binary)
    command = [chosen, "detect"]
    if thresholds is not None:
        try:
            command += ["--thresholds", json.dumps(dict(thresholds), allow_nan=False)]
        except (TypeError, ValueError) as error:
            raise ValueError(f"the thresholds cannot be sent as JSON: {error}") from error
    try:
        done = subprocess.run(command, input=records_stream(records), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{chosen} detect did not finish in {timeout:g} s") from error
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", "replace").strip() or f"exit status {done.returncode}"
        if "invalid value" in message and "--thresholds" in message:
            # The server names the key and why: an option the caller chose, not a failure of the binary.
            refused = message.split("for '--thresholds <JSON>':", 1)[-1].split("\n", 1)[0].strip()
            raise ValueError(f"the Rust detectors refuse these thresholds: {refused}")
        if "unrecognized subcommand" in message or "unexpected argument" in message:
            message += " (this tensorquill-server is too old to have `detect` or `--thresholds`: rebuild it)"
        raise RuntimeError(f"{chosen} detect failed: {message}")
    try:
        report = json.loads(done.stdout)
    except ValueError as error:
        raise RuntimeError(f"{chosen} detect did not print a report: {done.stdout[:200]!r}") from error
    if not isinstance(report, dict) or not isinstance(report.get("incidents"), list):
        raise RuntimeError(f"{chosen} detect printed something that is not a report")
    return report
