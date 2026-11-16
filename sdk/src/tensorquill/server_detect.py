"""The Rust detectors on records in hand, through `tensorquill-server detect`.

The server binary judges one rank's rows read from an Arrow stream and prints the report it
would write as that rank's `incidents.json`. `server_report` is the Python side: records in,
that report out. `build_evidence(..., detectors="rust")` takes its incidents and its per-step
MoE samples from it (`detect --samples`), and passes its own threshold options on as the
server's one `--thresholds` JSON object.

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


def rust_ready(binary: str | os.PathLike[str] | None = None) -> str:
    """What `server_report` needs, checked without running it: the binary (returned) and
    `pyarrow`. Raises `RuntimeError` naming what is missing."""
    chosen = server_binary(binary)
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError("the Rust detectors read Arrow: pip install 'tensorquill[flight]'") from error
    return chosen


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
    samples: bool = False,
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

    `samples=True` asks for one more key, `samples` (`detect --samples`): the per-step MoE
    samples as `build_evidence` lists them, one for every record and router, judged by the
    same rules as the incidents. The server's own `incidents.json` never has them. For a long
    MoE run they are most of the output: 97 MB for 10,000 steps of 24 routers of 32 experts.

    Raises `RuntimeError` when the binary or `pyarrow` is missing, or the binary fails.
    """
    records = list(records)
    if not records:
        raise ValueError("no records to judge")
    chosen = server_binary(binary)
    command = [chosen, "detect", *(["--samples"] if samples else [])]
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
            message += " (this tensorquill-server is too old to have `detect`, `--thresholds` or `--samples`: rebuild it)"
        raise RuntimeError(f"{chosen} detect failed: {message}")
    try:
        report = json.loads(done.stdout)
    except ValueError as error:
        raise RuntimeError(f"{chosen} detect did not print a report: {done.stdout[:200]!r}") from error
    if not isinstance(report, dict) or not isinstance(report.get("incidents"), list):
        raise RuntimeError(f"{chosen} detect printed something that is not a report")
    if samples and not isinstance(report.get("samples"), list):
        raise RuntimeError(f"{chosen} detect --samples printed a report without samples")
    return report


# --- every rank of a run: `tensorquill-server detect --run` -----------------------------------
#
# `detect` reads one rank. With `--run` it reads the rows of every rank of one run, grouped by
# their `rank` column, judges each rank as before, and compares the ranks' batch hashes as the
# server does for a run's status. That comparison is `duplicate_batches`, whose rule reads the
# rows of several ranks.

# What the cross-rank comparison reads of a record: the keys, and its `data`.
_DATA_KEYS = ("schema_version", "run_id", "rank", "attempt", "step", "ts", "data")


def _whole(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def run_stream(records: Iterable[Mapping[str, Any]]) -> bytes:
    """The records of every rank of one run as the Arrow IPC stream `tensorquill-server detect
    --run` reads: grouped by rank and then by attempt, oldest first, and otherwise in the
    order given (as `records_stream` orders one rank)."""
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("the Rust detectors read Arrow: pip install 'tensorquill[flight]'") from error
    from .flight import flight_schema, record_to_row

    rows = [
        record_to_row(record if isinstance(record.get("run_id"), str) else {**record, "run_id": ""})
        for record in sorted(records, key=lambda r: (_whole(r.get("rank", 0)), _whole(r.get("attempt", 0))))
    ]
    schema = flight_schema()
    out = io.BytesIO()
    with pa.ipc.new_stream(out, schema) as writer:
        for start in range(0, len(rows), _BATCH_ROWS):
            writer.write_batch(pa.RecordBatch.from_pylist(rows[start:start + _BATCH_ROWS], schema=schema))
    return out.getvalue()


def server_run_report(
    records: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any] | None = None,
    binary: str | os.PathLike[str] | None = None,
    timeout: float | None = 300.0,
) -> dict[str, Any]:
    """The server's detectors over every rank of one run, from records in hand: what
    `tensorquill-server detect --run` prints.

    Pass every attempt of every rank (`load_records(..., all_attempts=True)`); records of a
    second run are refused. Returns `{"run_id", "ranks", "reports", "ranks_compared",
    "duplicate_batches", "incidents", ...}`: `reports` holds each rank's report as
    `server_report` gives it for that rank alone, in rank order; `duplicate_batches` is the
    cross-rank finding or None; `incidents` lists it as an incident; and `ranks_compared` are
    the ranks that had a hashed batch to compare, whether or not anything was found.

    `thresholds`, `binary` and the errors are `server_report`'s.
    """
    records = list(records)
    if not records:
        raise ValueError("no records to judge")
    chosen = server_binary(binary)
    command = [chosen, "detect", "--run"]
    if thresholds is not None:
        try:
            command += ["--thresholds", json.dumps(dict(thresholds), allow_nan=False)]
        except (TypeError, ValueError) as error:
            raise ValueError(f"the thresholds cannot be sent as JSON: {error}") from error
    try:
        done = subprocess.run(command, input=run_stream(records), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{chosen} detect --run did not finish in {timeout:g} s") from error
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", "replace").strip() or f"exit status {done.returncode}"
        if "invalid value" in message and "--thresholds" in message:
            refused = message.split("for '--thresholds <JSON>':", 1)[-1].split("\n", 1)[0].strip()
            raise ValueError(f"the Rust detectors refuse these thresholds: {refused}")
        if "unrecognized subcommand" in message or "unexpected argument" in message:
            message += " (this tensorquill-server is too old to have `detect --run`: rebuild it)"
        raise RuntimeError(f"{chosen} detect --run failed: {message}")
    try:
        document = json.loads(done.stdout)
    except ValueError as error:
        raise RuntimeError(f"{chosen} detect --run did not print a document: {done.stdout[:200]!r}") from error
    if not isinstance(document, dict) or not isinstance(document.get("reports"), list) \
            or not isinstance(document.get("incidents"), list):
        raise RuntimeError(f"{chosen} detect --run printed something that is not a run's document")
    return document


def server_duplicate_batches(
    records: Iterable[Mapping[str, Any]],
    *,
    binary: str | os.PathLike[str] | None = None,
    timeout: float | None = 300.0,
) -> dict[str, Any] | None:
    """`duplicate_batches` across the ranks of a run, from the Rust rule: the incident
    `data_checks.detect_duplicate_batches(resolve_attempts(records))` gives (here with its
    `class`), or None.

    Pass every attempt of every rank: the Rust side leaves out what a resume rolled back, as
    `resolve_attempts` does. Only each record's keys and `data` are sent, so this costs far
    less than `server_run_report`, and nothing else of a rank is judged. None is also the
    answer when fewer than two ranks drew a hashed batch; `server_run_report`'s
    `ranks_compared` tells the two apart.
    """
    slim = [{key: record[key] for key in _DATA_KEYS if key in record} for record in records]
    if not slim:
        return None
    document = server_run_report(slim, binary=binary, timeout=timeout)
    found = [i for i in document["incidents"] if isinstance(i, dict) and i.get("kind") == "duplicate_batches"]
    return found[0] if found else None
