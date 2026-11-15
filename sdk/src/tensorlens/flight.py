"""Opt-in streaming of a rank's local `.jsonl` file to `tensorlens-server` over Arrow Flight.

The local file is always written first and stays the source of truth. A daemon thread
tails it from the byte offset saved in `<run_id>.rank<rank>.offset`, sends complete lines
as Arrow record batches on one long-lived `DoPut` stream, and advances the offset file
only when the server acknowledges that the bytes are stored. Nothing here runs on the
training thread, and a server that is down, slow or full only delays the stream: the
sender keeps no backlog in memory and re-reads the file from the saved offset.

`pyarrow` is imported only when an endpoint is set (`pip install "tensorlens[flight]"`).
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

SCHEMA_VERSION = 1
BATCH_STEPS = 50
BATCH_SECONDS = 2.0
BATCH_BYTES = 4 * 1024 * 1024
BACKOFF_BASE = 0.1
BACKOFF_CAP = 30.0
CLOSE_TIMEOUT = 5.0
POLL_SECONDS = 0.2
ENDPOINT_ENV = "TENSORLENS_ENDPOINT"
TOKEN_ENV = "TENSORLENS_TOKEN"
TLS_CA_ENV = "TENSORLENS_TLS_CA"

_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


# --- Record <-> Arrow row conversion -------------------------------------------------


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("bool is not a float")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value in _NON_FINITE:
        return _NON_FINITE[value]  # JsonlSink writes non-finite numbers as these strings
    raise TypeError(f"not a number: {value!r}")


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"not an integer: {value!r}")
    return value


def _str(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"not a string: {value!r}")
    return value


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"not a bool: {value!r}")
    return value


def _list(item: Callable[[Any], Any]) -> Callable[[Any], list[Any]]:
    def convert(value: Any) -> list[Any]:
        if not isinstance(value, list):
            raise TypeError(f"not a list: {value!r}")
        return [item(v) for v in value]

    return convert


def _map(item: Callable[[Any], Any]) -> Callable[[Any], list[tuple[str, Any]]]:
    def convert(value: Any) -> list[tuple[str, Any]]:
        if not isinstance(value, dict):
            raise TypeError(f"not an object: {value!r}")
        return [(_str(k), item(v)) for k, v in value.items()]

    return convert


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


# JSONL key -> (Arrow column, converter). A value that does not fit its column's type, and
# any key not listed here, travels in `extra_json`, so no field is ever dropped.
_FIELDS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "ts": ("ts", _float),
    "lr": ("lr", _list(_float)),
    "grad_norm": ("grad_norm", _float),
    "clip_grad_norm": ("clip_grad_norm", _float),
    "weight_norm": ("weight_norm", _float),
    "weights_unchanged": ("weights_unchanged", _list(_str)),
    "step_skipped": ("step_skipped", _bool),
    "grad_scale": ("grad_scale", _float),
    "scalars": ("scalars", _map(_float)),
    "grad_norms": ("grad_norms", _map(_float)),
    "weight_norms": ("weight_norms", _map(_float)),
    "moe_top_k": ("moe_top_k", _map(_int)),
    "moe": ("moe_counts", _map(_list(_int))),
    "replica_spread": ("replica_spread", _map(_list(_float))),
    "layers": ("layers_json", _json),
    "moe_stats": ("moe_stats_json", _json),
    "probe": ("probe_json", _json),
}
_KEY_COLUMNS = ("schema_version", "run_id", "rank", "attempt", "step")
# Capture always writes these two, as null when there is no value, so a null column reads
# back as a present `None`. Every other null column reads back as an absent key.
_ALWAYS = ("ts", "grad_norm")
_COLUMNS = (*_KEY_COLUMNS, *(column for column, _ in _FIELDS.values()), "extra_json")


def flight_schema() -> Any:
    """The Arrow schema of one step per row (`schema_version` 1)."""
    import pyarrow as pa

    f64 = pa.float64()
    return pa.schema(
        [
            pa.field("schema_version", pa.int32(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("rank", pa.int32(), nullable=False),
            pa.field("attempt", pa.int64(), nullable=False),  # 0: written before attempts
            pa.field("step", pa.int64(), nullable=False),
            pa.field("ts", f64),
            pa.field("lr", pa.list_(f64)),
            pa.field("grad_norm", f64),
            pa.field("clip_grad_norm", f64),
            pa.field("weight_norm", f64),
            pa.field("weights_unchanged", pa.list_(pa.string())),
            pa.field("step_skipped", pa.bool_()),
            pa.field("grad_scale", f64),
            pa.field("scalars", pa.map_(pa.string(), f64)),
            pa.field("grad_norms", pa.map_(pa.string(), f64)),
            pa.field("weight_norms", pa.map_(pa.string(), f64)),
            pa.field("moe_top_k", pa.map_(pa.string(), pa.int32())),
            pa.field("moe_counts", pa.map_(pa.string(), pa.list_(pa.int64()))),
            pa.field("replica_spread", pa.map_(pa.string(), pa.list_(f64))),
            pa.field("layers_json", pa.string()),
            pa.field("moe_stats_json", pa.string()),
            pa.field("probe_json", pa.string()),
            pa.field("extra_json", pa.string()),
        ]
    )


def record_to_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """One JSONL step record as a row of `flight_schema()`."""
    row: dict[str, Any] = dict.fromkeys(_COLUMNS)
    extra: dict[str, Any] = {}
    for key, value in record.items():
        if key in _KEY_COLUMNS:
            continue
        spec = _FIELDS.get(key)
        if spec is None or (value is None and key not in _ALWAYS):
            extra[key] = value
            continue
        try:
            row[spec[0]] = spec[1](value)
        except (TypeError, ValueError):
            extra[key] = value
    row["schema_version"] = _int(record.get("schema_version", 0))
    row["run_id"] = _str(record["run_id"])
    row["rank"] = _int(record.get("rank", 0))
    row["attempt"] = _int(record.get("attempt", 0))
    row["step"] = _int(record["step"])
    if extra:
        row["extra_json"] = _json(extra)
    return row


def records_to_batch(records: Iterable[Mapping[str, Any]]) -> Any:
    import pyarrow as pa

    return pa.RecordBatch.from_pylist([record_to_row(r) for r in records], schema=flight_schema())


def _plain(value: Any) -> Any:
    """Undo the Arrow representation: maps come back as key-value pairs and non-finite
    floats go back to the strings `JsonlSink` writes."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, list):
        if value and all(isinstance(v, tuple) and len(v) == 2 for v in value):
            return {k: _plain(v) for k, v in value}
        return [_plain(v) for v in value]
    return value


def row_to_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of `record_to_row`: a row read back from the server's Parquet parts
    (`table.to_pylist()`) as the dict that `evidence.py` reads from a `.jsonl` file."""
    record: dict[str, Any] = {}
    if row["schema_version"]:
        record["schema_version"] = row["schema_version"]
    record.update(run_id=row["run_id"], rank=row["rank"])
    if row.get("attempt"):
        record["attempt"] = row["attempt"]
    record["step"] = row["step"]
    for key, (column, convert) in _FIELDS.items():
        value = row.get(column)
        if value is None:
            if key in _ALWAYS:
                record[key] = None
        elif convert is _json:
            record[key] = json.loads(value)
        elif column in _MAP_COLUMNS:
            record[key] = {k: _plain(v) for k, v in value}  # an empty map stays a dict
        else:
            record[key] = _plain(value)
    if row.get("extra_json"):
        record.update(json.loads(row["extra_json"]))
    return record


_MAP_COLUMNS = frozenset(
    {"scalars", "grad_norms", "weight_norms", "moe_top_k", "moe_counts", "replica_spread"}
)


# --- The local file and its offset ----------------------------------------------------


def offset_path(jsonl_path: str | Path) -> Path:
    """`<run_id>.rank<rank>.jsonl` -> `<run_id>.rank<rank>.offset`."""
    return Path(jsonl_path).with_suffix(".offset")


def read_offset(path: Path) -> int:
    try:
        return max(0, int(path.read_text(encoding="ascii").strip() or 0))
    except (OSError, ValueError):
        return 0


def write_offset(path: Path, offset: int) -> None:
    """Write and rename, so a crash leaves either the old offset or the new one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"{offset}\n", encoding="ascii")
    os.replace(tmp, path)


def read_lines(path: Path, offset: int, max_lines: int, max_bytes: int) -> tuple[list[bytes], int]:
    """Up to `max_lines` complete lines starting at byte `offset`, and the offset just after
    them. A trailing line with no newline yet is never returned: the writer may still be in
    the middle of it. One line longer than `max_bytes` is returned alone."""
    try:
        with open(path, "rb") as file:
            file.seek(offset)
            data = file.read(max_bytes)
            if b"\n" not in data:
                if len(data) < max_bytes:
                    return [], offset
                while True:  # one line longer than the cap: read through its newline
                    more = file.read(1024 * 1024)
                    if not more:
                        return [], offset
                    cut = more.find(b"\n")
                    if cut >= 0:
                        data += more[: cut + 1]
                        break
                    data += more
    except OSError:
        return [], offset
    lines = data.split(b"\n")[:-1][:max_lines]  # the last piece is the incomplete tail
    return lines, offset + sum(len(line) + 1 for line in lines)


def parse_lines(lines: Iterable[bytes], run_id: str) -> tuple[list[dict[str, Any]], int]:
    """The step records among `lines`, and how many lines were skipped. A line is skipped
    when it is not a JSON step record of this run: for example what a killed process left
    before the next process appended to the same file."""
    records: list[dict[str, Any]] = []
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            ok = (
                isinstance(record, dict)
                and record.get("run_id") == run_id
                and isinstance(record.get("step"), int)
                and not isinstance(record["step"], bool)
            )
        except ValueError:
            ok = False
        if ok:
            records.append(record)
        else:
            skipped += 1
    return records, skipped


# --- The sender ------------------------------------------------------------------------


def _location(endpoint: str) -> str:
    if endpoint.startswith("grpc://"):
        return endpoint
    if endpoint.startswith("grpcs://"):
        return "grpc+tls://" + endpoint[len("grpcs://") :]
    raise ValueError(f"endpoint must be grpc://host:port or grpcs://host:port, got {endpoint!r}")


def _read_tls_root_certs(tls_root_certs: bytes | str | os.PathLike[str] | None) -> bytes | None:
    if tls_root_certs is None:
        env_path = os.environ.get(TLS_CA_ENV)
        if not env_path:
            return None
        tls_root_certs = env_path
    if isinstance(tls_root_certs, (bytes, bytearray, memoryview)):
        return bytes(tls_root_certs)
    if isinstance(tls_root_certs, (str, os.PathLike)):
        return Path(tls_root_certs).read_bytes()
    raise TypeError(f"tls_root_certs must be bytes or a file path, got {type(tls_root_certs).__name__}")


class FlightSender:
    """Tails one rank's `.jsonl` file and streams it to the ingest server.

    One `DoPut` stream stays open. Each batch carries `end_byte_offset`, the file position
    just after its last line; the server answers with the offset it has stored durably, and
    only then does the `.offset` file move. After any failure the sender sleeps (full-jitter
    exponential backoff) and starts again from the saved offset, so at most one batch is in
    memory at a time and the server's `(run_id, rank, attempt, step)` filter drops what it already has.
    """

    def __init__(
        self,
        path: str | Path,
        endpoint: str,
        run_id: str,
        rank: int = 0,
        *,
        token: str | None = None,
        tls_root_certs: bytes | str | os.PathLike[str] | None = None,
        batch_steps: int = BATCH_STEPS,
        batch_seconds: float = BATCH_SECONDS,
        batch_bytes: int = BATCH_BYTES,
        poll_seconds: float = POLL_SECONDS,
        backoff_base: float = BACKOFF_BASE,
        backoff_cap: float = BACKOFF_CAP,
    ) -> None:
        import pyarrow.flight  # noqa: F401  fail here, at init, when pyarrow is missing

        self.path = Path(path)
        self.offset_path = offset_path(self.path)
        self.endpoint = endpoint
        self.run_id = run_id
        self.rank = rank
        self.last_error: BaseException | None = None
        self.skipped_lines = 0
        self._location = _location(endpoint)
        self._token = token if token is not None else os.environ.get(TOKEN_ENV)
        self._tls_root_certs = (
            _read_tls_root_certs(tls_root_certs)
            if (tls_root_certs is not None or self._location.startswith("grpc+tls://"))
            else None
        )
        self._batch_steps = batch_steps
        self._batch_seconds = batch_seconds
        self._batch_bytes = batch_bytes
        self._poll_seconds = poll_seconds
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._acked = read_offset(self.offset_path)
        self._progress = False
        self._reported = False
        self._pid = os.getpid()
        self._closing = threading.Event()
        self._deadline = math.inf
        self._thread = threading.Thread(
            target=self._run, name=f"tensorlens-flight-rank{rank}", daemon=True
        )
        self._thread.start()

    @property
    def acked_offset(self) -> int:
        """Bytes of the local file that the server has stored."""
        return self._acked

    def unacked_bytes(self) -> int:
        try:
            return max(0, self.path.stat().st_size - self._acked)
        except OSError:
            return 0

    def close(self, timeout: float = CLOSE_TIMEOUT) -> bool:
        """Send the complete lines that are left and wait up to `timeout` seconds for the
        server to store them. Returns whether the whole file was acknowledged; if not, says
        where the local file is. Call after the local file's writer has been closed.

        In a forked child (a `DataLoader` worker, a `multiprocessing` process) this does
        nothing and returns False: a fork copies this object but not its thread, and the
        process that called `init` is the one that streams the file."""
        if os.getpid() != self._pid:
            return False
        self._deadline = time.monotonic() + timeout
        self._closing.set()
        self._thread.join(timeout)
        left = self.unacked_bytes()
        if left:
            reason = f" ({self.last_error})" if self.last_error is not None else ""
            print(
                f"tensorlens: {left} bytes of {self.path} were not confirmed by "
                f"{self.endpoint}{reason}. The local file is complete; streaming resumes from "
                f"{self.offset_path.name} the next time this run ID streams.",
                file=sys.stderr,
            )
        return left == 0

    # Everything below runs on the sender thread.

    def _run(self) -> None:
        attempt = 0
        retried_on_close = False
        while True:
            was_closing = self._closing.is_set()
            try:
                self._stream()
                return
            except BaseException as error:  # noqa: BLE001  the stream must never raise into training
                self.last_error = error
            if self._progress:
                self._progress = False
                attempt = 0
                self._reported = False
            elif not self._reported and not self._closing.is_set():
                # Say once per outage that nothing is arriving: a wrong token or address
                # would otherwise stay silent until close(). close() reports on its own.
                self._reported = True
                print(
                    f"tensorlens: streaming to {self.endpoint} is failing ({self.last_error}). Training "
                    f"continues and {self.path} stays complete; the sender keeps retrying.",
                    file=sys.stderr,
                )
            if self._closing.is_set():
                if time.monotonic() >= self._deadline:
                    return
                if not was_closing and not retried_on_close:
                    retried_on_close = True
                    attempt = 0
                    continue
                delay = random.uniform(0.0, min(self._backoff_cap, self._backoff_base * 2**attempt))
                attempt += 1
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(delay, remaining))
                if time.monotonic() >= self._deadline:
                    return
                continue
            delay = random.uniform(0.0, min(self._backoff_cap, self._backoff_base * 2**attempt))
            attempt += 1
            if self._closing.wait(delay):
                if time.monotonic() >= self._deadline:
                    return
                retried_on_close = True
                attempt = 0

    def _ack(self, offset: int) -> None:
        if offset > self._acked:
            write_offset(self.offset_path, offset)
            self._acked = offset
            self._progress = True
            self.last_error = None

    def _stream(self) -> None:
        import pyarrow as pa
        import pyarrow.flight as flight

        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        if self._acked > size:  # the file was replaced; the server's step filter covers a resend
            self._acked = 0
        sent = self._acked
        schema = flight_schema()
        headers = [(b"authorization", f"Bearer {self._token}".encode())] if self._token else []
        client_kwargs: dict[str, Any] = {
            "generic_options": [("grpc.keepalive_time_ms", 30_000), ("grpc.keepalive_timeout_ms", 10_000)],
        }
        if self._tls_root_certs is not None:
            client_kwargs["tls_root_certs"] = self._tls_root_certs
        client = flight.FlightClient(self._location, **client_kwargs)
        try:
            descriptor = flight.FlightDescriptor.for_path("runs", self.run_id, f"rank={self.rank}")
            writer, reader = client.do_put(descriptor, schema, flight.FlightCallOptions(headers=headers))
            failure: list[BaseException] = []
            ended = threading.Event()

            def read_acks() -> None:
                try:
                    while True:
                        buffer = reader.read()
                        if buffer is None:
                            return
                        self._ack(int(json.loads(buffer.to_pybytes())["end_byte_offset"]))
                except BaseException as error:  # noqa: BLE001
                    failure.append(error)
                finally:
                    ended.set()

            acks = threading.Thread(target=read_acks, name="tensorlens-flight-acks", daemon=True)
            acks.start()
            waiting_since: float | None = None
            try:
                # The ack reader returning is how a stream the server ended shows up here;
                # the server's status (such as "ingest buffer full") is raised by close().
                while not ended.is_set():
                    closing = self._closing.is_set()  # read before the file, so nothing is missed
                    lines, end = read_lines(self.path, sent, self._batch_steps, self._batch_bytes)
                    now = time.monotonic()
                    if lines and waiting_since is None:
                        waiting_since = now
                    full = len(lines) >= self._batch_steps or end - sent >= self._batch_bytes
                    if lines and (full or closing or now - waiting_since >= self._batch_seconds):
                        records, skipped = parse_lines(lines, self.run_id)
                        self.skipped_lines += skipped
                        steps = [r["step"] for r in records]
                        metadata = {
                            "schema_version": SCHEMA_VERSION,
                            "run_id": self.run_id,
                            "rank": self.rank,
                            "min_step": min(steps) if steps else None,
                            "max_step": max(steps) if steps else None,
                            "end_byte_offset": end,
                        }
                        batch = pa.RecordBatch.from_pylist(
                            [record_to_row(r) for r in records], schema=schema
                        )
                        writer.write_with_metadata(batch, pa.py_buffer(json.dumps(metadata).encode()))
                        sent = end
                        waiting_since = None
                        continue
                    if closing and not lines:
                        break
                    ended.wait(self._poll_seconds) if closing else self._closing.wait(self._poll_seconds)
                if not ended.is_set():
                    writer.done_writing()  # the server stores what it holds and sends the last ack
                    ended.wait(max(0.0, self._deadline - time.monotonic()))
            except BaseException:
                try:
                    writer.close()
                except BaseException:  # noqa: BLE001  the first failure is the one to report
                    pass
                raise
            writer.close()
            if failure:
                raise failure[0]
            if self._acked < sent or not self._closing.is_set():
                raise ConnectionError("the server ended the stream before confirming every batch")
        finally:
            client.close()


def send_file(
    path: str | Path,
    endpoint: str,
    run_id: str,
    rank: int = 0,
    *,
    timeout: float = 60.0,
    token: str | None = None,
    tls_root_certs: bytes | str | os.PathLike[str] | None = None,
) -> bool:
    """Send what a finished run's `.jsonl` file still owes the server. Returns whether the
    server confirmed the whole file."""
    return FlightSender(
        path, endpoint, run_id, rank, token=token, tls_root_certs=tls_root_certs
    ).close(timeout)


# --- The hook used by `tensorlens.init` ------------------------------------------------


class StreamingSink:
    """A sink that also stops the sender, after the local file is closed and complete."""

    def __init__(self, sink: Any, sender: FlightSender) -> None:
        self._sink = sink
        self.sender = sender
        self.path = sink.path

    def write(self, records: list[dict[str, Any]]) -> None:
        self._sink.write(records)

    def close(self) -> None:
        self._sink.close()
        self.sender.close()


def attach(
    sink: Any,
    run_id: str,
    rank: int,
    endpoint: str | None = None,
    *,
    token: str | None = None,
    tls_root_certs: bytes | str | os.PathLike[str] | None = None,
) -> Any:
    """Start streaming `sink`'s file when `endpoint` or `TENSORLENS_ENDPOINT` is set;
    otherwise return `sink` unchanged, with no socket opened and `pyarrow` not imported."""
    endpoint = endpoint or os.environ.get(ENDPOINT_ENV)
    if not endpoint:
        return sink
    _location(endpoint)
    path = getattr(sink, "path", None)
    if path is None:
        raise ValueError(
            "endpoint= streams the local .jsonl file, so the sink must be a JsonlSink "
            f"(or left as the default), not {type(sink).__name__}"
        )
    try:
        sender = FlightSender(
            path, endpoint, run_id, rank, token=token, tls_root_certs=tls_root_certs
        )
    except ImportError as error:
        raise ImportError(
            'endpoint= needs pyarrow: pip install "tensorlens[flight]"'
        ) from error
    return StreamingSink(sink, sender)
