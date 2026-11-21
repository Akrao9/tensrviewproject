"""Lane AD4 (wave AD): the writer's files.

Item 2: two Flight Recorder analyses at once (`collectives.py`).
Item 3: a sink written again after a failed write (`sink.py`).
Item 4: a record field the typed Arrow schema cannot encode (`flight.py`).
Item 5: what the sender keeps when a stream ends without it having closed it (`flight.py`).
"""

from __future__ import annotations

import logging
import pickle
import subprocess
import sys
import textwrap
import threading
import types
from pathlib import Path

import pytest

def _src() -> str:
    """The `src` folder of the `tensorquill` this test imported, for a script run in a fresh process."""
    import tensorquill

    return str(Path(tensorquill.__file__).resolve().parents[1])


# --- Item 2: two Flight Recorder analyses at once -------------------------------------------

def _fr_entry(record_id, name, seq, state="completed"):
    return {
        "profiling_name": f"nccl:{name}", "state": state, "process_group": ("0", "default_pg"),
        "input_sizes": [[4, 4]], "output_sizes": [[4, 4]], "input_dtypes": "float32", "output_dtypes": "float32",
        "collective_seq_id": str(seq), "p2p_seq_id": "0", "time_created_ns": 0, "frames": [],
        "record_id": record_id, "is_p2p": False,
    }


def _fr_desync(tmp_path: Path) -> Path:
    """Two ranks that issued different collectives at #2: the analyzer names a culprit (INFO)."""
    directory = tmp_path / "fr"
    directory.mkdir()
    ok = _fr_entry(0, "all_reduce", 1)
    for rank, second in ((0, "all_reduce"), (1, "all_gather")):
        document = {"entries": [ok, _fr_entry(1, second, 2, "scheduled")], "version": "2.8",
                    "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}}}
        (directory / f"nccl_trace_rank_{rank}").write_bytes(pickle.dumps(document))
    return directory


class _Seen(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def fr_logger():
    pytest.importorskip("torch.distributed.flight_recorder.components.builder")  # PyTorch 2.10 or later
    from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger

    logger = FlightRecorderLogger().logger
    seen = _Seen()
    logger.addHandler(seen)
    try:
        yield logger, seen
    finally:
        logger.removeHandler(seen)


def _state(logger: logging.Logger):
    return list(logger.handlers), logger.level, logger.propagate, list(logger.filters)


def _two_at_once(monkeypatch, target, *, where: str):
    """Run `target(name)` in threads A and B, ordered A in, B in, A out, B out at `where`
    ("build_db"): each blocks there until the other has come in, and B leaves after A."""
    import torch.distributed.flight_recorder.components.builder as builder

    real = builder.build_db
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

    def ordered(*args, **kwargs):
        if threading.current_thread().name == "A":
            a_in.set()
            assert b_in.wait(10)
            return real(*args, **kwargs)
        assert a_in.wait(10)
        b_in.set()
        assert a_out.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(builder, "build_db", ordered)
    results: dict[str, object] = {}
    errors: list[BaseException] = []

    def run(name: str) -> None:
        try:
            results[name] = target(name)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
        finally:
            if name == "A":
                a_out.set()

    threads = [threading.Thread(target=run, args=(n,), name=n) for n in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return results, errors


def test_fr_two_analyses_at_once_leave_the_logger_as_it_was_and_each_keeps_its_own_messages(
        tmp_path, monkeypatch, fr_logger):
    from tensorquill.collectives import analyze_flight_recorder

    logger, seen = fr_logger
    directory = _fr_desync(tmp_path)
    alone = analyze_flight_recorder(directory)["problem"]["mismatch"]
    assert alone and "Culprit rank 1" in alone
    before = _state(logger)
    seen.messages.clear()
    results, errors = _two_at_once(
        monkeypatch, lambda name: analyze_flight_recorder(directory)["problem"]["mismatch"], where="build_db")
    assert not errors
    # Before: B's list restored A's collector as the logger's only handler, with propagate False,
    # and A's culprit sentence went into B's list.
    assert _state(logger) == before
    assert results == {"A": alone, "B": alone}
    assert seen.messages == []  # neither analysis printed through the logger's own handlers


def test_fr_a_handler_added_by_other_code_during_an_analysis_is_kept(tmp_path, monkeypatch, fr_logger):
    import torch.distributed.flight_recorder.components.builder as builder

    from tensorquill.collectives import analyze_flight_recorder

    logger, seen = fr_logger
    directory = _fr_desync(tmp_path)
    analyze_flight_recorder(directory)
    other = _Seen()
    real = builder.build_db

    def adding(*args, **kwargs):
        logger.addHandler(other)  # other code, while the analysis runs
        return real(*args, **kwargs)

    monkeypatch.setattr(builder, "build_db", adding)
    try:
        analyze_flight_recorder(directory)
        assert other in logger.handlers and seen in logger.handlers  # before: the list put back dropped it
    finally:
        logger.removeHandler(other)
    assert other.messages == []


def test_fr_an_analysis_that_raises_alone_or_beside_another_leaves_the_logger_as_it_was(
        tmp_path, monkeypatch, fr_logger):
    import torch.distributed.flight_recorder.components.builder as builder

    from tensorquill.collectives import analyze_flight_recorder

    logger, seen = fr_logger
    directory = _fr_desync(tmp_path)
    alone = analyze_flight_recorder(directory)["problem"]["mismatch"]
    before = _state(logger)
    real = builder.build_db

    def fails(*args, **kwargs):
        logger.info("half way")
        raise RuntimeError("the analyzer failed")

    monkeypatch.setattr(builder, "build_db", fails)
    with pytest.raises(RuntimeError, match="the analyzer failed"):
        analyze_flight_recorder(directory)
    assert _state(logger) == before and seen.messages == []
    monkeypatch.setattr(builder, "build_db", real)

    # A raises while B runs, ordered A in, B in, A out (raising), B out.
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

    def ordered(*args, **kwargs):
        if threading.current_thread().name == "A":
            a_in.set()
            assert b_in.wait(10)
            logger.info("half way")
            raise RuntimeError("the analyzer failed")
        assert a_in.wait(10)
        b_in.set()
        assert a_out.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(builder, "build_db", ordered)
    results: dict[str, object] = {}

    def run(name):
        try:
            results[name] = analyze_flight_recorder(directory)["problem"]["mismatch"]
        except RuntimeError as error:
            results[name] = str(error)
        finally:
            if name == "A":
                a_out.set()

    threads = [threading.Thread(target=run, args=(n,), name=n) for n in ("A", "B")]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert results == {"A": "the analyzer failed", "B": alone}
    assert _state(logger) == before and seen.messages == []
    # And afterwards a record from this thread reaches the logger's handlers again.
    logger.warning("after")
    assert seen.messages == ["after"]


def test_fr_two_analyses_at_once_with_the_logger_set_above_info(tmp_path, monkeypatch, fr_logger):
    """The culprit sentence is logged at INFO: a logger set above INFO is lowered while analyses
    run and set back once the last one ends. Before, the second analysis to end set back the INFO
    the first had set, so the logger was left at INFO."""
    from tensorquill.collectives import analyze_flight_recorder

    logger, seen = fr_logger
    directory = _fr_desync(tmp_path)
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        before = _state(logger)
        assert "Culprit rank 1" in analyze_flight_recorder(directory)["problem"]["mismatch"]
        assert _state(logger) == before
        results, errors = _two_at_once(
            monkeypatch, lambda name: analyze_flight_recorder(directory)["problem"]["mismatch"], where="build_db")
        assert not errors and all("Culprit rank 1" in results[n] for n in ("A", "B"))
        assert _state(logger) == before and logger.level == logging.WARNING and seen.messages == []
    finally:
        logger.setLevel(level)


def test_fr_two_imports_at_once_leave_stdout_and_stderr_as_they_were(tmp_path):
    """The analyzer's modules are imported inside each analysis. Before, each import ran with the
    process's `sys.stdout` and `sys.stderr` swapped for buffers; two at once, ordered A in, B in,
    A out, B out, left both swapped for A's buffers. Run in a fresh process, with the builder
    module made to hold each import at `build_db` until both have come in."""
    directory = _fr_desync(tmp_path)
    script = textwrap.dedent(f"""
        import sys, threading, types, importlib
        sys.path.insert(0, {_src()!r})
        name = "torch.distributed.flight_recorder.components.builder"
        real = importlib.import_module(name)
        a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

        class Held(types.ModuleType):
            def __getattr__(self, attr):
                if attr == "build_db":
                    if threading.current_thread().name == "A":
                        a_in.set(); b_in.wait(10)
                    else:
                        a_in.wait(10); b_in.set(); a_out.wait(10)
                return getattr(real, attr)

        sys.modules[name] = Held(name)
        out, err = sys.stdout, sys.stderr
        from tensorquill.collectives import analyze_flight_recorder

        def run(n):
            try:
                analyze_flight_recorder({str(directory)!r})
            finally:
                if n == "A":
                    a_out.set()

        threads = [threading.Thread(target=run, args=(n,), name=n) for n in "AB"]
        [t.start() for t in threads]; [t.join(30) for t in threads]
        print("stdout kept" if sys.stdout is out else "stdout swapped", file=sys.__stdout__)
        print("stderr kept" if sys.stderr is err else "stderr swapped", file=sys.__stdout__)
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert done.stdout.split("\n")[:2] == ["stdout kept", "stderr kept"], done.stdout + done.stderr


def test_fr_the_analyzers_own_console_handler_writes_to_the_real_stderr(tmp_path):
    """The analyzer makes its console handler at its first import, from `sys.stderr` as it is then.
    Before, that import ran with `sys.stderr` swapped for a buffer, so in a process whose first
    analysis was TensorQuill's the handler wrote into that buffer for good."""
    pytest.importorskip("torch.distributed.flight_recorder.components.builder")
    directory = _fr_desync(tmp_path)
    script = textwrap.dedent(f"""
        import sys, logging
        sys.path.insert(0, {_src()!r})
        from tensorquill.collectives import analyze_flight_recorder
        analyze_flight_recorder({str(directory)!r})
        streams = [h.stream for h in logging.getLogger("Flight Recorder").handlers if hasattr(h, "stream")]
        print("real" if streams and all(s is sys.__stderr__ for s in streams) else f"not real: {{streams}}")
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert done.stdout.strip() == "real", done.stdout + done.stderr


# --- Item 3: a sink written again after a write that failed partway --------------------------

_SINK_SCRIPT = """
import json, resource, signal, sys
from pathlib import Path
sys.path.insert(0, {src!r})
from tensorquill.evidence import read_jsonl
from tensorquill.flight import parse_lines, read_lines
from tensorquill.sink import JsonlSink

path, pad = Path({path!r}), {pad}
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # write(2) then writes what fits and fails with EFBIG
soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)

def rec(step):
    return {{"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": step, "ts": 1.0,
            "grad_norm": 1.0, "pad": "x" * pad}}

sink = JsonlSink(path)
sink.write([rec(0), rec(1)])
resource.setrlimit(resource.RLIMIT_FSIZE, (path.stat().st_size + pad // 2, hard))
try:
    sink.write([rec(2)])
    failed = False
except OSError:
    failed = True
torn = not path.read_bytes().endswith(b"\\n")
resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
sink.write([rec(3), rec(4)])  # the same sink again, the limit gone: this write returns
sink.close()
records, left_out = read_jsonl(path)
lines, _ = read_lines(path, 0, 1000, 1 << 22)
sent, skipped = parse_lines(lines, "r")
print(json.dumps({{"failed": failed, "torn": torn, "read": [r["step"] for r in records], "left_out": left_out,
                  "sent": [r["step"] for r in sent], "skipped": skipped}}))
"""


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGXFSZ"), reason="needs RLIMIT_FSIZE and SIGXFSZ")
@pytest.mark.parametrize("pad", [150, 20000], ids=["a line under Python's buffer", "a line over it"])
def test_sink_the_next_write_after_one_that_failed_partway_is_read(tmp_path, pad):
    """A real partial write: a file-size limit lets the third step's line in part, then fails it.
    With the limit raised, the same sink's next write returns, and its two steps must be read by the
    evidence reader and the stream sender alike. Under the buffer, Python keeps the rest of the
    failed line and completes it, so nothing is lost. Over it, the rest is dropped at the failure:
    before, the next step was appended onto the fragment and read by neither (steps 0, 1, 4)."""
    script = _SINK_SCRIPT.format(src=_src(), path=str(tmp_path / "r.rank0.jsonl"), pad=pad)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    found = __import__("json").loads(done.stdout)
    assert found["failed"] and found["torn"]  # the write did fail partway, and left part of a line
    expected = [0, 1, 2, 3, 4] if pad < 8192 else [0, 1, 3, 4]  # only the failed write's step can be lost
    assert found["read"] == expected and found["sent"] == expected
    assert found["left_out"] == found["skipped"] == (0 if pad < 8192 else 1)


# --- Items 4 and 5: an in-process stand-in for the server, whose ending a test chooses -------

def _flight():
    return pytest.importorskip("pyarrow.flight")


def _record(step, **more):
    return {"schema_version": 1, "run_id": "run-a", "rank": 0, "attempt": 1, "step": step, "ts": 1.0 + step,
            "grad_norm": 0.5, "lr": [0.01], **more}


def _append(path: Path, records) -> None:
    import json

    with path.open("a", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, separators=(",", ":")) + "\n")


def _wait(condition, timeout: float = 15.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.02)


@pytest.fixture
def stand_in():
    import json

    flight = _flight()
    pa = pytest.importorskip("pyarrow")

    class StandIn(flight.FlightServerBase):
        """Keeps every row and acks every batch, unless `ending` says how to end a stream after its
        first batch: "unavailable" raises UNAVAILABLE, "return" ends it with OK."""

        def __init__(self) -> None:
            super().__init__("grpc://127.0.0.1:0")
            self.rows: list[dict] = []
            self.ending: str | None = None
            self.headers: list = []

        def do_put(self, context, descriptor, reader, writer):
            batches = 0
            while True:
                try:
                    chunk = reader.read_chunk()
                except StopIteration:
                    return
                metadata = json.loads(chunk.app_metadata.to_pybytes())
                if chunk.data is not None and chunk.data.num_rows:
                    batches += 1
                    if batches >= 2 and self.ending == "unavailable":
                        raise flight.FlightUnavailableError("going away")
                    if batches >= 2 and self.ending == "return":
                        return
                    self.rows.extend(chunk.data.to_pylist())
                writer.write(pa.py_buffer(json.dumps({"end_byte_offset": metadata["end_byte_offset"]}).encode()))

    server = StandIn()
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    server.endpoint = f"grpc://127.0.0.1:{server.port}"
    yield server
    server.shutdown()


def _sender(path, endpoint, **options):
    from tensorquill.flight import FlightSender

    options = {"batch_seconds": 0.02, "poll_seconds": 0.02, "backoff_base": 0.02, "backoff_cap": 0.1, **options}
    return FlightSender(path, endpoint, "run-a", 0, **options)


# --- Item 4: a field the typed Arrow schema cannot encode ------------------------------------

_UNENCODABLE = {
    "an integer too large for a float": {"grad_norm": 10**400},
    "a logged scalar too large for a float": {"scalars": {"loss": 10**400, "acc": 0.5}},
    "a top-k past int32": {"moe_top_k": {"router": 2**40}},
    "an expert count past int64": {"moe": {"router": [1, 2**70]}},
    "a parameter name with a lone surrogate": {"grad_norms": {"w\ud800": 1.0, "b": 2.0}},
    "a frozen name with a lone surrogate": {"weights_unchanged": ["w\ud800"]},
    "a data wait too large for a float": {"data_wait": 10**400},
}


@pytest.mark.parametrize("fields", list(_UNENCODABLE.values()), ids=list(_UNENCODABLE))
def test_unencodable_field_the_step_is_sent_with_that_field_in_extra_json(fields):
    """Before, `rows_to_send` left the whole step out (counted as skipped) for each of these; the
    stream went on. Now the rest of the row is sent, the value travels in `extra_json`, and reads
    back as written."""
    pytest.importorskip("pyarrow")
    from tensorquill.flight import flight_schema, row_to_record, rows_to_send

    bad, good = _record(5, **fields), _record(6)
    moved: list = []
    batch, steps, skipped = rows_to_send([bad, good], flight_schema(), moved)
    assert steps == [5, 6] and skipped == 0
    assert moved == [(5, key) for key in fields]
    back = {row["step"]: row_to_record(row) for row in batch.to_pylist()}
    for key, value in fields.items():
        assert back[5][key] == value
    assert back[5]["ts"] == 6.0 and back[5]["lr"] == [0.01] and back[6] == row_to_record(
        rows_to_send([good], flight_schema())[0].to_pylist()[0])


def test_unencodable_key_column_still_skips_the_line():
    """A key column cannot travel in `extra_json`: such a line is skipped and counted, as before."""
    pytest.importorskip("pyarrow")
    from tensorquill.flight import flight_schema, rows_to_send

    moved: list = []
    batch, steps, skipped = rows_to_send([_record(5, rank=2**40), _record(6), _record(7, grad_norm=10**400)],
                                         flight_schema(), moved)
    assert steps == [6, 7] and skipped == 1 and moved == [(7, "grad_norm")]


def test_unencodable_field_through_the_sender_every_step_reaches_the_server_and_it_is_said_once(
        tmp_path, stand_in, capsys):
    path = tmp_path / "run-a.rank0.jsonl"
    _append(path, [_record(0), _record(1, grad_norm=10**400), _record(2, moe_top_k={"r": 2**40}), _record(3)])
    sender = _sender(path, stand_in.endpoint)
    assert sender.close(timeout=20) is True
    assert sorted(row["step"] for row in stand_in.rows) == [0, 1, 2, 3]  # before: 0 and 3
    assert sender.moved_fields == 2 and sender.skipped_lines == 0
    said = [line for line in capsys.readouterr().err.splitlines() if "extra_json" in line]
    assert len(said) == 1 and "step 1's 'grad_norm'" in said[0]


# --- Item 5: what the sender keeps when a stream ends without it having closed it ------------

def test_stream_end_a_server_that_ends_the_stream_with_unavailable_is_explained(tmp_path, stand_in):
    import json

    stand_in.ending = "unavailable"
    path = tmp_path / "run-a.rank0.jsonl"
    _append(path, [_record(0)])
    sender = _sender(path, stand_in.endpoint, token="SECRET-TOKEN-123", batch_steps=1)
    try:
        _wait(lambda: sender.acked_offset > 0)
        first_ack = sender.acked_offset
        _append(path, [_record(1), _record(2)])
        _wait(lambda: sender.last_stream_end is not None)
        end = dict(sender.last_stream_end)
        assert end["grpc_status"] == "UNAVAILABLE" and end["grpc_code"] == 14
        assert "going away" in end["details"] and end["error"] == "FlightUnavailableError"
        assert end["stream_opened"] and end["seconds_open"] >= 0
        assert end["batches_sent"] >= 1 and end["bytes_sent"] >= first_ack and end["arrow_bytes_sent"] > 0
        assert end["acks"] >= 1 and end["last_acked_offset"] == first_ack and end["start_offset"] == 0
        assert end["sender_closing"] is False
        assert "SECRET" not in json.dumps(end) and "authorization" not in json.dumps(end).lower()
        # The server takes the rest again: progress clears `last_error`, and the explanation stays.
        stand_in.ending = None
        _wait(lambda: sender.acked_offset == path.stat().st_size)
        _wait(lambda: sender.last_error is None)
        assert sender.last_stream_end["grpc_status"] == "UNAVAILABLE"
    finally:
        stand_in.ending = None
        assert sender.close(timeout=20) is True


def test_stream_end_a_server_that_ends_the_stream_with_ok_is_kept_too(tmp_path, stand_in):
    stand_in.ending = "return"
    path = tmp_path / "run-a.rank0.jsonl"
    _append(path, [_record(0)])
    sender = _sender(path, stand_in.endpoint, batch_steps=1)
    try:
        _wait(lambda: sender.acked_offset > 0)
        _append(path, [_record(1)])
        _wait(lambda: sender.last_stream_end is not None)
        end = sender.last_stream_end
        assert end["grpc_status"] is None and end["grpc_code"] is None and end["error"] == "ConnectionError"
        assert "ended the stream" in end["details"] and end["batches_sent"] >= 2
    finally:
        stand_in.ending = None
        assert sender.close(timeout=20) is True


def test_stream_end_a_stream_the_sender_closed_keeps_nothing(tmp_path, stand_in):
    path = tmp_path / "run-a.rank0.jsonl"
    _append(path, [_record(step) for step in range(5)])
    sender = _sender(path, stand_in.endpoint)
    assert sender.close(timeout=20) is True
    assert sender.last_stream_end is None and sender.last_error is None
