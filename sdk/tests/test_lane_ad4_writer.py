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
