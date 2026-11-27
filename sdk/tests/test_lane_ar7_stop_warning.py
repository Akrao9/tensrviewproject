"""Lane AR-7, item A: the stop's warning says what the records file holds, for each cause of the stop.

Before, `_stop` said for every cause that "the steps already handed to the writer are written, no
later step is". After a sink failure that is not so: the writer passes over every block queued
after the write that raised (`_drain`), so steps handed to the writer are lost. Each cause is held
here to what the file contains after it:

- the sink raised on the writer thread: the file ends at the last write that returned, and the
  steps still queued are passed over;
- the sink's close raised at `close()` (a stream's close too): every write had returned;
- the writer fell a full queue behind: what was queued is still written as the sink takes it;
- the writer had not finished at `close()`: what it was handed is written only as far as it gets.

No server binary.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture

DEADLINE = 20.0

# The clause every cause's text shares, which other tests pin.
SHARED = ("no later step is written, nor a scalar still pending, and the records file gets no end marker, so "
          "it reads as cut off. The cause is session.writer_error.")
PASSED_OVER = ("Training goes on and does not wait; the records written before that are kept, the steps still "
               "queued for the writer are passed over, " + SHARED)
STILL_WRITTEN = ("Training goes on and does not wait; the steps already handed to the writer are still written "
                 "while the sink takes them, " + SHARED)
UNFINISHED = ("Training goes on and does not wait; the writer has not finished, so the steps already handed to "
              "it are written only as far as it gets before the process ends, " + SHARED)
OLD = "the steps already handed to the writer are written, no later step is"


def _told(caught) -> list[str]:
    return [str(w.message) for w in caught if "TensorQuill" in str(w.message)]


def _steps(path: Path) -> list[int]:
    return [json.loads(line)["step"] for line in path.read_text().splitlines() if line.strip()
            and json.loads(line).get("record") != "scalars"]


def _train(model: nn.Module, opt: torch.optim.Optimizer, n: int) -> None:
    for _ in range(n):
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()


class HeldFailingJsonl(tensorquill.JsonlSink):
    """`good` writes, then one that waits for `gate` and raises."""

    def __init__(self, path: Path, good: int) -> None:
        super().__init__(path)
        self.good = good
        self.gate = threading.Event()
        self.waiting = threading.Event()

    def write(self, records):
        if self.good <= 0:
            self.waiting.set()
            self.gate.wait(DEADLINE)
            raise OSError("disk gone")
        self.good -= 1
        super().write(records)


def test_after_a_sink_failure_the_warning_says_the_queued_steps_are_passed_over_as_the_file_shows(tmp_path):
    """Steps 3 to 9 are handed to the writer while its fourth write is held; that write raises, the
    writer passes over the rest, and the file holds 0, 1 and 2. The warning must not say that the
    steps handed to the writer are written."""
    path = tmp_path / "run.rank0.jsonl"
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = HeldFailingJsonl(path, good=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, run_id="run", every=1, heartbeat_seconds=None,
                                   stack_seconds=None)
        _train(model, opt, 10)
        assert sink.waiting.wait(DEADLINE)
        handed_after = session._queue.qsize()
        assert handed_after >= 5  # steps after 3 are queued for the writer, which holds step 3's write
        sink.gate.set()
        deadline = time.monotonic() + DEADLINE
        while session._error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        while not session._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)  # the writer passes over what was queued after the failing write
        session.close()
    assert _steps(path) == [0, 1, 2]
    told = _told(caught)
    assert told == [f"TensorQuill stopped capturing for this session: its sink raised OSError: disk gone. "
                    f"{PASSED_OVER}"]
    assert OLD not in told[0]


def test_a_writer_behind_still_writes_what_was_queued_and_says_so(monkeypatch):
    """The writer fell a full queue behind: the blocks queued before the stop are written once the sink
    returns, as the warning says."""
    monkeypatch.setattr(capture, "QUEUE_STEPS", 2)
    release = threading.Event()

    class Slow(tensorquill.MemorySink):
        def write(self, records):
            release.wait(DEADLINE)
            super().write(records)

    sink = Slow()
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=sink, every=1, heartbeat_seconds=None, stack_seconds=None)
        deadline = time.monotonic() + DEADLINE
        while not session._stopped and time.monotonic() < deadline:
            _train(model, opt, 1)
        assert session._stopped
        handed = session._next_step - 1  # the step whose block did not fit is not among them
        release.set()
        session.close()
    written = [r["step"] for r in sink.records]
    assert written and written == list(range(len(written))) and len(written) < handed + 1
    told = _told(caught)
    assert len(told) == 1 and told[0].endswith(STILL_WRITTEN), told
    assert "behind training" in told[0]


def test_a_writer_unfinished_at_close_says_its_steps_are_written_only_as_far_as_it_gets(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "CLOSE_SECONDS", 0.3)
    path = tmp_path / "slow.rank0.jsonl"
    release = threading.Event()

    class Slow(tensorquill.JsonlSink):
        def write(self, records):
            release.wait(DEADLINE)
            super().write(records)

    session = tensorquill.init(nn.Linear(2, 2), sink=Slow(path), run_id="slow", every=50, auto_step=False,
                               heartbeat_seconds=None, stack_seconds=None)
    session.step()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.close()
    release.set()
    told = _told(caught)
    assert len(told) == 1 and told[0].endswith(UNFINISHED), told
    assert "had not finished" in told[0]


def test_a_sink_whose_close_raises_says_what_was_written_is_kept(tmp_path):
    path = tmp_path / "shut.rank0.jsonl"

    class FailingClose(tensorquill.JsonlSink):
        def close(self):
            super().close()
            raise OSError("close failed")

    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = tensorquill.init(model, sink=FailingClose(path), run_id="shut", every=1, heartbeat_seconds=None,
                                   stack_seconds=None)
        _train(model, opt, 4)
        session.close()
    assert _steps(path) == [0, 1, 2, 3]
    told = _told(caught)
    assert told == [f"TensorQuill stopped capturing for this session: close failed. {PASSED_OVER}"]


@pytest.mark.parametrize("text", [PASSED_OVER, STILL_WRITTEN, UNFINISHED])
def test_every_cause_s_text_shares_the_clause_other_tests_pin(text):
    assert text.endswith(SHARED)
    assert "nor a scalar still pending" in text
