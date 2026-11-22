"""Lane C of wave AF: a file rewritten in place under `--watch` (WAVE_AF_CONTRACT.md, section 3, EV-07).

The watcher took a `.jsonl` file as unchanged when its length and both 4 KiB ends were what they
were. Its modification time (`st_mtime_ns`) now joins that signature: a rewrite of the same length
and ends with another modification time is read again, by both engines. An append is still read
as an append, and a `touch` that changes no byte gives the same evidence bytes.

Times are set with `os.utime(ns=...)`, never by waiting, and what is read is counted, not timed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tensorquill import evidence as ev

STEPS = 240
SPIKE_AT = 150  # in the middle: more than 4 KiB from either end


def line(step: int, norm: str = "1.00") -> str:
    """One record whose `grad_norm` is written as text of a fixed width, so a rewrite of it keeps
    the file's length."""
    return ('{"schema_version": 1, "run_id": "w", "rank": 0, "attempt": 0, "step": %d, "grad_norm": %s, '
            '"scalars": {"loss": 0.5}}\n' % (step, norm))


def write_run(path: Path) -> None:
    path.write_text("".join(line(step) for step in range(STEPS)))


def rewrite_in_place(path: Path, step: int, norm: str) -> None:
    """Change one record in place, its length kept, and give the file a later modification time."""
    data = path.read_bytes()
    old, new = line(step).encode(), line(step, norm).encode()
    assert len(old) == len(new) and data.count(old) == 1
    at = data.index(old)
    assert at > ev._Tail.CHECKED_BYTES and len(data) - at - len(old) > ev._Tail.CHECKED_BYTES
    before = os.stat(path)
    with path.open("r+b") as file:
        file.seek(at)
        file.write(new)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    after = os.stat(path)
    assert after.st_size == before.st_size and after.st_mtime_ns != before.st_mtime_ns


def touch(path: Path) -> None:
    before = os.stat(path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


class CountedLoads:
    """Counts the lines the watcher parses: `json.loads` as `evidence` calls it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real = ev.json.loads

        def loads(text, *args, **kwargs):
            self.count += 1
            return real(text, *args, **kwargs)

        monkeypatch.setattr(ev.json, "loads", loads)


def test_a_rewrite_of_the_same_length_and_ends_is_read_again(tmp_path):
    path = tmp_path / "w.jsonl"
    write_run(path)
    tail = ev._Tail(path, None)
    assert tail.poll() and len(tail.records) == STEPS
    rewrite_in_place(path, SPIKE_AT, "9.99")
    assert tail.poll(), "the rewrite was not seen"
    assert tail.restarts == 1
    assert len(tail.records) == STEPS and tail.records[SPIKE_AT]["grad_norm"] == 9.99


def test_an_append_is_still_read_as_an_append(tmp_path, monkeypatch):
    path = tmp_path / "w.jsonl"
    write_run(path)
    tail = ev._Tail(path, None)
    tail.poll()
    counted = CountedLoads(monkeypatch)
    before = os.stat(path)
    with path.open("a") as file:
        file.write("".join(line(step) for step in range(STEPS, STEPS + 5)))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))  # an append moves the time too
    assert tail.poll()
    assert tail.restarts == 0 and counted.count == 5, "the append was read again from the start"
    assert [r["step"] for r in tail.records] == list(range(STEPS + 5))


def test_a_line_finished_after_a_partial_one_is_an_append(tmp_path, monkeypatch):
    """A last line with no newline is not read; its end arriving later grows the file, so it is an
    append: the size compared is the file's at the last read, not what was read of it."""
    path = tmp_path / "w.jsonl"
    write_run(path)
    whole = line(STEPS)
    with path.open("a") as file:
        file.write(whole[:20])
    tail = ev._Tail(path, None)
    tail.poll()
    assert len(tail.records) == STEPS
    counted = CountedLoads(monkeypatch)
    with path.open("a") as file:
        file.write(whole[20:])
    touch(path)
    assert tail.poll()
    assert tail.restarts == 0 and counted.count == 1 and len(tail.records) == STEPS + 1


def test_a_touch_reads_the_file_again_and_finds_the_same_records(tmp_path):
    path = tmp_path / "w.jsonl"
    write_run(path)
    tail = ev._Tail(path, None)
    tail.poll()
    kept = [dict(r) for r in tail.records]
    touch(path)
    tail.poll()
    assert tail.records == kept


def _engine(engine: str) -> dict:
    if engine == "rust":
        from tensorquill import server_detect

        if server_detect.find_binary() is None:
            pytest.skip("no tensorquill-server binary")
        pytest.importorskip("pyarrow")
    return {"detectors": engine}


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_the_watcher_judges_a_rewrite_in_place_again(tmp_path, engine):
    """`watch` on the file, with either engine: the second look sees the rewrite and judges the
    new values, as one read of the rewritten file does."""
    options = _engine(engine)
    path = tmp_path / "w.jsonl"
    write_run(path)
    seen: list[dict] = []

    def update(evidence, written):
        seen.append(json.loads(written.read_text()))
        if len(seen) == 1:
            rewrite_in_place(path, SPIKE_AT, "99.0")

    ev.watch(path, tmp_path / "out.json", interval=0.0, polls=2, on_update=update, **options)
    assert len(seen) == 2, "the rewrite was not read: nothing was written for it"
    once = ev.build_evidence(path, **options)
    assert [i["kind"] for i in seen[1]["incidents"]] == [i["kind"] for i in once["incidents"]]
    assert seen[1]["incidents"] != seen[0]["incidents"]


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_a_touch_gives_the_same_evidence_bytes(tmp_path, engine):
    options = _engine(engine)
    path = tmp_path / "w.jsonl"
    write_run(path)
    out = tmp_path / "out.json"
    written: list[bytes] = []

    def update(evidence, where):
        written.append(where.read_bytes())
        if len(written) == 1:
            touch(path)

    ev.watch(path, out, interval=0.0, polls=2, on_update=update, **options)
    # The touch was read again (a second write), and it changed no byte of the evidence.
    assert len(written) == 2 and written[1] == written[0]
    assert out.read_bytes() == written[0]
