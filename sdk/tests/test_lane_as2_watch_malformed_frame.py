"""Lane AS-2, item 4: what a `--watch` look with the Rust detectors shows when
`tensorquill-server detect --follow` ends as an error on a malformed later frame.

Lane AR-7 chose that a record batch Arrow's reader cannot decode, in a frame after the first, ends
`detect --follow` as an error (exit 1, the message on standard error) and not as a refusal
(`{"follow": {"refused": ...}}`, exit 0), which the follower would answer with a full recomputation.
The author accepted that choice on Nov 27, 2026, with one check asked: the look that meets such a
frame must read as not checked, with the reason.

The frame: a later look's records, as `server_detect.records_stream` makes them, with the record
batch's `bodyLength` set to half and the body cut to that (lane AR-4's AQ5-R1 case, as
`server/tests/lane_ar7_detect_guard_test.rs` builds it). The real sender cannot make one, so the
watcher's `records_stream` is replaced for the frames that hold a record of step 40 or later.

The owner's expectation is held by `test_the_watch_library_reads_a_malformed_later_frame_as_not_checked`
and `test_the_watch_command_reads_a_malformed_later_frame_as_not_checked`. On 1dddcce neither holds:
the look raises `RuntimeError` out of `watch`, and the command ends with exit 2 and
`error: ... detect failed: cannot read the Arrow IPC stream: a record batch in it is malformed ...`
on standard error, and the evidence file left is the one before, which reads as checked and healthy.
Both are strict expected failures until that is decided. The guards beside them pin the frame
(the binary itself ends `--follow` with exit 1 on it) and the harness (the same watch with the frame
left whole is judged as an append).
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from tensorquill import server_detect as sd

from test_lane_af_w_watch import RUN, _binary_or_skip, _children_left, records, write_jsonl  # noqa: E402

SRC = Path(__file__).parents[1] / "src"
FIRST_CUT = 40  # a frame holding a record of this step or later is sent cut short

# What the owner expects and what 1dddcce does instead, for the two strict expected failures.
NOT_YET = ("the owner's check (Nov 27, 2026): a --watch look whose detect --follow ends as an error on a malformed "
           "later frame must read as not checked, with the reason. On 1dddcce the look raises instead: watch() raises "
           "RuntimeError and the command exits 2, leaving the earlier evidence file, which reads as checked")


def cut_short(whole: bytes) -> bytes:
    """`whole`, an Arrow IPC stream, with its first record batch's `bodyLength` halved and its body cut
    to that, then the end-of-stream marker: the stream reader reads the shorter body, and decoding the
    buffers the header names runs past it. As `ipc_cut_short` of lane AR-7's Rust test."""

    def message_at(at: int) -> tuple[int, int]:
        # Each message: 0xFFFFFFFF, the header's length, the header (a flatbuffer `Message`), its body.
        assert whole[at:at + 4] == b"\xff" * 4
        return at + 8, struct.unpack_from("<i", whole, at + 4)[0]

    schema_at, schema_length = message_at(0)
    header_at, header_length = message_at(schema_at + schema_length)
    header = bytearray(whole[header_at:header_at + header_length])
    table = struct.unpack_from("<I", header, 0)[0]
    vtable = table - struct.unpack_from("<i", header, table)[0]
    field = struct.unpack_from("<H", header, vtable + 4 + 2 * 3)[0]  # `bodyLength`, the fourth field
    assert field, "the record batch's header has no bodyLength"
    body = struct.unpack_from("<q", header, table + field)[0]
    half = (body // 2) & ~7
    assert 0 < half < body
    struct.pack_into("<q", header, table + field, half)
    body_at = header_at + header_length
    return whole[:header_at] + bytes(header) + whole[body_at:body_at + half] + b"\xff\xff\xff\xff\x00\x00\x00\x00"


def _holds_a_late_step(listed: list) -> bool:
    return any(isinstance(r.get("step"), int) and r["step"] >= FIRST_CUT for r in listed)


@pytest.fixture
def binary():
    found = _binary_or_skip()
    yield found
    assert _children_left(found) == [], "a detect --follow child outlived its watcher"


def _frames(*streams: bytes) -> bytes:
    return b"".join(struct.pack("<Q", len(stream)) + stream for stream in streams)


def test_the_binary_ends_follow_with_an_error_on_the_cut_frame(binary):
    """A guard on the frame: `detect --samples --follow` answers the whole first frame, then ends with
    exit 1 and the malformed-batch message, as lane AR-7's Rust test has it, and prints no refusal."""
    good = sd.records_stream(records(range(FIRST_CUT)))
    bad = cut_short(sd.records_stream(records(range(FIRST_CUT, FIRST_CUT + 5))))
    done = subprocess.run([binary, "detect", "--samples", "--follow"], input=_frames(good, bad),
                          capture_output=True, timeout=120)
    answers = [json.loads(line) for line in done.stdout.splitlines()]
    assert done.returncode == 1, done.stderr
    assert [a.get("follow", {}).get("mode") for a in answers] == ["full"], answers
    said = done.stderr.decode("utf-8", "replace")
    assert "cannot read the Arrow IPC stream: a record batch in it is malformed" in said, said


def _watch_in_process(tmp_path: Path, monkeypatch, *, cut: bool) -> tuple[list[dict], Path]:
    """`watch(..., detectors="rust")` over a growing `.jsonl` file: 40 steps, then 5 appended after the
    first write. With `cut`, the frame of the append is sent cut short. The evidence of each write."""
    path = tmp_path / f"{RUN}.rank0.jsonl"
    write_jsonl(path, records(range(FIRST_CUT)))
    real = sd.records_stream
    if cut:
        monkeypatch.setattr(sd, "records_stream",
                            lambda given: (lambda listed: cut_short(real(listed)) if _holds_a_late_step(listed)
                                           else real(listed))(list(given)))
    seen: list[dict] = []

    def grow(evidence, written):
        seen.append(json.loads(Path(written).read_text()))
        if len(seen) == 1:
            write_jsonl(path, records(range(FIRST_CUT, FIRST_CUT + 5)), "a")

    ev.watch(path, tmp_path / "out.json", interval=0.0, polls=3, on_update=grow, detectors="rust")
    return seen, path


def test_the_same_watch_with_the_frame_whole_judges_the_append(binary, tmp_path, monkeypatch):
    """A guard on the harness: with the append's frame left whole, the second look is an append and its
    evidence reads as checked."""
    looks: list[dict] = []
    monkeypatch.setattr(sd, "LOOKS", looks)
    seen, _ = _watch_in_process(tmp_path, monkeypatch, cut=False)
    assert [look["mode"] for look in looks] == ["full", "append"], looks
    assert len(seen) == 2 and seen[-1]["run_state"]["state"] == "healthy", seen[-1]["run_state"]
    assert all(seen[-1]["kind_checks"][kind]["value"] != "unavailable" for kind in ev.SERVER_KINDS)


def _reads_as_not_checked_for_the_malformed_frame(evidence: dict) -> None:
    """The owner's expectation of the look's evidence: the kinds the Rust detectors judge read
    unavailable, a `not_checked` entry gives the binary's reason, and the run reads degraded."""
    reasons = [str(item.get("reason")) for item in evidence.get("not_checked") or ()]
    assert any("malformed" in reason for reason in reasons), evidence.get("not_checked")
    unavailable = [kind for kind in ev.SERVER_KINDS if evidence["kind_checks"][kind]["value"] == "unavailable"]
    assert unavailable == list(ev.SERVER_KINDS), evidence["kind_checks"]
    assert evidence["run_state"]["state"] == "degraded", evidence["run_state"]


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason=NOT_YET)
def test_the_watch_library_reads_a_malformed_later_frame_as_not_checked(binary, tmp_path, monkeypatch):
    seen, _ = _watch_in_process(tmp_path, monkeypatch, cut=True)
    assert len(seen) >= 2, f"no evidence was written for the look that met the malformed frame: {len(seen)} writes"
    _reads_as_not_checked_for_the_malformed_frame(seen[-1])


# The command, `python -m tensorquill <file> --watch --detectors rust`, in its own process, with the
# frames that hold a late step cut short as above.
_COMMAND = textwrap.dedent('''
    import runpy, sys
    sys.path.insert(0, sys.argv[1])
    from tensorquill import server_detect as sd
    from test_lane_as2_watch_malformed_frame import cut_short, _holds_a_late_step
    real = sd.records_stream
    def records_stream(given):
        listed = list(given)
        return cut_short(real(listed)) if _holds_a_late_step(listed) else real(listed)
    sd.records_stream = records_stream
    sys.argv = ["tensorquill", *sys.argv[2:]]
    runpy.run_module("tensorquill", run_name="__main__")
''')


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=NOT_YET)
def test_the_watch_command_reads_a_malformed_later_frame_as_not_checked(binary, tmp_path):
    path = tmp_path / f"{RUN}.rank0.jsonl"
    out = tmp_path / "watched.json"
    write_jsonl(path, records(range(FIRST_CUT)))
    command = [sys.executable, "-c", _COMMAND, str(Path(__file__).parent), str(path), "--watch", "--detectors",
               "rust", "--interval", "0.05", "-o", str(out)]
    job = subprocess.Popen(command, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(SRC)},
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert job.stdout is not None
        updates = 0
        for line in job.stdout:
            if line.startswith("updated "):
                updates += 1
                if updates == 1:
                    write_jsonl(path, records(range(FIRST_CUT, FIRST_CUT + 5)), "a")
                else:
                    break
        ended = job.wait(60) if updates < 2 else job.poll()  # the output ended: so does the command
        said = job.stderr.read() if ended is not None else ""
        left = json.loads(out.read_text()) if out.exists() else None
        assert ended is None, (f"the watch command ended with exit {ended} after {updates} writes: {said.strip()}; "
                               f"the evidence file left reads {left and left['run_state']}, with "
                               f"{left and sum(left['kind_checks'][k]['value'] == 'unavailable' for k in ev.SERVER_KINDS)} "
                               f"of {len(ev.SERVER_KINDS)} Rust kinds unavailable")
        assert updates == 2
        _reads_as_not_checked_for_the_malformed_frame(json.loads(out.read_text()))
    finally:
        if job.poll() is None:
            job.terminate()
            try:
                job.wait(30)
            except subprocess.TimeoutExpired:
                job.kill()
                job.wait()
        for pipe in (job.stdout, job.stderr):
            if pipe is not None:
                pipe.close()
