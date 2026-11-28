"""The cases of `WAVE_AS_CONTRACT.md` (the folder route): `fixtures/folder_route/cases.json`.

The page states how `tensorquill-server` reads a folder of the SDK's rank files itself (step 2 of the local-first
0.1.0). Nothing of the route is built at the base (`1dddcce`). What runs here:

- The page and the cases: every case named in its own section, every row of the table "Where each decision
  stands" present, and the cases file strict JSON that cargo can read too.
- Conversion (section 2): every conversion case recomputed from today's `flight.py`, which is what the server
  stores today for a file sent with `send_file`: without pyarrow for the rows Arrow does not touch, with it for
  all. The line cases on `flight.read_lines`, a torn tail on `sink.JsonlSink`, the names on `sink.py`'s and
  `manifest.py`'s rules. These are pins: today's Python, which the Rust converter is held to.
- Every hand-written case (end states, beats, store keys, restarts, the end message, the first-open figure)
  against the page's rules, written plainly below as functions. They are the page in code, not product code.
- Targets, strict expected failures that name their lane and check the product path they name: the server's
  `--records` and what it stores, the restart through it and the folder never written into (lanes FR-1 and
  FR-2); the stop notice, the beat file, the end batch on the stream and the `atexit` close (lane FR-3). The
  lane turns each into an ordinary test by removing its mark when its change makes it pass.

The targets that need the server binary skip without it; those that need pyarrow skip without it. CI's server
job has both.

The page is kept outside this repository. The tests that read it take it through `conftest.contract_page`,
inside the test and after the checks that need no page, so in a checkout with no page (CI's among them) those
tests are skipped and every other test here runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import conftest
import tensorquill
from tensorquill import flight, manifest, sink

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FOLDER = HERE / "fixtures" / "folder_route"
TEXT = (FOLDER / "cases.json").read_text()
DOCUMENT = json.loads(TEXT)
CASES = DOCUMENT["cases"]
NAMES = DOCUMENT["names"]
RULES = DOCUMENT["rules"]
DEADLINE = 20.0  # seconds: a wait for something that never comes ends here

LANE_CONVERT = ("lanes FR-1 (the converter, WAVE_AS_CONTRACT.md section 2) and FR-2 (the folder reader, section 1): "
                "tensorquill-server --records is not built")
LANE_FOLDER = "lane FR-2 (the folder reader, WAVE_AS_CONTRACT.md sections 1, 4 and 10): tensorquill-server --records is not built"
LANE_SDK_STOP = "lane FR-3 (the SDK, WAVE_AS_CONTRACT.md section 5.1): no stop notice is written at close"
LANE_SDK_BEAT = "lane FR-3 (the SDK, WAVE_AS_CONTRACT.md section 5.2): no beat file is written"
LANE_SDK_END = "lane FR-3 (the SDK, WAVE_AS_CONTRACT.md section 9): no end batch is sent on the stream"
LANE_SDK_EXIT = "lane FR-3 (the SDK, WAVE_AS_CONTRACT.md section 9, Q3 b): no atexit close"
LANE_READER = "lane FR-3 (sink.read_end_marker, WAVE_AS_CONTRACT.md sections 5.1 and 11): the stop notice is not read"


def of(kind):
    return [c for c in CASES if c["kind"] == kind]


def ids(cases):
    return [c["name"] for c in cases]


def _make_cases():
    spec = importlib.util.spec_from_file_location("folder_route_make_cases", FOLDER / "make_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAKE = _make_cases()


def page():
    """The contract page's text, or the calling test skipped where no page is at hand (conftest.contract_page)."""
    return conftest.contract_page(DOCUMENT["contract"])


def section_text(number):
    text = page()
    start = text.index(f"\n## {number}. ")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def line_bytes(case):
    return bytes.fromhex(case["line_hex"]) if "line_hex" in case else case["line"].encode("utf-8")


def same_row(got, want):
    """Rows as cases.json writes them, compared with each value's type (1 is not 1.0)."""
    return json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)


# --- The page and the cases -------------------------------------------------------------------------------


def test_the_document_and_the_page():
    assert DOCUMENT["contract"] == "WAVE_AS_CONTRACT.md"
    assert len(DOCUMENT["base"]) == 40
    names = ids(CASES)
    assert len(names) == len(set(names))
    for case in CASES:
        assert {"name", "section", "kind", "about"} <= set(case), case.get("name")
    # From here on the page is read: a checkout with no page skips the rest.
    text = page()
    for case in CASES:
        assert f"`{case['name']}`" in section_text(case["section"]), f"{case['name']} is not named in section {case['section']}"
    assert text.startswith("# Wave AS contract: the folder route")
    assert "## Where each decision stands" in text
    table = text[text.index("## Where each decision stands"):text.index("\n## 1. ")]
    for row in [f"D{n}" for n in range(1, 16)] + [f"R{n}" for n in range(1, 8)] + [f"Q{n}" for n in range(1, 18)]:
        assert f"\n| {row} |" in table, row
    rows = [line for line in table.splitlines() if line.startswith("| ") and not line.startswith("| #")]
    assert sum("**Decided" in line for line in rows) == 15
    assert sum("**Question**" in line for line in rows) == 17
    for number in range(1, 15):
        assert f"\n## {number}. " in text, number
    assert "Nothing here moves `PROTOCOL`" in text


def test_the_page_cites_the_read_api_contract_and_does_not_restate_its_rules():
    """The seam with lane AS-5: the page names the read API contract's topics and states none of their rules."""
    text = page()
    assert "the read API contract" in text and "`WAVE_AN_CONTRACT.md`" in text
    for word in ("X-Frame-Options", "/healthz", "50052", "allowlist of names"):
        assert word not in text, word


def test_the_cases_file_is_strict_json():
    """cargo reads the same file: no NaN or Infinity token, and no lone surrogate once it is read."""

    def refuse(token):
        raise ValueError(f"{token} is not strict JSON")

    json.loads(TEXT, parse_constant=refuse)

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    for text in strings(DOCUMENT):
        assert not any(0xD800 <= ord(c) <= 0xDFFF for c in text), text[:80]


# --- Section 2: conversion, as today's sender makes it --------------------------------------------------------


CONVERSION = of("conversion")


@pytest.mark.parametrize("case", CONVERSION, ids=ids(CONVERSION))
def test_today_s_python_gives_each_conversion_case(case):
    """Without pyarrow: the skip, the left-out count, and the row where Arrow's check moves nothing."""
    records, skipped = flight.parse_lines([line_bytes(case)], case["run_id"])
    assert skipped == case["skipped"]
    left_out = 0
    for record in records:
        try:
            flight.check(record, None, flight._sender_fields(record))
        except flight.MalformedRecordError:
            left_out += 1
    assert left_out == case["left_out"]
    sent = len(records) - left_out
    assert (case["row"] is not None) == (sent == 1)
    if sent and not case["arrow_moves"]:
        assert same_row(MAKE.encode_row(flight._row(records[0])), case["row"])
    if records and not case["arrow_moves"]:  # without the schema, only where Arrow's check moves nothing
        assert _record_to_row(records[0], None, case["row"]) == case["record_to_row"]


def _record_to_row(record, schema, row):
    """Section 2.1: what `flight.record_to_row` makes of a record against the case's row, as the case writes it."""
    try:
        return "same" if same_row(MAKE.encode_row(flight.record_to_row(record, schema)), row) else "differs"
    except flight.MalformedRecordError as error:
        return {"refuses": error.field}


@pytest.mark.parametrize("case", CONVERSION, ids=ids(CONVERSION))
def test_the_sender_stores_each_conversion_case(case):
    """With pyarrow: the row today's sender sends (`rows_to_send`), as Arrow reads it back."""
    pytest.importorskip("pyarrow")
    records, _ = flight.parse_lines([line_bytes(case)], case["run_id"])
    moved, refused = [], []
    batch, _, left_out = flight.rows_to_send(records, flight.flight_schema(), moved, refused)
    rows = batch.to_pylist()
    assert left_out == case["left_out"]
    assert [key for _, key in moved] == case["moved"]
    assert same_row(MAKE.encode_row(rows[0]), case["row"]) if rows else case["row"] is None
    if records:  # with the schema, record_to_row gives the sender's row wherever it takes the record
        assert _record_to_row(records[0], flight.flight_schema(), case["row"]) == case["record_to_row"]


(MIXED,) = of("mixed")


def test_a_file_of_step_and_scalar_only_records_is_read_in_one_pass():
    """Section 2.5, without pyarrow: every line is a record of the run, and one is the scalar-only record."""
    lines = MIXED["file"].encode("utf-8").split(b"\n")[:-1]
    records, skipped = flight.parse_lines(lines, MIXED["run_id"])
    assert skipped == MIXED["skipped"]
    assert [r["step"] for r in records] == [r["step"] for r in MIXED["rows"]]
    assert [flight.is_scalar_record(r) for r in records] == [r["seq"] is not None for r in MIXED["rows"]]


def test_a_file_of_step_and_scalar_only_records_is_sent_as_one_batch():
    """Section 2.5, with pyarrow: one batch, the scalar-only row's step not among the training steps."""
    pytest.importorskip("pyarrow")
    lines = MIXED["file"].encode("utf-8").split(b"\n")[:-1]
    records, _ = flight.parse_lines(lines, MIXED["run_id"])
    batch, steps, left_out = flight.rows_to_send(records, flight.flight_schema())
    assert steps == MIXED["training_steps"] and left_out == MIXED["left_out"]
    assert [{k: r[k] for k in ("step", "seq", "record_version")} for r in batch.to_pylist()] == MIXED["rows"]


LINES = of("lines")


@pytest.mark.parametrize("case", LINES, ids=ids(LINES))
def test_today_s_python_reads_each_lines_case(case, tmp_path):
    path = tmp_path / f"{case['run_id']}.rank0.jsonl"
    offset = 0
    for read in case["reads"]:
        path.write_bytes(read["file"].encode("utf-8"))
        lines, offset = flight.read_lines(path, offset, 50, case["max_bytes"])
        assert [line.decode("utf-8") for line in lines] == read["lines"]
        assert offset == read["offset"]


def test_a_torn_tail_is_given_its_newline_and_skipped(tmp_path):
    """Section 6: the next session's sink repairs the fragment, which a reader then skips and counts."""
    (case,) = of("torn")
    path = tmp_path / f"{case['run_id']}.rank0.jsonl"
    path.write_bytes(case["file"].encode("utf-8"))
    writer = sink.JsonlSink(path)
    writer.write([case["appended"]])
    writer.close()
    lines, _ = flight.read_lines(path, 0, 50, 1 << 20)
    records, skipped = flight.parse_lines(lines, case["run_id"])
    assert [r["step"] for r in records] == case["expect"]["records"]
    assert skipped == case["expect"]["skipped"]


def test_the_names_today_s_readers_take():
    (case,) = [c for c in of("names") if c["name"] == "names_records_and_second_writer"]
    for name, parts in case["files"].items():
        got = sink.records_file_parts(name)
        assert (list(got) if got is not None else None) == parts, name


def test_the_proposed_names_are_taken_for_nothing_else():
    """Section 5.2: the beat file and the stop notice match no name a reader of a folder takes today."""
    (case,) = [c for c in of("names") if c["name"] == "names_beat_file_and_stop_notice_are_no_records_file"]
    run, rank, pid = case["run_id"], case["rank"], case["pid"]
    stems = [f"{run}.rank{rank}", f"{run}.rank{rank}.{pid}"]
    made = [NAMES[key].format(stem=stem) for stem in stems for key in ("beat_file", "stop_notice")]
    assert sorted(made) == sorted(case["proposed"])
    for name in case["proposed"]:
        assert sink.records_file_parts(name) is None, name
        assert manifest._RANK_FILE.fullmatch(name) is None, name
        for suffix in (sink.RECORDS_SUFFIX, sink.END_SUFFIX, manifest.MANIFEST_SUFFIX, ".offset"):
            assert not name.endswith(suffix), (name, suffix)
    records = Path(f"{stems[0]}.jsonl")
    taken = case["taken"]
    assert records.name == taken[0] and sink.records_file_parts(records.name) == (run, rank, None)
    assert sink.end_marker_path(records).name == taken[1] == NAMES["end_marker"].format(stem=stems[0])
    assert flight.offset_path(records).name == taken[2]
    assert f"{run}{manifest.MANIFEST_SUFFIX}" == taken[3]


# --- Sections 5 and 6: the folder's end states, the beat file, a killed job -----------------------------------


def folder_state(folder):
    """The rank's state from what the folder holds (sections 5.1, 5.3, 5.4 and 6, the recommended options)."""
    stopped = {"state": "capture_stopped", "reason": "sink_error", "step": 7}
    if folder.get("stop") == "present":  # a notice whose size is the file's; a stale one is no notice
        return {"state": "capture_stopped", "ended": stopped}
    marker = folder.get("marker")
    if marker == "closed":
        return {"state": "finished", "ended": {"state": "closed", "outcome": "closed"}}
    if marker == "raised":
        return {"state": "finished", "ended": {"state": "closed", "outcome": "raised", "exception": "RuntimeError"}}
    beat = folder.get("beat", "none")
    if beat == "none" or folder.get("fs", "local") == "network":
        return {"state": "active", "liveness": "not_checked"}
    floor = RULES["silent_floor_seconds"]
    if beat == "stale":
        return {"state": "lost" if folder["stale_for"] >= floor else "reconnecting"}
    return {"state": "silent" if folder.get("work_still_for", 0) >= floor else "active"}


END_STATES = of("end_state")


@pytest.mark.parametrize("case", END_STATES, ids=ids(END_STATES))
def test_each_end_state_case_follows_the_page_s_rules(case):
    assert folder_state(case["folder"]) == case["expect"]


def _records(path, run_id, steps):
    path.write_text("".join(json.dumps({"schema_version": 1, "run_id": run_id, "rank": 0, "step": s}) + "\n"
                            for s in steps))


@pytest.mark.parametrize("marker", ["closed", "raised", "written_after_close"])
def test_today_s_python_reads_the_end_marker_as_the_cases_say(marker, tmp_path):
    """A pin: `sink.read_end_marker` already gives what the end-state cases read from a marker."""
    path = tmp_path / "fr-marker.rank0.jsonl"
    _records(path, "fr-marker", range(3))
    error = RuntimeError("boom") if marker == "raised" else None
    assert sink.write_end_marker(path, run_id="fr-marker", rank=0, attempt=1, exception=error)
    if marker == "written_after_close":
        with path.open("a") as file:
            file.write(json.dumps({"schema_version": 1, "run_id": "fr-marker", "rank": 0, "step": 3}) + "\n")
    read = sink.read_end_marker(path)
    if marker == "written_after_close":
        assert read["state"] == "written_after_close"
        assert folder_state({"marker": marker, "beat": "fresh"})["state"] == "active"
        return
    case = next(c for c in END_STATES if c["folder"] == {"marker": marker})
    want = case["expect"]["ended"]
    assert read["state"] == "closed" and read["outcome"] == want["outcome"]
    assert read.get("exception") == want.get("exception")


def _stop_notice(path, *, reason="sink_error", step=7, records_bytes=None):
    stem = path.name[: -len(sink.RECORDS_SUFFIX)]
    notice = path.with_name(NAMES["stop_notice"].format(stem=stem))
    notice.write_text(json.dumps({"marker": NAMES["stop_marker"], "version": 1, "run_id": "fr-stop", "rank": 0,
                                  "attempt": 1, "reason": reason, "step": step,
                                  "records_bytes": path.stat().st_size if records_bytes is None else records_bytes}))
    return notice


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_READER)
def test_a_python_reader_reads_the_stop_notice(tmp_path):
    path = tmp_path / "fr-stop.rank0.jsonl"
    _records(path, "fr-stop", range(8))
    _stop_notice(path)
    read = sink.read_end_marker(path)
    assert read["state"] == "capture_stopped", f"a stop notice beside the file reads {read['state']!r}"
    assert (read.get("reason"), read.get("step")) == ("sink_error", 7)


def beat_freshness(interval, looks):
    """Section 5.3 (a): a beat is a change of the counter seen at a look, timed by the server's clock; the first
    look counts; stale once none was seen for max(3 intervals, 30 s). No time inside the file is read."""
    window = max(RULES["beat_stale_intervals"] * interval, RULES["beat_stale_min_seconds"])
    seen_at, last, out = None, None, []
    for look in looks:
        if seen_at is None or look["beat"] != last:
            seen_at, last = look["at"], look["beat"]
        out.append("fresh" if look["at"] - seen_at < window else "stale")
    return out


BEATS = of("beat")


@pytest.mark.parametrize("case", BEATS, ids=ids(BEATS))
def test_each_beat_case_follows_the_page_s_rules(case):
    assert beat_freshness(case["interval_seconds"], case["looks"]) == case["expect"]


def test_a_wall_time_in_a_beat_file_changes_nothing():
    case = next(c for c in BEATS if c["name"] == "beat_a_wall_time_in_the_file_is_never_read")
    without = [{k: v for k, v in look.items() if k != "written_at"} for look in case["looks"]]
    assert beat_freshness(case["interval_seconds"], without) == case["expect"]


def test_the_stale_window_is_today_s_heartbeat_window():
    """The beat file's window is the one `watchdog.rs` already gives a heartbeat."""
    text = (ROOT / "server" / "src" / "watchdog.rs").read_text()
    assert "const HEARTBEAT_STALE_INTERVALS: f64 = 3.0;" in text
    assert "const HEARTBEAT_MIN_STALE: f64 = 30.0;" in text
    assert "pub const DEFAULT_SILENT_FLOOR: Duration = Duration::from_secs(300);" in text
    assert flight.HEARTBEAT_SECONDS == 15.0


class FailingJsonl(sink.JsonlSink):
    """Writes its first `good` blocks, then raises, as a disk that fills up."""

    def __init__(self, path, good):
        super().__init__(path)
        self.good = good

    def write(self, records):
        if self.good <= 0:
            raise OSError("disk gone")
        self.good -= 1
        super().write(records)


def _steps(model, opt, n=1):
    for _ in range(n):
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()


def _stopped_and_closed(tmp_path, run_id):
    """A file-only session whose sink fails at its second write, stepped until capture stops, then closed."""
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / f"{run_id}.rank0.jsonl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, run_id=run_id, every=1, sink=FailingJsonl(path, good=1))
        deadline = time.monotonic() + DEADLINE
        while not session._stopped and time.monotonic() < deadline:
            _steps(model, opt)
            time.sleep(0.01)  # the writer's turn
        stopped = session._stopped
        session.close()
    assert stopped, "capture did not stop"
    return path


def test_a_stopped_session_writes_no_end_marker(tmp_path):
    """A pin (R6, kept by section 5.1 a): a stopped session leaves no end marker, so no reader takes it for closed."""
    path = _stopped_and_closed(tmp_path, "fr-pin-stop")
    assert not sink.end_marker_path(path).exists()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_SDK_STOP)
def test_a_stopped_session_leaves_a_stop_notice_at_close(tmp_path):
    path = _stopped_and_closed(tmp_path, "fr-stop-notice")
    stem = path.name[: -len(sink.RECORDS_SUFFIX)]
    notice = path.with_name(NAMES["stop_notice"].format(stem=stem))
    assert notice.exists(), f"no {notice.name} beside the records file after a stopped session's close"
    document = json.loads(notice.read_text())
    assert document["marker"] == NAMES["stop_marker"] and document["version"] == 1
    assert document["reason"] == "sink_error" and document["records_bytes"] == path.stat().st_size
    assert document["step"] is None or isinstance(document["step"], int)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_SDK_BEAT)
def test_a_session_writes_its_beat_file(tmp_path):
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "fr-beat.rank0.jsonl"
    beat = path.with_name(NAMES["beat_file"].format(stem="fr-beat.rank0"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session = tensorquill.init(model, run_id="fr-beat", every=1, sink=sink.JsonlSink(path))
        try:
            _steps(model, opt, 2)
            deadline = time.monotonic() + 5.0
            while not beat.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert beat.exists(), f"no {beat.name} beside the records file within 5 s of the session's start"
            document = json.loads(beat.read_text())
        finally:
            session.close()
    assert document["marker"] == NAMES["beat_marker"] and document["version"] == 1
    assert (document["run_id"], document["rank"], document["attempt"]) == ("fr-beat", 0, session.attempt)
    assert document["phase"] == "train" and isinstance(document["work"], int) and document["beat"] >= 1
    assert document["interval_seconds"] == 15.0


# --- Section 3: the default derived store's key -----------------------------------------------------------------


def store_key(spec):
    """Section 3.2 (a): 16 hex digits of the SHA-256 of the real path, with a newline and the machine's name
    where the cache folder is on a network filesystem or its type cannot be read (D3)."""
    text = spec["real"] + ("\n" + spec["host"] if spec["fs"] in ("network", "unknown") else "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: RULES["store_key_hex_digits"]]


STORE_KEYS = of("store_key")


@pytest.mark.parametrize("case", STORE_KEYS, ids=ids(STORE_KEYS))
def test_each_store_key_case_follows_the_page_s_rules(case):
    assert (store_key(case["first"]) == store_key(case["second"])) is case["same"]


def test_one_folder_by_two_paths_has_one_real_path(tmp_path):
    """The key is of the real path, so a folder reached through a symbolic link is the same store."""
    folder = tmp_path / "runs" / "exp1"
    folder.mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(folder, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links cannot be made here")
    assert os.path.realpath(alias) == os.path.realpath(folder)


# --- Section 10: a restart ---------------------------------------------------------------------------------------


def restart_lines(case):
    if "attempts" in case:
        return [(attempt, s) for attempt, first, end in case["attempts"] for s in range(first, end)]
    return [(1, s) for s in range(case["lines"])]


def restart(case, offset_first=False):
    """The first server read `after_lines` lines, stored `parts` parts of `flush` lines and wrote the offset of
    `offset_parts` of them (or, with `offset_first`, the offset of every line it read: the wrong order). The
    second reads from the saved offset to the end and drops a row at or below its attempt's watermark."""
    lines, flush, kill = restart_lines(case), case["flush"], case["kill"]
    stored = list(lines[: kill["parts"] * flush])
    offset = kill["after_lines"] if offset_first else kill["offset_parts"] * flush
    assert offset_first or offset <= len(stored), "an offset past what is stored"
    watermark = {}
    for attempt, s in stored:
        watermark[attempt] = max(s, watermark.get(attempt, -1))
    for attempt, s in lines[offset:]:
        if s > watermark.get(attempt, -1):
            stored.append((attempt, s))
            watermark[attempt] = s
    return {"each_step_once": sorted(stored) == sorted(set(stored)) and set(stored) <= set(lines),
            "lost": sorted(set(lines) - set(stored))}


RESTARTS = of("restart")


@pytest.mark.parametrize("case", RESTARTS, ids=ids(RESTARTS))
def test_each_restart_case_follows_the_page_s_rules(case):
    assert restart(case) == case["expect"]


def test_an_offset_written_before_its_part_would_lose_steps():
    """Why the offset moves only after the part: the other order loses the rows read and not yet stored."""
    case = next(c for c in RESTARTS if c["name"] == "restart_after_a_part_before_its_offset")
    assert restart(case, offset_first=True)["lost"] == [(1, s) for s in range(10, 15)]


# --- Section 9: the end message ---------------------------------------------------------------------------------


def end_reads(stream):
    """What the server reads at a stream's end (section 9, the owner's Q1 a, Q2 a)."""
    if stream["close"] == "error":
        return {"state": "reconnecting", "then": "lost"}
    end = stream.get("rank_end")
    if end is not None:
        return {"state": "finished" if end["state"] == "closed" else "capture_stopped", "ended": end}
    if stream["end_version"]:
        return {"state": "reconnecting", "then": "lost",
                "ended": {"state": "without_notice", "seen": "clean_close_without_end"}}
    return {"state": "finished"}


END_MESSAGES = of("end_message")


@pytest.mark.parametrize("case", END_MESSAGES, ids=ids(END_MESSAGES))
def test_each_end_message_case_follows_the_page_s_rules(case):
    assert end_reads(case["stream"]) == case["expect"]


def test_the_folder_and_the_stream_say_the_same_end():
    by_name = {c["name"]: c for c in CASES}
    assert by_name["folder_closed_reads_finished"]["expect"] == by_name["end_closed"]["expect"]
    assert (by_name["folder_stop_notice_reads_capture_stopped"]["expect"]
            == by_name["end_capture_stopped_every_beat_lost"]["expect"])


def _metadata_server():
    """An in-process stand-in for the server that keeps every batch's metadata and acknowledges it."""
    pa = pytest.importorskip("pyarrow")
    pflight = pytest.importorskip("pyarrow.flight")

    class Server(pflight.FlightServerBase):
        def __init__(self):
            super().__init__("grpc://127.0.0.1:0")
            self.metadata = []
            self.rows = []
            self.lock = threading.Lock()

        def do_action(self, context, action):
            return iter(())

        def do_put(self, context, descriptor, reader, writer):
            while True:
                try:
                    chunk = reader.read_chunk()
                except StopIteration:
                    return
                metadata = json.loads(chunk.app_metadata.to_pybytes())
                with self.lock:
                    self.metadata.append(metadata)
                    self.rows.append(0 if chunk.data is None else chunk.data.num_rows)
                writer.write(pa.py_buffer(json.dumps({"end_byte_offset": metadata["end_byte_offset"]}).encode()))

    return Server()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_SDK_END)
def test_a_closed_session_sends_its_end_on_the_stream(tmp_path):
    server = _metadata_server()
    try:
        model = nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = tensorquill.init(model, run_id="fr-end", every=1, sink=sink.JsonlSink(tmp_path / "fr-end.rank0.jsonl"),
                                       endpoint=f"grpc://127.0.0.1:{server.port}", heartbeat_seconds=60.0)
            _steps(model, opt, 3)
            session.close()
        batches = list(server.metadata)
    finally:
        server.shutdown()
    assert batches, "the session sent no batch"
    key, end = NAMES["end_version_key"], NAMES["end_key"]
    assert all(b.get(key) == 1 for b in batches), f"a batch without {key}: 1"
    assert end in batches[-1], f"the stream's last batch has no {end}"
    assert batches[-1][end] == {"state": "closed", "outcome": "closed"}
    assert server.rows[-1] == 0


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_SDK_EXIT)
def test_a_script_that_ends_without_close_leaves_an_end_marker(tmp_path):
    path = tmp_path / "fr-exit.rank0.jsonl"
    script = (
        "import torch, tensorquill\n"
        "model = torch.nn.Linear(4, 4)\n"
        "opt = torch.optim.SGD(model.parameters(), lr=0.1)\n"
        f"tensorquill.init(model, run_id='fr-exit', every=1, sink=tensorquill.JsonlSink({str(path)!r}))\n"
        "for _ in range(2):\n"
        "    opt.zero_grad(); model(torch.ones(2, 4)).sum().backward(); opt.step()\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "sdk" / "src"), *filter(None, [env.get("PYTHONPATH")])])
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert path.exists() and path.stat().st_size > 0
    marker = sink.read_end_marker(path)
    assert marker["state"] == "closed", f"a script that ended without close() left {marker['state']!r}"
    assert marker["outcome"] == "closed"


# --- Section 8: the first open ------------------------------------------------------------------------------------


def test_the_first_open_figure():
    (case,) = of("figure")
    for folder in case["folders"]:
        assert folder["mb"] / case["mb_per_second_per_core"] == folder["seconds_one_core"]
    text = section_text(8)
    assert f"**{case['page_seconds']} seconds**" in text
    assert f"**{case['mb_per_second_per_core']} MB of lines a second per core**" in text


# --- Targets through the server: `--records` (lanes FR-1 and FR-2) --------------------------------------------------


def _binary():
    chosen = os.environ.get("TENSORQUILL_SERVER_BIN")
    if not chosen or not Path(chosen).is_file():
        pytest.skip("needs the tensorquill-server binary (TENSORQUILL_SERVER_BIN, or one built under server/target)")
    return chosen


def _has_records_flag(binary):
    shown = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=30)
    assert "--records" in shown.stdout, "tensorquill-server has no --records"


def _listing(folder):
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir())


class _Server:
    """`tensorquill-server --records` in a child process, stopped by `stop()` (SIGTERM, then SIGKILL)."""

    def __init__(self, binary, records, store):
        self.process = subprocess.Popen(
            [binary, "--records", str(records), "--store", str(store), "--flush-seconds", "0.2",
             "--watchdog-seconds", "0.5"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def kill(self):
        self.process.kill()
        self.process.wait(timeout=30)

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.kill()


def _stored(store, run_id):
    import pyarrow.parquet as pq

    rows = []
    for part in sorted((store / "runs" / run_id).rglob("part-*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    for part in sorted((store / "runs" / run_id).rglob("scalars-*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    return rows


def _wait(condition, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.2)
    return condition()


def _as_values(row):
    """A stored row with its JSON text columns read as values, keeping key order (section 2.4 a)."""
    out = dict(row)
    for column in ("extra_json", "data_json", "checkpoint_json", "system_json", "probe_json"):
        if out.get(column) is not None:
            out[column] = json.loads(out[column], object_pairs_hook=lambda pairs: [list(p) for p in pairs])
    return out


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_FOLDER)
def test_the_server_reads_a_folder_with_records():
    _has_records_flag(_binary())


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_CONVERT)
def test_the_server_stores_each_conversion_case_from_a_folder(tmp_path):
    pytest.importorskip("pyarrow")
    binary = _binary()
    _has_records_flag(binary)
    records, store = tmp_path / "records", tmp_path / "store"
    records.mkdir()
    for case in CONVERSION:
        (records / f"{case['run_id']}.rank0.jsonl").write_bytes(line_bytes(case) + b"\n")
    server = _Server(binary, records, store)
    try:
        sent = [c for c in CONVERSION if c["row"] is not None and c["row"]["schema_version"] == 2]
        assert _wait(lambda: all(_stored(store, c["run_id"]) for c in sent)), "not every row was stored"
        for case in sent:
            (row,) = _stored(store, case["run_id"])
            assert same_row(_as_values(MAKE.encode_row(row)), _as_values(case["row"])), case["name"]
    finally:
        server.stop()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_FOLDER)
def test_the_folder_of_records_is_never_written_into(tmp_path):
    pytest.importorskip("pyarrow")
    binary = _binary()
    _has_records_flag(binary)
    records, store = tmp_path / "records", tmp_path / "store"
    records.mkdir()
    _records(records / "fr-untouched.rank0.jsonl", "fr-untouched", range(20))
    before = _listing(records)
    server = _Server(binary, records, store)
    try:
        assert _wait(lambda: len(_stored(store, "fr-untouched")) == 20), "the folder's steps were not stored"
    finally:
        server.stop()
    assert _listing(records) == before


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE_FOLDER)
def test_a_server_restarted_during_a_run_stores_each_step_once(tmp_path):
    """The step's exit check (section 10, F4), and F5's time printed, not judged."""
    pytest.importorskip("pyarrow")
    binary = _binary()
    _has_records_flag(binary)
    records, store = tmp_path / "records", tmp_path / "store"
    records.mkdir()
    path = records / "fr-restart.rank0.jsonl"
    _records(path, "fr-restart", range(30))
    server = _Server(binary, records, store)
    try:
        assert _wait(lambda: len(_stored(store, "fr-restart")) >= 10), "nothing was stored before the stop"
        server.kill()
        with path.open("a") as file:
            file.writelines(json.dumps({"schema_version": 1, "run_id": "fr-restart", "rank": 0, "step": s}) + "\n"
                            for s in range(30, 50))
        started = time.monotonic()
        server = _Server(binary, records, store)
        assert _wait(lambda: len({r["step"] for r in _stored(store, "fr-restart")}) == 50), "steps were lost"
        print(f"restart to all 50 steps stored: {time.monotonic() - started:.2f} s")
        steps = [r["step"] for r in _stored(store, "fr-restart")]
        assert sorted(steps) == list(range(50)), "a step was stored twice or lost"
    finally:
        server.stop()
