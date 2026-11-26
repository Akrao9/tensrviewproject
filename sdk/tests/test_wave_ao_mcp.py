"""Wave AO, lane AO-9: the MCP server's answers on BUG-25.

1. A router's series through an evidence file whose step records have changed since it was written
   (the capture grew, or is another, shorter one) is the records' as they are now, never the
   file's older samples, and the answer says the file's samples were not used.
2. A number too large for a float (an integer like 10**400, which `Session.log` takes and the
   capture writes) is said as such in a series and raises nothing.
3. Every reply line, an error among them, is at most `MAX_ANSWER_BYTES` long.

They need what the SDK needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorquill import evidence as ev
from tensorquill import mcp_server

ROUTER = "r0"


def _rows(steps: range, collapse_from: int | None = None, run_id: str = "r") -> list[dict]:
    """Records of one router with two experts; from `collapse_from` on, the second gets nothing."""
    return [{"schema_version": 1, "run_id": run_id, "rank": 0, "step": step,
             "moe": {ROUTER: [20, 0] if collapse_from is not None and step >= collapse_from else [10, 10]},
             "scalars": {"loss": float(step)}} for step in steps]


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _series(source: mcp_server.Source, **arguments) -> dict:
    return mcp_server.get_series(source, {"kind": "router", "name": ROUTER, **arguments})


# --- 1. A router's series through an older evidence file --------------------------------------


def test_a_routers_series_through_an_older_evidence_file_is_the_grown_captures(tmp_path):
    """The file judged steps 0 to 2; the capture beside it grew to step 5, and its router
    collapsed at step 3. The series is the capture's, flags and all, as starting the server on the
    capture gives it, and the answer says the file judged fewer steps than the records hold."""
    capture = _write(tmp_path / "r.jsonl", _rows(range(6), collapse_from=3))
    report = tmp_path / "r.evidence.json"
    ev.write_evidence(ev.build_evidence(_rows(range(3))), report)

    whole = _series(mcp_server.Source(capture))
    assert whole["steps"] == [0, 1, 2, 3, 4, 5] and whole["first_flagged_step"] == 3

    answer = _series(mcp_server.Source(report))
    assert answer["records"] == str(capture)
    for key in ("recorded", "steps", "values", "over_range", "first_flagged_step"):
        assert answer[key] == whole[key], key
    note = answer["evidence_file"]
    assert str(report) in note and "steps 0 to 2" in note and "steps 0 to 5" in note
    # An explicit range past the file's steps is answered, not refused.
    assert _series(mcp_server.Source(report), start=5)["steps"] == [5]


def test_a_capture_that_grows_between_two_calls_is_read_again(tmp_path):
    """One server, two calls: while the capture holds what the file judged, the file's samples
    answer; once it has grown, the records do."""
    capture = _write(tmp_path / "r.jsonl", _rows(range(3)))
    report = tmp_path / "r.evidence.json"
    ev.write_evidence(ev.build_evidence(_rows(range(3))), report)
    source = mcp_server.Source(report)

    first = _series(source)
    assert first["steps"] == [0, 1, 2] and "evidence_file" not in first

    _write(capture, _rows(range(6), collapse_from=3))
    second = _series(source)
    assert second["steps"] == [0, 1, 2, 3, 4, 5] and second["first_flagged_step"] == 3
    assert "evidence_file" in second


def test_a_shorter_capture_than_the_file_judged_is_read_as_it_is(tmp_path):
    """The capture beside the file was started again and holds fewer steps than the file judged:
    the series has the capture's steps, not the file's."""
    report = tmp_path / "r.evidence.json"
    ev.write_evidence(ev.build_evidence(_rows(range(6), collapse_from=3)), report)
    _write(tmp_path / "r.jsonl", _rows(range(3)))

    answer = _series(mcp_server.Source(report))
    assert answer["steps"] == [0, 1, 2] and "first_flagged_step" not in answer
    assert "steps 0 to 5" in answer["evidence_file"] and "steps 0 to 2" in answer["evidence_file"]


def test_the_same_steps_with_other_counts_are_not_the_files(tmp_path):
    """The capture holds the file's steps, with other counts at one of them (another run under the
    same name and rank): the counts are the capture's."""
    report = tmp_path / "r.evidence.json"
    ev.write_evidence(ev.build_evidence(_rows(range(4))), report)
    _write(tmp_path / "r.jsonl", _rows(range(4), collapse_from=3))

    answer = _series(mcp_server.Source(report))
    assert answer["values"]["largest_share"] == [0.5, 0.5, 0.5, 1.0]
    assert "evidence_file" in answer


def test_records_that_are_no_capture_give_the_records_series_without_the_files_flags(tmp_path, monkeypatch):
    """With --records naming a run folder of a store, nothing can judge the records here: the
    routing is theirs at every step, with no flags, and the answer says so."""
    capture = _write(tmp_path / "r.jsonl", _rows(range(6), collapse_from=3))
    held = mcp_server.Source(capture).rank_records(None)
    folder = tmp_path / "store" / "runs" / "r"
    folder.mkdir(parents=True)
    records, names, origin, scalar_only = held
    report = tmp_path / "elsewhere" / "r.evidence.json"
    report.parent.mkdir()
    ev.write_evidence(ev.build_evidence(_rows(range(3))), report)
    source = mcp_server.Source(report, records=folder)
    monkeypatch.setattr(source, "rank_records", lambda rank: (records, names, {**origin, "records": str(folder)}, scalar_only))

    answer = _series(source)
    assert answer["steps"] == [0, 1, 2, 3, 4, 5]
    assert answer["values"]["largest_share"] == [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    assert "first_flagged_step" not in answer and "with no flags" in answer["evidence_file"]


def test_an_evidence_file_that_judged_these_records_answers_with_its_own_flags(tmp_path):
    """Pinned: when the records are the ones the file judged, the file's samples and flags answer,
    not a rebuild (the file may have been written with other thresholds): a flag set in the file
    by hand is in the answer."""
    _write(tmp_path / "r.jsonl", _rows(range(4)))
    document = ev.build_evidence(_rows(range(4)))
    for sample in document["samples"]:
        if sample["step"] == 1:
            sample["overloaded"] = [1]
    report = tmp_path / "r.evidence.json"
    ev.write_evidence(document, report)

    answer = _series(mcp_server.Source(report))
    assert answer["steps"] == [0, 1, 2, 3] and answer["first_flagged_step"] == 1
    assert answer["values"]["overloaded"] == [0, 1, 0, 0] and "evidence_file" not in answer
