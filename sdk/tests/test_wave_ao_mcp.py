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


# --- 2. A number too large for a float --------------------------------------------------------

BIG = 10**400  # `Session.log` takes it and the capture writes it; float() of it raises OverflowError

SERIES_OF = [  # kind, name, the record's field holding it
    ("scalar", "x", lambda v: {"scalars": {"x": v}}),
    ("scalar", "grad_norm", lambda v: {"grad_norm": v}),
    ("layer", "blk", lambda v: {"layers": {"blk": {"abs_max": v}}}),
    ("parameter", "w", lambda v: {"grad_norms": {"w": v}}),
]


def _with(field, values: list) -> list[dict]:
    return [{"schema_version": 1, "run_id": "r", "rank": 0, "step": step, **field(value)} for step, value in enumerate(values)]


def _call(source: mcp_server.Source, name: str, arguments: dict, request_id=1) -> bytes:
    request = {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    return mcp_server.handle_line(source, json.dumps(request).encode())


@pytest.mark.parametrize("kind, name, field", SERIES_OF, ids=[f"{k}-{n}" for k, n, _ in SERIES_OF])
def test_a_series_holding_an_integer_too_large_for_a_float_is_answered(tmp_path, kind, name, field):
    """The integer is shown as recorded, left out of min, max and median, and counted apart from
    the values that are not finite: it is a finite number that no float holds."""
    capture = _write(tmp_path / "r.jsonl", _with(field, [1.0, 2.0, BIG, 4.0]))
    answer = mcp_server.get_series(mcp_server.Source(capture), {"kind": kind, "name": name})
    column = next(iter(answer["over_range"]))
    assert answer["values"][column] == [1.0, 2.0, BIG, 4.0]
    over = answer["over_range"][column]
    assert over == {"recorded": 4, "min": 1.0, "min_step": 0, "max": 4.0, "max_step": 3, "median": 2.0,
                    "first": 1.0, "last": 4.0, "too_large_for_a_float": 1, "first_too_large_step": 2}

    reply = json.loads(_call(mcp_server.Source(capture), "get_series", {"kind": kind, "name": name}))
    assert reply["result"]["isError"] is False


def test_a_series_whose_every_value_is_too_large_for_a_float_has_no_range(tmp_path):
    capture = _write(tmp_path / "r.jsonl", _with(lambda v: {"scalars": {"x": v}}, [BIG, -BIG]))
    over = mcp_server.get_series(mcp_server.Source(capture), {"kind": "scalar", "name": "x"})["over_range"]["value"]
    assert over == {"recorded": 2, "too_large_for_a_float": 2, "first_too_large_step": 0}


def test_values_that_are_not_finite_are_still_counted_as_before(tmp_path):
    """Pinned: a NaN and an Inf are `not_finite`, as before."""
    capture = _write(tmp_path / "r.jsonl", _with(lambda v: {"scalars": {"x": v}}, [1.0, "nan", "inf", 2.0]))
    over = mcp_server.get_series(mcp_server.Source(capture), {"kind": "scalar", "name": "x"})["over_range"]["value"]
    assert over["not_finite"] == 2 and over["first_not_finite_step"] == 1 and "too_large_for_a_float" not in over


@pytest.mark.xfail(strict=True, reason="lane AO-5: evidence._finite (evidence.py:437 at 453f1e2) calls math.isfinite"
                   " on the integer; the attention series reads the probe through evidence._attention_values")
def test_an_attention_series_holding_an_integer_too_large_for_a_float_is_answered(tmp_path):
    probe = lambda v: {"probe": {"attention": {"att": {"entropy_share": [v], "max_logit": [0.5]}}}}
    capture = _write(tmp_path / "r.jsonl", _with(probe, [0.5, 0.6, BIG, 0.7]))
    answer = mcp_server.get_series(mcp_server.Source(capture), {"kind": "attention", "name": "att"})
    assert answer["steps"] == [0, 1, 2, 3]


def test_a_capture_the_detectors_cannot_read_for_a_large_integer_is_said_so(tmp_path):
    """A router's series on a capture runs the detectors; where they raise on the integer (lane
    AO-5's to fix), the answer says the detectors could not read the records, and the server does
    not report a failure of its own."""
    rows = [{**row, "grad_norm": BIG if row["step"] == 2 else 1.0} for row in _rows(range(4))]
    capture = _write(tmp_path / "r.jsonl", rows)
    reply = json.loads(_call(mcp_server.Source(capture), "get_series", {"kind": "router", "name": ROUTER}))
    text = reply["result"]["content"][0]["text"]
    assert "failed inside the server" not in text
    if reply["result"]["isError"]:  # until lane AO-5's fix lands
        assert "could not read the records" in text and "OverflowError" in text


@pytest.mark.xfail(strict=True, reason="lane AO-5: build_evidence raises OverflowError on a grad_norm of 10**400"
                   " (evidence._finite, evidence.py:437 at 453f1e2)")
def test_a_routers_series_on_a_capture_with_a_large_integer_is_answered(tmp_path):
    rows = [{**row, "grad_norm": BIG if row["step"] == 2 else 1.0} for row in _rows(range(4))]
    capture = _write(tmp_path / "r.jsonl", rows)
    assert _series(mcp_server.Source(capture))["steps"] == [0, 1, 2, 3]


@pytest.mark.xfail(strict=True, reason="lane AO-6: divergence._number (divergence.py:118 at 453f1e2) calls"
                   " float() on the integer, which first_divergence reaches through compare_runs")
def test_compare_runs_over_a_run_holding_an_integer_too_large_for_a_float_is_answered(tmp_path):
    for run_id, third in (("a", BIG), ("b", 3.0)):
        _write(tmp_path / f"{run_id}.jsonl", [{"schema_version": 1, "run_id": run_id, "rank": 0, "step": step,
                                               "scalars": {"loss": third if step == 2 else 1.0 + step}} for step in range(4)])
    reply = json.loads(_call(mcp_server.Source(tmp_path / "a.jsonl"), "compare_runs", {"runs": ["a", "b"]}))
    assert "failed inside the server" not in reply["result"]["content"][0]["text"]
