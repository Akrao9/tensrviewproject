"""Where the lanes of wave AB meet, on the tree that holds all of them.

Lane AB5 added `runs.refused_batches`, the batches the server refused whole for a bad key column,
and shows them in the listing, `compare` and the MCP answers. The `watchdog` block of the evidence
(the all-ranks view's, and one rank's built from a server report) is built in `evidence.py`, lane
AB1's file, from `runs.dropped_rows` alone, and neither lane could change the other's file.

Fail before (the trial tree without the change in `evidence.py`): neither block has
`refused_batches`, and a change in refused batches is no news to the watch loop's stamp.
"""

from __future__ import annotations

from tensorquill import evidence as ev
from tensorquill import runs


def _status(*entries: dict) -> dict:
    return {"ranks": list(entries)}


REFUSED = {"rank": 1, "refused_batches": 2, "refused_batch_keys": {"step": 2}}
DROPPED = {"rank": 0, "dropped_rows": 5, "dropped_row_keys": {"step": 2, "unsupported_schema_version": 3}}


def test_the_all_ranks_block_shows_refused_batches_apart_from_dropped_rows():
    said = ev._dropped_rows_said(_status(DROPPED, REFUSED))
    assert said["dropped_rows"]["rows"] == 5 and said["dropped_rows"]["rank_ranges"] == ev._rank_ranges([0], 8)
    refused = said["refused_batches"]
    assert refused == {"batches": 2, "by_key": {"step": 2}, "rank_ranges": ev._rank_ranges([1], 8),
                       "note": runs.REFUSED_BATCHES_NOTE}
    # A count of batches, of which the server counts one sent again once: not an upper bound,
    # and in no count of rows.
    assert "upper_bound" not in refused and "rows" not in refused


def test_a_run_with_refused_batches_alone_has_that_key_and_no_dropped_rows():
    said = ev._dropped_rows_said(_status({"rank": 0}, REFUSED))
    assert set(said) == {"refused_batches"} and said["refused_batches"]["batches"] == 2


def test_a_run_with_neither_has_a_block_as_before():
    assert ev._dropped_rows_said(_status({"rank": 0}, {"rank": 1})) == {}
    assert ev._dropped_rows_said(None) == {}
    assert ev._rank_dropped_rows_said(_status({"rank": 0}), 0) == {}


def test_one_ranks_block_carries_its_own_refused_batches_only():
    status = _status({**DROPPED, **{k: v for k, v in REFUSED.items() if k != "rank"}}, {"rank": 1, "refused_batches": 7})
    own = ev._rank_dropped_rows_said(status, 0)
    assert own["dropped_rows"]["rows"] == 5
    assert own["refused_batches"] == {"batches": 2, "by_key": {"step": 2}, "note": runs.REFUSED_BATCHES_NOTE}
    assert ev._rank_dropped_rows_said(status, 1) == {"refused_batches": {"batches": 7, "note": runs.REFUSED_BATCHES_NOTE}}


# --- The two engines on the behaviour cases -------------------------------------------------------------
#
# Lane AB1 built `STEP_AND_EVIDENCE_CONTRACT.md` in the Python detectors and lane AB2 in the server, each
# against the cases alone. On the first merged tree they parted on nine cases, all with a skip only the
# caller reported: the server counted such a row as input to `loss_scale_trouble` (so the kind read
# `checked`) and the Python reader did not (`not_configured`). Neither lane's tests compare the two.

import json
import math
from pathlib import Path

import pytest

from tensorquill import server_detect

_CASES = json.loads((Path(__file__).parent / "fixtures" / "behaviour" / "cases.json").read_text())["cases"]


def _case_records(body):
    records = body.get("records") if body["status"] == "pin" else (body.get("given") or {}).get("records")
    return records if isinstance(records, list) else None


_OVER_RECORDS = [(name, _case_records(body)) for name, body in _CASES.items() if _case_records(body) is not None]


def _numbers(value):
    if isinstance(value, dict):
        return {k: _numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_numbers(v) for v in value]
    return float(value) if value in ("nan", "inf", "-inf") else value


def _comparable(value):
    """NaN is not equal to itself: written as a word, so two reports that hold one compare equal."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, dict):
        return {k: _comparable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_comparable(v) for v in value]
    return value


def test_the_cases_over_records_are_all_here():
    assert len(_OVER_RECORDS) == 30


@pytest.mark.parametrize("name,records", _OVER_RECORDS, ids=[n for n, _ in _OVER_RECORDS])
def test_both_engines_say_the_same_of_a_behaviour_case(name, records):
    try:
        server_detect.rust_ready()
    except RuntimeError as error:
        pytest.skip(str(error))
    records = [{"schema_version": 1, "run_id": "case", "rank": 0, "attempt": 1, **r} for r in _numbers(records)]
    python = ev.build_evidence(records, detectors="python")
    rust = ev.build_evidence(records, detectors="rust")
    for key in ("incidents", "not_checked", "skipped_updates", "kind_checks", "run_state"):
        assert _comparable(python.get(key)) == _comparable(rust.get(key)), key
