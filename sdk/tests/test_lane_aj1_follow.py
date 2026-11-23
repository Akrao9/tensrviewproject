"""Lane AJ-1, A1 (WAVE_AJ_CONTRACT.md, section 4): after a resume below an earlier attempt's last step,
the watcher's later polls stay appends, as `detect --follow` takes them. The last step read is the
newest attempt's, not the highest over every attempt. The report is what a fresh `server_report`
gives at every poll.

Fail-before: on 212205d each poll after the resume restarted the child in full ("a record of step 61,
not above the last step read (100)")."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow")

from tensorquill.server_detect import FollowedDetect, find_binary, server_report  # noqa: E402

BINARY = find_binary()
pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")


def _record(step: int, attempt: int) -> dict:
    return {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": attempt, "step": step, "ts": float(step),
            "grad_norm": 1.0, "grad_norms": {"w": 1.0}}


def _same(followed: dict, records: list[dict]) -> bool:
    return json.dumps(followed, sort_keys=True) == json.dumps(server_report(records, samples=True), sort_keys=True)


@pytest.mark.parametrize(("old", "new"), [(1, 2), (3, 2)], ids=["newer-attempt-higher-number", "newer-attempt-lower-number"])
def test_appends_after_a_resume_below_the_old_top_stay_appends(old, new):
    """The newer attempt is the one that appears later (contract 1: an attempt number is an identity,
    not an order), so attempt 2 after attempt 3 is the newest too."""
    records = [_record(s, old) for s in range(101)] + [_record(s, new) for s in range(50, 61)]
    follow = FollowedDetect(BINARY)
    try:
        first = follow.report(records)
        assert follow.last["mode"] == "full" and _same(first, records)
        for step in range(61, 106):  # through the old top (100) and past it
            records = records + [_record(step, new)]
            report = follow.report(records)
            assert follow.last["mode"] == "append", (step, follow.last)
            assert _same(report, records), step
        records = records + [_record(106, old)]  # a late row of the superseded attempt
        report = follow.report(records)
        assert follow.last["mode"] == "full" and "attempt" in follow.last["why"], follow.last
        assert _same(report, records)
    finally:
        follow.close()


def test_a_step_not_above_the_newest_attempts_last_still_restarts():
    records = [_record(s, 1) for s in range(101)] + [_record(s, 2) for s in range(50, 61)]
    follow = FollowedDetect(BINARY)
    try:
        follow.report(records)
        records = records + [_record(60, 2)]  # the newest attempt's last step again
        report = follow.report(records)
        assert follow.last["mode"] == "full" and "(60)" in follow.last["why"], follow.last
        assert _same(report, records)
    finally:
        follow.close()


def test_no_records_is_still_refused():
    follow = FollowedDetect(BINARY)
    try:
        with pytest.raises(ValueError, match="no records to judge"):
            follow.report([])
    finally:
        follow.close()
