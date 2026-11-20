"""A status entry written before a rank's first batch says nothing about which attempt was judged.

Found in CI on Nov 20, 2026, on two pushes in a row (`e158ed8`, `73eac65`): the first rank to
send was left out of the all-ranks view, `judged_ranks` `[1, 2]` where `[0, 1, 2]` was expected.
The server gives a rank whose stream is open and that has stored nothing attempt 0 and
`last_step` null, and rewrites `watchdog.json` only every `--watchdog-seconds` (10 by default).
A file left at that entry stood beside the rank's report, and `_not_current_by_status` read
"attempt 1 was judged, and the server's status shows attempt 0": the rank became
`earlier_attempt`. Reproduced with the real server at its default interval by starting the first
send about 10 s after the server; here the same file is put in place by hand.

The rule keeps every case it had: another attempt that has stored steps, a new attempt marked
`attempt_without_steps`, and a claim of another attempt that has stored nothing.
"""

from __future__ import annotations

import json
import time

import pytest

from tensorquill import evidence as ev

BEFORE_FIRST_BATCH = {"rank": 0, "attempt": 0, "last_step": None, "state": "starting", "stream": "open"}


def test_an_entry_written_before_the_first_batch_is_not_compared():
    # fail-before: {0: "attempt 1 was judged, and the server's status shows attempt 0"}
    assert ev._not_current_by_status({"ranks": [BEFORE_FIRST_BATCH]}, {0: 1}) == {}
    assert ev._not_current_by_status({"ranks": [BEFORE_FIRST_BATCH]}, {0: 1754000000000}) == {}


def test_the_cases_the_rule_had_are_kept():
    def found(entry: dict, judged: int | None = 1) -> dict:
        return ev._not_current_by_status({"ranks": [{"rank": 0, **entry}]}, {0: judged})

    # another attempt that has stored steps
    assert "shows attempt 2" in found({"attempt": 2, "last_step": 5})[0]
    # the job started again, and the new attempt has stored nothing (the server's own flag)
    assert "started again as attempt 2" in found({"attempt": 2, "last_step": None, "attempt_without_steps": True})[0]
    assert "started again as attempt 0" in found({"attempt": 0, "last_step": None, "attempt_without_steps": True})[0]
    # a claim of another attempt that has stored nothing, on a server that did not know the rank
    assert "shows attempt 7" in found({"attempt": 7, "last_step": None})[0]
    # attempt 0 is also the attempt of rows that name none: with steps stored it is compared
    assert "shows attempt 0" in found({"attempt": 0, "last_step": 9})[0]
    assert found({"attempt": 0, "last_step": 9}, judged=0) == {}
    # an entry with no `last_step` key at all (an older server) is compared as before
    assert "shows attempt 0" in found({"attempt": 0})[0]
    # the same attempt, and an attempt not known on either side
    assert found({"attempt": 1, "last_step": 59}) == {}
    assert found({"attempt": 2, "last_step": 5}, judged=None) == {}
    assert found({"last_step": 5}) == {}


def test_the_all_ranks_view_judges_a_rank_whose_status_entry_is_from_before_its_first_batch(tmp_path):
    """The CI test's own scenario, with `watchdog.json` as the server left it when its 10 s tick
    fell while rank 0 was sending: one entry, rank 0, attempt 0, no step."""
    pytest.importorskip("pyarrow.flight")
    import test_watch_all_ranks as base
    from test_flight_e2e import BINARY

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    run = base.Run(tmp_path, *base.FAST, "--watchdog-seconds", "0.05")
    try:
        for rank in range(4):
            run.send(rank, base.steady(60, nan_at=(30, "blk2" if rank < 3 else "blk3")))
        status_file = run.folder / "watchdog.json"
        deadline = time.monotonic() + 20
        while (status := _whole(status_file, ranks=4)) is None:
            assert time.monotonic() < deadline, "the server did not write a status of all four ranks"
            time.sleep(0.02)
        # No server from here on, so the file stays as this test writes it.
        run.server.process.kill()
        run.server.process.wait(20)
        [first] = [entry for entry in status["ranks"] if entry["rank"] == 0]
        early = {**first, **BEFORE_FIRST_BATCH, "seconds_since_progress": 0.0}
        status_file.write_text(json.dumps({**status, "ranks": [early]}))
        evidence, _ = base.look(run, tmp_path / "out")
    finally:
        for server in base.STARTED:
            if server.process.poll() is None:
                server.process.kill()
                server.process.wait(20)
            if not server.process.stdout.closed:
                server.process.stdout.close()
        base.STARTED.clear()
    # fail-before: judged ranks [1, 2, 3] and the first group's ranks [1, 2], as in CI
    assert evidence["judged_ranks"] == [0, 1, 2, 3]
    shown = [(g["kind"], g["step"], g["ranks"], g["incident"]["layers"]) for g in evidence["incident_groups"]]
    assert shown == [("non_finite", 30, [0, 1, 2], ["blk2"]), ("non_finite", 30, [3], ["blk3"])]
    assert evidence["unknown_ranks"] == []


def _whole(status_file, *, ranks: int):
    try:
        status = json.loads(status_file.read_text())
    except (OSError, ValueError):
        return None
    entries = status.get("ranks") or []
    return status if len(entries) == ranks and all(e.get("last_step") == 59 for e in entries) else None
