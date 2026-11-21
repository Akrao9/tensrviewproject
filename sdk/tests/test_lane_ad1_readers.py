"""Lane AD1 (wave AD): the evidence readers, each item held by a test that fails on `eb5a71b`.

- Item 1: a status entry marked `attempt_without_steps` for the very attempt a current report
  judged was read as "started again", so the rank went unjudged (`earlier_attempt`). Both orders
  of arrival are covered, through both watchers, and the control still reads `earlier_attempt`.
- Item 2: the store path of `compare` through `combine_rank_evidence` gave evidence with no
  `watchdog` block, so the rows a server dropped and the batches it refused were not in it.
- Item 3: what each reader says of the thresholds it judged with.
- Item 5: steps newer than the last judgment (CLI-10), in `watch` and `watch_all_ranks`.
- Item 7: `run_of`, the public name of `_run_of`.

None needs `pyarrow` or the server binary: the stores are written by hand, and no part is opened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev

RUN = "ad1run"


def _report(rank: int, attempt: int, last: int, incidents: list[dict] | None = None, **extra) -> dict:
    return {
        "schema_version": 1, "source": "tensorquill-server", "run_id": RUN, "rank": rank, "attempt": attempt,
        "last_step": last, "steps": last + 1, "incidents": incidents or [], "checked": list(ev.SERVER_KINDS),
        "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, **extra,
    }


def _status(ranks: list[dict], summary: str = "active") -> dict:
    return {
        "schema_version": 1, "run_id": RUN, "summary": summary, "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": len(ranks), "missing_ranks": [], "incidents": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **r} for r in ranks],
        "pauses": {}, "recommended_silent_seconds": None,
    }


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "store" / "runs" / RUN
    run.mkdir(parents=True)
    return run


def _part(run: Path, rank: int, attempt: int, first: int, last: int) -> None:
    folder = run / f"rank={rank}" / f"attempt={attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"part-{first:08d}-{last:08d}.parquet").write_bytes(b"")


def _write_report(run: Path, report: dict) -> None:
    (run / f"rank={report['rank']}" / "incidents.json").write_text(json.dumps(report))


def _write_status(run: Path, status: dict) -> None:
    (run / "watchdog.json").write_text(json.dumps(status))


class _Steps:
    """A `trigger` for the watchers that does one thing before each look: `steps[n]()` before look n."""

    def __init__(self, *steps) -> None:
        self.steps, self.looks, self.seen = list(steps), 0, []

    def tick(self) -> None:
        if self.looks < len(self.steps):
            self.steps[self.looks]()
        self.looks += 1

    def look(self, evidence, written, rank=None) -> None:
        self.seen.append((self.looks, json.loads(json.dumps(evidence))))


def _watch_all(run: Path, out: Path, *steps) -> list[tuple[int, dict]]:
    trigger = _Steps(*steps)
    ev.watch(run, out, interval=0.0, polls=len(steps), rank="all", trigger=trigger)
    return trigger.seen


def _watch_one(run: Path, out: Path, *steps) -> list[tuple[int, dict]]:
    trigger = _Steps(*steps)
    ev.watch(run / "rank=0", out, interval=0.0, polls=len(steps), trigger=trigger)
    return trigger.seen


# --- Item 1: the same-attempt report read as an earlier attempt -----------------------------------

# The server's entry for a rank whose new attempt 2 has stored nothing yet.
WITHOUT_STEPS = {"rank": 0, "attempt": 2, "last_step": None, "attempt_without_steps": True, "state": "starting"}
OTHER = {"rank": 1, "attempt": 1, "last_step": 9}


def test_item1_the_helper_does_not_read_a_mark_for_the_judged_attempt_as_an_earlier_one():
    status = {"ranks": [WITHOUT_STEPS]}
    # fail-before: {0: "started again as attempt 2, which has stored no step yet: ..."}
    assert ev._not_current_by_status(status, {0: 2}) == {}
    # The control: a judgment of attempt 1 beside the mark for attempt 2 is an earlier attempt's.
    assert "started again as attempt 2" in ev._not_current_by_status(status, {0: 1})[0]
    # A judgment whose attempt is not known, and a mark that names no attempt, are read as before.
    assert "started again as attempt 2" in ev._not_current_by_status(status, {0: None})[0]
    unnamed = {k: v for k, v in WITHOUT_STEPS.items() if k != "attempt"}
    assert "started again" in ev._not_current_by_status({"ranks": [unnamed]}, {0: 2})[0]
    # And `run_state`, which reads the same helper.
    found = ev.run_state(incidents=[], steps_judged=True, status=_status([WITHOUT_STEPS]), judged_ranks=[0],
                         judged_attempts={0: 2})
    assert found["state"] == "healthy", found


def _store_at_attempt_2(run: Path) -> None:
    """Rank 0 stored attempt 1 (steps 0-9), then attempt 2 (steps 10-19), and its report is of attempt 2."""
    _part(run, 0, 1, 0, 9)
    _part(run, 0, 2, 10, 19)
    _write_report(run, _report(0, 2, 19))
    _part(run, 1, 1, 0, 9)
    _write_report(run, _report(1, 1, 9))


def _store_at_attempt_1(run: Path) -> None:
    _part(run, 0, 1, 0, 9)
    _write_report(run, _report(0, 1, 9))
    _part(run, 1, 1, 0, 9)
    _write_report(run, _report(1, 1, 9))


def test_item1_the_status_written_before_the_report_all_ranks(tmp_path):
    run = _run(tmp_path)
    seen = _watch_all(
        run, tmp_path / "all.json",
        # Look 1: attempt 2 has started and stored nothing; the store holds attempt 1 and its report.
        lambda: (_store_at_attempt_1(run), _write_status(run, _status([WITHOUT_STEPS, OTHER]))),
        # Look 2: attempt 2's first part and its report arrive; the status is not rewritten yet.
        lambda: _store_at_attempt_2(run),
    )
    first, second = seen[0][1], seen[-1][1]
    # The control: a report of attempt 1 beside the mark for attempt 2 stays an earlier attempt's.
    assert first["ranks_by_state"] == {"current": [1], ev.EARLIER_ATTEMPT: [0]}
    # fail-before: {"current": [1], "earlier_attempt": [0]} again
    assert second["ranks_by_state"] == {"current": [0, 1]}
    assert second["judged_ranks"] == [0, 1] and second["run_state"]["state"] == "healthy"


def test_item1_the_status_written_after_the_report_all_ranks(tmp_path):
    run = _run(tmp_path)
    seen = _watch_all(
        run, tmp_path / "all.json",
        # Look 1: attempt 2's report is there, with a status from before attempt 2 began.
        lambda: (_store_at_attempt_2(run), _write_status(run, _status([{"rank": 0, "attempt": 2, "last_step": 19},
                                                                       OTHER]))),
        # Look 2: a status snapshot taken before the first batch was stored lands after the report.
        lambda: _write_status(run, _status([WITHOUT_STEPS, OTHER])),
    )
    assert seen[0][1]["judged_ranks"] == [0, 1]
    # fail-before: rank 0 earlier_attempt, judged ranks [1]
    assert seen[-1][1]["judged_ranks"] == [0, 1] and seen[-1][1]["ranks_by_state"] == {"current": [0, 1]}
    # The control, in both orders: attempt 1's report beside the mark for attempt 2 is an earlier attempt's.
    for order in ("status first", "report first"):
        run = _run(tmp_path / order.replace(" ", "-"))
        status = lambda: _write_status(run, _status([WITHOUT_STEPS, OTHER]))
        steps = (status, lambda: _store_at_attempt_1(run)) if order == "status first" else (
            lambda: _store_at_attempt_1(run), status)
        seen = _watch_all(run, tmp_path / f"{order}.json", *steps)
        last = seen[-1][1]
        assert last["ranks_by_state"] == {"current": [1], ev.EARLIER_ATTEMPT: [0]}, order
        assert last["run_state"]["state"] == "unknown"


def test_item1_both_orders_through_the_single_rank_watcher(tmp_path):
    for order in ("status first", "report first"):
        run = _run(tmp_path / order.replace(" ", "-"))
        _store_at_attempt_1(run)  # the rank's folder is there when the watcher starts
        status = lambda: _write_status(run, _status([WITHOUT_STEPS, OTHER]))
        steps = (status, lambda: _store_at_attempt_2(run)) if order == "status first" else (
            lambda: _store_at_attempt_2(run), status)
        seen = _watch_one(run, tmp_path / f"{order}.json", *steps)
        last = seen[-1][1]
        assert last["detection"]["state"] == "current" and last["detection"]["report_attempt"] == 2
        codes = [r["code"] for r in last["run_state"]["reasons"]]
        # fail-before: unknown, with `ranks_unknown` for rank 0
        assert last["run_state"]["state"] == "healthy" and "ranks_unknown" not in codes, (order, codes)
    # The control through the same watcher: attempt 1's report beside the mark for attempt 2.
    run = _run(tmp_path / "control")
    _store_at_attempt_1(run)
    seen = _watch_one(run, tmp_path / "control.json", lambda: _write_status(run, _status([WITHOUT_STEPS, OTHER])))
    assert seen[-1][1]["run_state"]["state"] == "unknown"
    assert "ranks_unknown" in [r["code"] for r in seen[-1][1]["run_state"]["reasons"]]


def test_item1_the_combined_view(tmp_path):
    def per_rank(attempt: int) -> dict:
        return {"rank": 0, "incidents": [], "last_step": 19,
                "detection": {"source": "server", "state": "current", "report_attempt": attempt}}

    status = _status([WITHOUT_STEPS])
    # fail-before: judged [] and unknown [0]
    assert ev.combine_rank_evidence({0: per_rank(2)}, status=status, status_expected=True)["judged_ranks"] == [0]
    control = ev.combine_rank_evidence({0: per_rank(1)}, status=status, status_expected=True)
    assert control["judged_ranks"] == [] and control["unknown_ranks"] == [0]


# --- Item 2: the rows a server dropped and the batches it refused, in the combined view (L22, L28) ---

DROPPED = {"dropped_rows": 7, "dropped_row_keys": {"step": 7}, "refused_batches": 2, "refused_batch_keys": {"rank": 2}}


def _counted_status() -> dict:
    return _status([{"rank": 0, "attempt": 1, "last_step": 9, **DROPPED}, OTHER])


def _judged(rank: int) -> dict:
    return {"rank": rank, "incidents": [], "last_step": 9,
            "detection": {"source": "server", "state": "current", "report_attempt": 1}}


def test_item2_the_combined_view_carries_the_watchdog_block_with_the_counts():
    status = _counted_status()
    combined = ev.combine_rank_evidence({0: _judged(0), 1: _judged(1)}, status=status, status_expected=True)
    # fail-before: KeyError, no `watchdog` key at all
    block = combined["watchdog"]
    # The all-ranks view's own block, keys and wording, from the same status.
    assert block == {k: status[k] for k in ("summary", "written_at_ms", "world_size", "pauses",
                                            "recommended_silent_seconds") if k in status} | ev._dropped_rows_said(status)
    assert block["dropped_rows"]["rows"] == 7 and block["dropped_rows"]["upper_bound"] is True
    assert block["dropped_rows"]["rank_ranges"] == "0" and "upper bound" in block["dropped_rows"]["note"]
    assert block["refused_batches"]["batches"] == 2 and "upper_bound" not in block["refused_batches"]
    # Without such counts there is no block, as before; nor without a status (local files).
    quiet = _status([{"rank": 0, "attempt": 1, "last_step": 9}, OTHER])
    assert "watchdog" not in ev.combine_rank_evidence({0: _judged(0), 1: _judged(1)}, status=quiet, status_expected=True)
    assert "watchdog" not in ev.combine_rank_evidence({0: _judged(0), 1: _judged(1)})


def test_item2_the_store_path_of_compare_through_the_combined_view(tmp_path):
    """A rank `compare` cannot judge (its report is an earlier attempt's) goes through
    `combine_rank_evidence`, and its evidence now says what the server dropped and refused."""
    from tensorquill import compare

    run = _run(tmp_path)
    _part(run, 0, 1, 0, 9)
    _write_report(run, _report(0, 1, 9))
    _write_status(run, _status([{**WITHOUT_STEPS, **DROPPED}]))
    opened = compare._open_store({"path": str(run), "run_id": RUN, "manifest": None}, None, True)
    assert opened["read_from"]["unknown_ranks"] == [0]
    # fail-before: no `watchdog` block
    assert opened["evidence"]["watchdog"]["dropped_rows"]["rows"] == 7
    assert opened["evidence"]["watchdog"]["refused_batches"]["batches"] == 2


# --- Item 3: which thresholds a reader used ---------------------------------------------------------

LISTED = {"overload_threshold": 0.5, "drift_tolerance": 0.001, "starve_window": 20, "starve_ratio": 0.5,
          "top_k": None, "update_window": 30}


def _report_look(tmp_path: Path, thresholds: dict) -> tuple[Path, dict]:
    run = _run(tmp_path)
    _part(run, 0, 1, 0, 9)
    report = _report(0, 1, 9)
    report["thresholds"] = {**thresholds, "spike_sustain": ev.SPIKE_SUSTAIN}
    _write_report(run, report)
    look = ev.read_server_report(run / "rank=0")
    assert look["state"] == "current"
    return run, look


def test_item3_a_report_reader_gives_one_value_for_each_threshold_and_says_which_it_assumed(tmp_path):
    # A report whose starve_window is not a window the detectors take, and that lists no top_k
    # (a server older than the key): the reader cannot know what was applied for either.
    odd = {k: v for k, v in LISTED.items() if k != "top_k"} | {"starve_window": "20"}
    run, look = _report_look(tmp_path / "odd", odd)
    evidence = ev._report_evidence(run / "rank=0", look, None)
    # fail-before: the top said "20", `detection.thresholds` 50
    assert evidence["starve_window"] == evidence["detection"]["thresholds"]["starve_window"] == ev.STARVE_WINDOW
    assert evidence["overload_threshold"] == evidence["detection"]["thresholds"]["overload_threshold"] == 0.5
    assert evidence["detection"]["thresholds_assumed"] == ["starve_window", "top_k"]
    assert ev.thresholds_assumed(look["report"]) == ["starve_window", "top_k"]
    # A report of this version's server lists all six, `top_k` null included: nothing is assumed.
    run, look = _report_look(tmp_path / "whole", LISTED)
    whole = ev._report_evidence(run / "rank=0", look, None)
    assert "thresholds_assumed" not in whole["detection"] and whole["detection"]["thresholds"] == LISTED
    assert ev.thresholds_assumed(look["report"]) == []


def test_item3_the_all_ranks_view_says_which_thresholds_a_rank_s_report_did_not_list(tmp_path):
    run = _run(tmp_path)
    for rank, thresholds in ((0, LISTED), (1, {"overload_threshold": 0.5})):
        _part(run, rank, 1, 0, 9)
        _write_report(run, _report(rank, 1, 9) | {"thresholds": {**thresholds, "spike_sustain": ev.SPIKE_SUSTAIN}})
    evidence = ev.all_ranks_evidence(run)
    entries = {e["rank"]: e["detection"] for e in evidence["ranks"]}
    assert "thresholds_assumed" not in entries[0]
    # fail-before: KeyError
    assert entries[1]["thresholds_assumed"] == ["drift_tolerance", "starve_window", "starve_ratio", "top_k",
                                                "update_window"]


def test_item3_the_combined_view_says_which_ranks_were_judged_with_which_thresholds():
    by_server = {"rank": 0, "incidents": [], "last_step": 9,
                 "detection": {"source": "server", "state": "current", "report_attempt": 1, "thresholds": LISTED}}
    here = {"rank": 1, "incidents": [], "last_step": 9}  # `build_evidence` at the defaults lists none
    combined = ev.combine_rank_evidence({0: by_server, 1: here})
    # fail-before: neither key, so the two ranks read as judged under one set (none said)
    assert combined["thresholds"] == LISTED
    assert combined["thresholds_by_rank"] == [
        {"ranks": [0], "rank_ranges": "0", "thresholds": LISTED},
        {"ranks": [1], "rank_ranges": "1", "thresholds": ev.THRESHOLD_DEFAULTS},
    ]
    line = ev.thresholds_line(combined)
    assert line.startswith("Thresholds that are not the defaults: the ranks were judged with on rank 0: ")
    assert "starve_window 20 (default: 50)" in line
    # Every rank under one set off the defaults: `thresholds` alone, as `build_evidence` would list it.
    one = ev.combine_rank_evidence({0: by_server, 1: {**here, "thresholds": LISTED}})
    assert one["thresholds"] == LISTED and "thresholds_by_rank" not in one
    assert "the ranks were judged with" in ev.thresholds_line(one)
    # At the defaults throughout, the combined view is what it was: neither key, and no line.
    quiet = ev.combine_rank_evidence({0: here, 1: {**here, "rank": 1}})
    assert "thresholds" not in quiet and "thresholds_by_rank" not in quiet and ev.thresholds_line(quiet) is None


# --- Item 7: `run_of`, the public name of `_run_of` ------------------------------------------------

def test_item7_run_of_is_public_and_the_private_name_still_works():
    # fail-before: AttributeError, no `run_of`
    assert ev.run_of({"run_id": "a"}) == "a" and ev.run_of({}) == "" and ev.run_of({"run_id": 7}) == ""
    assert ev._run_of is ev.run_of
    from tensorquill.evidence import _run_of  # what divergence.py imports until the merge

    assert _run_of({"run_id": "b"}) == "b"


# --- Item 5: steps newer than the last judgment (CLI-10) --------------------------------------------

THROTTLED = {"kind": "gpu_throttled", "step": 15, "reasons": ["power"]}  # an observation: it decides no state


def _awaited(run: Path):
    """Look 1: rank 0's report covers steps 0-9, the whole store. Look 2: steps 10-19 are stored and
    not reported yet, and the status gives an observation, which is news inside the wait."""
    def first():
        _store_at_attempt_1(run)
        _write_status(run, _status([{"rank": 0, "attempt": 1, "last_step": 9}, OTHER]))

    def second():
        _part(run, 0, 1, 10, 19)
        status = _status([{"rank": 0, "attempt": 1, "last_step": 19}, OTHER])
        status["incidents"] = [THROTTLED]
        _write_status(run, status)
    return first, second


def test_item5_both_watchers_show_the_last_result_with_the_steps_it_does_not_cover(tmp_path):
    one = _run(tmp_path / "one")
    _store_at_attempt_1(one)
    seen_one = _watch_one(one, tmp_path / "one.json", *_awaited(one))
    every = _run(tmp_path / "all")
    seen_all = _watch_all(every, tmp_path / "all.json", *_awaited(every))
    # Look 1 is current: no block, as today.
    assert "freshness" not in seen_one[0][1] and "freshness" not in seen_all[0][1]
    single, view = seen_one[-1][1], seen_all[-1][1]
    assert single["detection"]["state"] == "behind" and [i["kind"] for i in single["incidents"]] == ["gpu_throttled"]
    block = {"current": False, "judged_to_step": 9, "records_to_step": 19, "why": "records_newer"}
    # fail-before: no block; and the single-rank watcher said unknown where the all-ranks one said healthy
    assert single["freshness"] == block and view["freshness"] == block
    assert next(e for e in view["ranks"] if e["rank"] == 0)["freshness"] == block
    assert "freshness" not in next(e for e in view["ranks"] if e["rank"] == 1)
    assert single["run_state"]["state"] == view["run_state"]["state"] == "healthy"
    # The reason the code already names is kept, and decides nothing inside the wait.
    [awaited] = [r for r in single["run_state"]["reasons"] if r["code"] == "report_not_current"]
    assert awaited["state"] == "healthy"
    for evidence in (single, view):
        assert ev.run_state_line(evidence).endswith("; as of step 9; records reach step 19")
    # Outside the wait nothing changed: a report that is not current is unknown, as before.
    assert ev.run_state(incidents=[], steps_judged=True, status=None, status_expected=False,
                        detection={"source": "server", "state": "behind", "reason": "x"})["state"] == "unknown"


def test_item5_a_server_started_again_with_other_thresholds_carries_no_earlier_result_over(tmp_path):
    run = _run(tmp_path)

    def first():
        _store_at_attempt_1(run)
        _write_status(run, _status([{"rank": 0, "attempt": 1, "last_step": 9}, OTHER]))

    def second():
        # The server comes back with other thresholds: it reports on steps 0-14 with them, and has
        # stored 10-19 since.
        _part(run, 0, 1, 10, 19)
        _write_report(run, _report(0, 1, 14) | {"thresholds": {**LISTED, "spike_sustain": ev.SPIKE_SUSTAIN}})

    seen = _watch_all(run, tmp_path / "all.json", first, second)
    last = seen[-1][1]
    # fail-before: rank 0 kept reading as its last current report (judged [0, 1]) under the old thresholds
    assert last["judged_ranks"] == [1] and last["ranks_by_state"] == {"behind": [0], "current": [1]}
    block = {"current": False, "judged_to_step": 9, "records_to_step": 19, "why": "thresholds_changed"}
    assert last["freshness"] == block
    assert next(e for e in last["ranks"] if e["rank"] == 0)["freshness"] == block
    assert last["run_state"]["state"] == "unknown"
    assert ev.run_state_line(last).endswith(
        "; as of step 9; records reach step 19 (judged under other thresholds than those in force)")
    # The single-rank watcher's rule is the same one: the wait ends for such a report.
    look = {"state": "behind", "report": _report(0, 1, 14) | {"thresholds": {**LISTED, "spike_sustain": 3}}}
    assert ev._thresholds_moved(look, _report(0, 1, 9))
    assert not ev._thresholds_moved({**look, "report": _report(0, 1, 14)}, _report(0, 1, 9))


def test_item5_nothing_judged(tmp_path):
    run = _run(tmp_path)
    _part(run, 0, 1, 0, 9)  # a part, and no report (the server runs with --no-detect)
    evidence = ev.all_ranks_evidence(run)
    # fail-before: no block
    assert evidence["freshness"] == {"current": False, "judged_to_step": None, "records_to_step": 9,
                                     "why": "nothing_judged"}
    assert evidence["run_state"]["state"] == "unknown"
    assert ev.run_state_line(evidence).endswith("; nothing judged yet; records reach step 9")
    # A store whose every rank is current has none, so today's output is unchanged there.
    current = _run(tmp_path / "current")
    _store_at_attempt_1(current)
    assert "freshness" not in ev.all_ranks_evidence(current)
