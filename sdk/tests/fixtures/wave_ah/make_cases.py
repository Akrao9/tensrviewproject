"""Writes `cases.json`: the shared cases of wave AH's stopped-rank rule (P1-1), read by
`sdk/tests/test_wave_ah_reconnecting.py` and by `server/src/watchdog/wave_ah_tests.rs`.

The rule, the same in both engines: a rank of a status is stopped when its `state` is `silent`,
`lost` or `unresponsive`, or when its `state` is `reconnecting` and its `severity` is `warning` or
`critical`. The server gives a reconnecting rank a severity only when the same attempt was already
stopped when its stream dropped (its entry then carries `stopped_as`). Each case is a status as
the server writes it and the progress incident both engines must read from it. Written by hand,
not computed from either engine.

    python make_cases.py
"""

import json
from pathlib import Path


def rank(number, state, severity=None, last_step=40, stream=None, **more):
    stream = stream or {"reconnecting": "dropped", "lost": "dropped", "finished": "finished"}.get(state, "open")
    return {"rank": number, "phase": "train", "phase_seconds": 500.0, "heartbeat_age_seconds": 0.2,
            "severity": severity, "attempt": 7, "last_step": last_step, "work": last_step + 1,
            "seconds_since_progress": 500.0, "stream": stream, "state": state, **more}


def status(summary, *ranks, missing=()):
    return {"schema_version": 1, "run_id": "ah", "written_at_ms": 1000, "server_running": True,
            "silent_floor_seconds": 300.0, "silent_factor": 5.0, "critical_seconds": 900.0, "start_seconds": 1800.0,
            "world_size": len(ranks) + len(missing), "missing_ranks": list(missing), "summary": summary,
            "ranks": list(ranks)}


def held(number, severity, was="silent", **more):
    return rank(number, "reconnecting", severity, stopped_as=was, **more)


CASES = {
    "a_healthy_rank_that_disconnects_is_not_stopped": {
        "status": status("active", rank(0, "active", last_step=90), rank(1, "reconnecting", last_step=80)),
        "expected": [],
    },
    "a_healthy_disconnect_beside_a_summary_degraded_for_a_missing_rank_is_not_listed_as_stopped": {
        "status": status("degraded", rank(0, "active", last_step=90), rank(1, "reconnecting", last_step=80), missing=[2]),
        "expected": [{"kind": "rank_silent", "severity": "warning", "assessment": "ranks_never_connected", "step": 90,
                      "ranks": [], "other_ranks": [0, 1]}],
    },
    "a_critical_rank_that_disconnects_keeps_its_incident": {
        "status": status("degraded", rank(0, "active", last_step=90), held(1, "critical")),
        "expected": [{"kind": "rank_silent", "severity": "critical", "assessment": "suspected_hang", "step": 40,
                      "ranks": [1], "other_ranks": [0]}],
    },
    "a_warning_rank_that_disconnects_keeps_its_warning": {
        "status": status("degraded", rank(0, "active", last_step=90), held(1, "warning")),
        "expected": [{"kind": "rank_silent", "severity": "warning", "assessment": "suspected_hang", "step": 40,
                      "ranks": [1], "other_ranks": [0]}],
    },
    "an_unresponsive_rank_that_disconnects_keeps_its_incident": {
        "status": status("degraded", rank(0, "active", last_step=90), held(1, "critical", was="unresponsive")),
        "expected": [{"kind": "rank_silent", "severity": "critical", "assessment": "suspected_hang", "step": 40,
                      "ranks": [1], "other_ranks": [0]}],
    },
    "the_only_unfinished_rank_disconnects_while_critical_and_the_job_is_still_stalled": {
        "status": status("stalled", rank(0, "finished", last_step=976), held(1, "critical", last_step=49)),
        "expected": [{"kind": "job_stalled", "severity": "critical", "assessment": "suspected_hang", "step": 976,
                      "ranks": [1]}],
    },
    "after_the_grace_period_the_rank_is_lost_and_still_critical": {
        "status": status("degraded", rank(0, "active", last_step=90), rank(1, "lost", "critical", stopped_as="silent")),
        "expected": [{"kind": "rank_silent", "severity": "critical", "assessment": "lost_connection", "step": 40,
                      "ranks": [1], "other_ranks": [0], "lost_ranks": [1]}],
    },
    "a_held_rank_and_a_healthy_disconnect_side_by_side": {
        "status": status("degraded", rank(0, "active", last_step=90), held(1, "critical"),
                         rank(2, "reconnecting", last_step=88)),
        "expected": [{"kind": "rank_silent", "severity": "critical", "assessment": "suspected_hang", "step": 40,
                      "ranks": [1], "other_ranks": [0, 2]}],
    },
    "a_held_rank_beside_a_lost_one_is_not_all_lost": {
        "status": status("degraded", rank(0, "active", last_step=90), held(1, "critical"),
                         rank(2, "lost", "warning", last_step=30)),
        "expected": [{"kind": "rank_silent", "severity": "critical", "assessment": "suspected_hang", "step": 30,
                      "ranks": [1, 2], "other_ranks": [0], "lost_ranks": [2]}],
    },
}

if __name__ == "__main__":
    document = {
        "about": "Wave AH, P1-1: which ranks of a server status are stopped, and the progress incident read from it."
                 " `expected` lists each incident's kind, severity, assessment and step, the rank numbers under"
                 " `ranks` and `other_ranks` (absent for job_stalled), and `lost_ranks` when the incident has it.",
        "cases": CASES,
    }
    Path(__file__).with_name("cases.json").write_text(json.dumps(document, indent=1) + "\n")
