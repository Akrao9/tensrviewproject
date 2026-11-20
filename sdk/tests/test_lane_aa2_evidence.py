"""Lane AA2 of wave AA: the evidence (`evidence.py`).

1. Decision 4 of Nov 20, 2026 (L26): the words for a server status with no `replica` key. A
   server with detection on writes the key in every status (contract 8), so a status without it
   is from a server started with `--no-detect` or from one older than the comparison. The words
   name both, or `--no-detect` alone where the status's `run_state` says the server runs so.
2. Decision 8 (L26): two sources drifting on one step stay two entries in the stored result,
   each with its scope, and the incident shown names both sources. Already so on the base: the
   tests here pin it on one rank's evidence, the local all-ranks view and the store's view.
3. L20: the all-ranks view carries each judged rank's probe `head_trends`, under
   `head_trends_by_rank`, from what the view already reads.
4. L22: the all-ranks view's `watchdog` block carries the rows the server dropped for a null
   key, as an upper bound; a run that dropped none reads as before.

Fail before: each section says how its tests were run on the base (`2e2f4dc`, exported with
`git archive` and put on `PYTHONPATH` in place of this tree's `sdk/src`).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev

sys.path.insert(0, str(Path(__file__).parent))

RUN = "r1"
CASES = Path(__file__).parent / "fixtures" / "replica"


def _case(name: str) -> tuple[dict, float]:
    document = json.loads((CASES / "incident_cases.json").read_text())
    return json.loads(json.dumps(document["cases"][name])), document["tolerance"]


def _report(rank: int, **extra: object) -> dict:
    """A current server report of one rank, as `read_server_report` takes it."""
    return {
        "schema_version": 1, "source": "tensorquill-server", "run_id": RUN, "rank": rank, "attempt": 1,
        "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
        "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, "kind_checks": ev.kind_checks(ev.input_counts([]), []),
        **extra,
    }


def _status(ranks: list[int], replica: object = None, **extra: object) -> dict:
    status = {
        "schema_version": 1, "run_id": RUN, "summary": "active", "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": len(ranks), "missing_ranks": [],
        "ranks": [{"rank": r, "attempt": 1, "last_step": 9, "phase": "train", "state": "active", "severity": None,
                   "stream": "open"} for r in ranks],
        "pauses": {}, "recommended_silent_seconds": None, "incidents": [], "ranks_compared": ranks,
        **extra,
    }
    if replica is not None:
        status["replica"] = replica
    return status


def _store(tmp_path: Path, ranks: list[int], status: dict, reports: dict[int, dict] | None = None) -> Path:
    """A run folder in a server's store: one current report per rank (an empty part names its
    attempt and steps) and the run's status."""
    run = tmp_path / "store" / "runs" / RUN
    for rank in ranks:
        folder = run / f"rank={rank}" / "attempt=1"
        folder.mkdir(parents=True)
        (folder / "part-00000000-00000009.parquet").write_bytes(b"")
        (folder.parent / "incidents.json").write_text(json.dumps((reports or {}).get(rank) or _report(rank)))
    (run / "watchdog.json").write_text(json.dumps(status))
    return run


@pytest.fixture
def no_part_is_opened(monkeypatch):
    """The store's all-ranks view reads the reports and the status, and opens no part."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("the store's all-ranks view opened a part")
    monkeypatch.setattr(ev, "_read_part", refuse)
    monkeypatch.setattr(ev, "_replica_judged", refuse)


# --- 1. Decision 4: a status with no `replica` key ----------------------------------------------
#
# Fail before: on the base every test here but the control fails, 6 of 7 with the parameters (the
# detail said "an older tensorquill-server" alone, for a `--no-detect` server too).

NO_DETECT_STATE = {"state": "unknown", "reasons": [
    {"state": "unknown", "code": "no_steps", "detail": "no step has been judged yet"},
    {"state": "unknown", "code": "ranks_unknown", "ranks": [0, 1],
     "detail": "2 ranks without a current report from this server (this server runs with --no-detect): nothing is"
               " said about incidents on them"},
]}


def test_a_status_that_says_no_detect_is_not_called_an_older_server(tmp_path, no_part_is_opened):
    evidence = ev.all_ranks_evidence(_store(tmp_path, [0, 1], _status([0, 1], run_state=NO_DETECT_STATE)))
    said = evidence["replica_not_stated"]
    assert said["code"] == "status_without_replica"
    assert "runs with --no-detect" in said["detail"] and "older" not in said["detail"]
    assert "replica" not in evidence and ev.replica_line(evidence) is None
    # The same words through `combine_rank_evidence`, which `compare` uses for such a store.
    combined = ev.combine_rank_evidence({}, status=_status([0, 1], run_state=NO_DETECT_STATE), status_expected=True)
    assert combined["replica_not_stated"] == said


@pytest.mark.parametrize("run_state", [
    None,  # a server older than `run_state` in its status
    {"state": "healthy", "reasons": []},  # detection on, and older than the comparison
    {"state": "unknown", "reasons": [{"state": "unknown", "code": "ranks_unknown", "ranks": [1],
                                      "detail": "1 rank without a current report from this server: nothing is said"
                                                " about incidents on them"}]},
    {"state": "unknown", "reasons": "?"},  # not a list: read as saying nothing
])
def test_a_status_that_does_not_say_names_both(tmp_path, no_part_is_opened, run_state):
    status = _status([0, 1], **({"run_state": run_state} if run_state is not None else {}))
    said = ev.all_ranks_evidence(_store(tmp_path, [0, 1], status))["replica_not_stated"]
    assert said["code"] == "status_without_replica"
    assert "a server started with --no-detect, or an older tensorquill-server" in said["detail"]


def test_the_words_are_true_for_a_real_server_started_with_no_detect(tmp_path):
    """Through the real server: its status has no `replica` and says `--no-detect` in its
    `run_state`, and the view says so. Fail before: the detail called it an older server."""
    twa, quick = _server()
    from tensorquill.flight import send_file

    run = twa.Run(tmp_path, *twa.FAST, *quick, "--no-detect")
    run.file(0).write_text("".join(json.dumps({**r, "rank": 0}) + "\n" for r in twa.steady(20)))
    assert send_file(run.file(0), run.server.endpoint, twa.RUN, 0, timeout=30.0)
    status = _wait_for_status(run, lambda s: (s.get("run_state") or {}).get("state") == "unknown")
    assert "replica" not in status
    said = ev.all_ranks_evidence(run.folder)["replica_not_stated"]
    assert "runs with --no-detect" in said["detail"] and "older" not in said["detail"]
    assert run.server.terminate() == 0


def test_control_a_real_server_with_detection_on_writes_the_key(tmp_path):
    """What the words rest on: with detection on, the server writes `replica` in its status, an
    empty result for a run without observations, so the view states no reason. A control: it
    passes on the base too."""
    twa, quick = _server()
    run = twa.Run(tmp_path, *twa.FAST, *quick)
    run.send(0, twa.steady(20))
    status = _wait_for_status(run, lambda s: "replica" in s)
    assert status["replica"]["steps"] == [] and status["replica"]["replica_drift"] == []
    evidence = ev.all_ranks_evidence(run.folder)
    assert "replica_not_stated" not in evidence
    assert run.server.terminate() == 0


def _server():
    pytest.importorskip("pyarrow.flight")
    from test_flight_e2e import BINARY

    if BINARY is None:
        pytest.skip("tensorquill-server is not built")
    import test_watch_all_ranks as twa
    from test_server_reports import QUICK_STATUS

    return twa, QUICK_STATUS


def _wait_for_status(run, done) -> dict:
    deadline = time.monotonic() + 10
    while True:
        try:
            status = json.loads((run.folder / "watchdog.json").read_text())
            if done(status):
                return status
        except (FileNotFoundError, ValueError):
            pass
        assert time.monotonic() < deadline, "watchdog.json never said what was waited for"
        time.sleep(0.05)
