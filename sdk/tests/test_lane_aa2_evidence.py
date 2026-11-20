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


def _refuse_parts(monkeypatch) -> None:
    """From here on, opening a part or comparing observations here fails the test."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("the store's all-ranks view opened a part")
    monkeypatch.setattr(ev, "_read_part", refuse)
    monkeypatch.setattr(ev, "_replica_judged", refuse)


@pytest.fixture
def no_part_is_opened(monkeypatch):
    """The store's all-ranks view reads the reports and the status, and opens no part."""
    _refuse_parts(monkeypatch)


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


# --- 2. Decision 8: two sources on one step ------------------------------------------------------
#
# The author's rule: the evidence keeps two findings, each with its own scope, and a display may
# group them while showing both sources. These tests pin what the base already does, so they pass
# there too (run on the base export: the 4 pass); no code changed for this item.

UNSCOPED = "explicit_spread_without_a_scope_and_both_drift"
SCOPED = "explicit_spread_and_observations_on_one_step"
LAUNCH = "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47"


def _local_view(records: list[dict], tolerance: float) -> dict:
    """Every rank of a run kept as local files, as `compare` puts them together."""
    per_rank = {}
    for rank in sorted({r["rank"] for r in records}):
        own = [r for r in records if r["rank"] == rank]
        per_rank[rank] = {**ev.build_evidence(own, drift_tolerance=tolerance), "last_step": max(r["step"] for r in own)}
    return ev.combine_rank_evidence(per_rank, replica_records=records, drift_tolerance=tolerance)


def _two_sources_shown(view: dict, tolerance: float) -> None:
    """The stored result holds both entries, each with its own scope; the run's incident is the
    observations' and names the gathered spread; its words name both."""
    body, _ = _case(UNSCOPED)
    assert view["replica"]["replica_drift"] == body["expected"]["replica_drift"]
    observed, gathered = view["replica"]["replica_drift"]
    assert (observed["source"], observed["launch"], observed["group"]) == ("replica", LAUNCH, [0, 1])
    assert gathered["source"] == "replica_spread" and "launch" not in gathered and "group" not in gathered
    [found] = [i for i in view["run_incidents"] if i.get("kind") == "replica_drift"]
    assert (found["source"], found["step"], found["launch"], found["group"]) == ("replica", 99, LAUNCH, [0, 1])
    assert list(found["parameters"]) == ["w"] and found["tolerance"] == tolerance
    assert found["also"] == [{"source": "replica_spread", "parameters": ["w"]}]
    text = ev._describe(found)
    assert "from the observations of ranks 0-1" in text and "a gathered spread (replica_spread) shows drift" in text
    assert text in ev._all_ranks_summary(view)
    # Each rank's own record is judged as before (contract 6), as that rank's incident.
    [group] = [g for g in view["incident_groups"] if g["kind"] == "replica_drift"]
    assert group["ranks"] == [0, 1] and group["incident"]["step"] == 99 and list(group["incident"]["parameters"]) == ["w"]


def test_two_sources_on_one_step_in_the_local_all_ranks_view():
    body, tolerance = _case(UNSCOPED)
    _two_sources_shown(_local_view(body["records"], tolerance), tolerance)


def test_two_sources_on_one_step_in_the_store_view(tmp_path, no_part_is_opened):
    body, tolerance = _case(UNSCOPED)
    replica = {"v": 1, "tolerance": tolerance, **body["expected"]}
    evidence = ev.all_ranks_evidence(_store(tmp_path, [0, 1], _status([0, 1], replica)))
    _two_sources_shown_from_the_server(evidence, replica)


def _two_sources_shown_from_the_server(evidence: dict, replica: dict) -> None:
    """As `_two_sources_shown`, for the server's result: the spreads of the observations were
    computed by the server, so the incident names the parameters without them."""
    assert evidence["replica"] == replica
    [found] = [i for i in evidence["run_incidents"] if i.get("kind") == "replica_drift"]
    assert (found["source"], found["launch"], found["group"], found["parameters"]) == ("replica", LAUNCH, [0, 1], {"w": {}})
    assert found["also"] == [{"source": "replica_spread", "parameters": ["w"]}]
    assert ev._describe(found) in ev._all_ranks_summary(evidence)
    assert "a gathered spread (replica_spread) shows drift on that step too" in ev._all_ranks_summary(evidence)


def test_two_sources_on_one_step_in_one_ranks_evidence():
    """One rank's evidence compares no other rank, so the observations' step is `insufficient`
    there and only the gathered spread, judged on the rank's own record, is listed: the second
    entry is due only where every rank's observation is read (the two views above)."""
    body, tolerance = _case(UNSCOPED)
    evidence = ev.build_evidence(body["records"], rank=0, drift_tolerance=tolerance)  # a file of both ranks
    assert evidence["replica"]["replica_drift"] == [{"step": 99, "parameters": ["w"], "source": "replica_spread"}]
    [step] = evidence["replica"]["steps"]
    assert (step["coverage"], step["missing"]) == ("insufficient", [1])
    [found] = [i for i in evidence["incidents"] if i.get("kind") == "replica_drift"]
    assert found["step"] == 99 and list(found["parameters"]) == ["w"] and "source" not in found and "also" not in found
    check = evidence["kind_checks"]["replica_drift"]
    assert check["value"] == "checked"  # the gathered spread was judged, on the rank's own record
    assert check["coverage"]["missing_ranks"] == [1] and check["coverage"]["compared"] == 0


def test_a_spread_of_the_same_scope_is_one_finding_in_every_view(tmp_path, monkeypatch):
    """The other case: a gathered spread whose scope names the step's launch and group decides
    the parameters it holds (contract 5), so there is one entry, from it, and no incident from
    the observations, in each view."""
    body, tolerance = _case(SCOPED)
    [entry] = body["expected"]["replica_drift"]
    assert (entry["source"], entry["launch"], entry["group"]) == ("replica_spread", LAUNCH, [0, 1])
    local = _local_view(body["records"], tolerance)
    one = ev.build_evidence(body["records"], rank=0, drift_tolerance=tolerance)
    replica = {"v": 1, "tolerance": tolerance, **body["expected"]}
    _refuse_parts(monkeypatch)
    store = ev.all_ranks_evidence(_store(tmp_path, [0, 1], _status([0, 1], replica)))
    for view in (local, store):
        assert view["replica"]["replica_drift"] == [entry]
        assert [i for i in view["run_incidents"] if i.get("kind") == "replica_drift"] == []
        assert view["replica"]["steps"][0]["gathered"] == ["b", "w"]
    assert [g["kind"] for g in local["incident_groups"]] == ["replica_drift"]  # each rank's spread, as before
    assert one["replica"]["replica_drift"] == [entry]


# --- 3. L20: the all-ranks view carries each rank's `head_trends` --------------------------------
#
# Fail before: on the base the view had no `head_trends_by_rank` (KeyError in each test but the
# control, which says the key is absent where no rank has trends and passes there too).

TRENDS = {"probes": 3, "read": 2, "module": "lm_head", "first_step": 0, "latest_step": 20,
          "numbers": {"norm": {"first": 4.0, "latest": 6.0, "lowest": 4.0, "highest": 6.0, "probes": 2}},
          "overflow_positions": {"probes": 1, "positions": 3, "most": 3, "first_step": 10, "latest_step": 10}}


def test_the_store_view_carries_each_judged_ranks_head_trends(tmp_path, no_part_is_opened):
    """From each current report, as one rank's evidence built from that report has them, and
    nothing from a rank whose report is not current (it is not judged)."""
    other = {**TRENDS, "probes": 1, "read": 1}
    reports = {0: _report(0, head_trends=TRENDS), 1: _report(1), 2: _report(2, head_trends=other),
               3: _report(3, head_trends=TRENDS, last_step=5)}  # behind: stops before the stored step 9
    run = _store(tmp_path, [0, 1, 2, 3], _status([0, 1, 2, 3]), reports)
    evidence = ev.all_ranks_evidence(run)
    assert evidence["judged_ranks"] == [0, 1, 2]
    assert evidence["head_trends_by_rank"] == [{"rank": 0, "head_trends": TRENDS}, {"rank": 2, "head_trends": other}]
    assert "head_trends" not in evidence  # one rank's key, which a reader takes as one rank's
    one = ev._report_evidence(run / "rank=0", ev.read_server_report(run / "rank=0"), None)
    assert one["head_trends"] == evidence["head_trends_by_rank"][0]["head_trends"]


def test_control_no_rank_with_head_trends_gives_no_key(tmp_path, no_part_is_opened):
    evidence = ev.all_ranks_evidence(_store(tmp_path, [0, 1], _status([0, 1])))
    assert "head_trends_by_rank" not in evidence and "head_trends" not in evidence


def _probed(rank: int, with_head: bool) -> list[dict]:
    rows = []
    for step in range(12):
        row = {"schema_version": 1, "run_id": RUN, "rank": rank, "attempt": 1, "step": step, "ts": float(step),
               "grad_norm": 1.0}
        if with_head and step % 4 == 0:
            row["probe"] = {"head": {"module": "lm_head", "positions": 8, "nonfinite_positions": 0, "vocab": 32,
                                     "calls": 1, "norm": 4.0 + step, "mean": 0.1, "log_z": 3.0, "gap": 0.01,
                                     "gap_relative": 0.002, **({"overflow_positions": 2} if step == 4 else {})}}
        rows.append(row)
    return rows


def test_the_local_all_ranks_view_carries_each_ranks_head_trends():
    """`combine_rank_evidence`, as `compare` builds the view of a run's local files: each rank's
    evidence has its own trends, and the view lists them by rank."""
    per_rank = {0: ev.build_evidence(_probed(0, True)), 1: ev.build_evidence(_probed(1, False)),
                2: ev.build_evidence(_probed(2, True))}
    combined = ev.combine_rank_evidence(per_rank)
    assert [e["rank"] for e in combined["head_trends_by_rank"]] == [0, 2]
    assert combined["head_trends_by_rank"][0]["head_trends"] == per_rank[0]["head_trends"]
    assert combined["head_trends_by_rank"][0]["head_trends"]["overflow_positions"]["positions"] == 2


def test_the_store_view_of_a_real_server_carries_the_head_trends(tmp_path):
    """Through the real server: its report's trends of rank 0, as the Python detectors make them
    from rank 0's records, and none for rank 1, which probed nothing."""
    twa, _ = _server()
    run = twa.Run(tmp_path, *twa.FAST)
    records = {0: [{**r, "run_id": twa.RUN} for r in _probed(0, True)],
               1: [{**r, "run_id": twa.RUN} for r in _probed(1, False)]}
    for rank, rows in records.items():
        run.send(rank, rows)
    evidence = ev.all_ranks_evidence(run.folder)
    assert evidence["judged_ranks"] == [0, 1]
    assert evidence["head_trends_by_rank"] == [{"rank": 0, "head_trends": ev.build_evidence(records[0])["head_trends"]}]
    assert run.server.terminate() == 0
