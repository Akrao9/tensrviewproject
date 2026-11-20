"""Lane Z2 of the SDK-22 wave: the Python reader of replica observations (`evidence.py`).

`REPLICA_OBSERVATION_CONTRACT.md` says what each rank records under `replica` and how a reader
compares the ranks' records; `fixtures/replica/` holds its 39 cases, each with the object both
engines must give. `evidence.compare_replicas` is the Python engine. It is written from the
contract and does not use the reference reading in `test_replica_contract.py`, which this file
does not import: three readings (the reference, this one, the server's) that agree on every case.

Fail before (on `main` at 633c9da, run with the main checkout's `tensorquill`): every test here
fails there, the case tests with `ImportError` on `compare_replicas`, which did not exist; the
notes on the wiring tests say what each read on `main` where the import alone is not the point.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from tensorquill import evidence as ev

CASES = Path(__file__).parent / "fixtures" / "replica"
FILES = ("no_incident_cases.json", "incident_cases.json")


def _load() -> list[tuple[str, str, dict, float]]:
    out = []
    for name in FILES:
        document = json.loads((CASES / name).read_text())
        out += [(name, case, body, document["tolerance"]) for case, body in document["cases"].items()]
    return out


ALL = _load()
IDS = [f"{f.split('_')[0]}:{c}" for f, c, _, _ in ALL]


def _case(name: str) -> tuple[dict, float]:
    [(body, tolerance)] = [(b, t) for _, c, b, t in ALL if c == name]
    return json.loads(json.dumps(body)), tolerance


# --- 1. The reader against the contract's cases -------------------------------------------------


def test_there_are_39_cases():
    assert len(ALL) == 39 and len({c for _, c, _, _ in ALL}) == 39


@pytest.mark.parametrize("file,case,body,tolerance", ALL, ids=IDS)
def test_each_case_gives_the_object_the_fixtures_spell(file, case, body, tolerance):
    assert ev.compare_replicas(body["records"], tolerance=tolerance) == body["expected"]


def _by_rank(records: list[dict], reverse: bool = False) -> list[dict]:
    """The records as several rank files give them, file by file: each rank's records in their
    own order, the ranks one after another."""
    ranks = sorted({r["rank"] for r in records}, reverse=reverse)
    return [r for rank in ranks for r in records if r["rank"] == rank]


@pytest.mark.parametrize("file,case,body,tolerance", ALL, ids=IDS)
def test_each_case_reads_the_same_fed_file_by_file(file, case, body, tolerance):
    """Only each rank's own order is contract 1's order; how the rank files follow each other
    does not change the object."""
    for order in (_by_rank(body["records"]), _by_rank(body["records"], reverse=True)):
        assert ev.compare_replicas(order, tolerance=tolerance) == body["expected"], case
    # Any interleaving that keeps each rank's own order.
    rng = random.Random(case)
    queues = {rank: [r for r in body["records"] if r["rank"] == rank] for rank in {r["rank"] for r in body["records"]}}
    mixed = []
    while any(queues.values()):
        mixed.append(queues[rng.choice([rank for rank, left in queues.items() if left])].pop(0))
    assert ev.compare_replicas(mixed, tolerance=tolerance) == body["expected"], case


def test_the_stale_launch_case_fed_file_by_file():
    """The case whose result depends on the order of records: rank 1 has two attempts under one
    launch. File by file (rank 0's file, then rank 1's), as the interleaved arrival order."""
    body, tolerance = _case("launch_read_from_an_earlier_start")
    old0, old1, new0, new1 = body["records"]
    for order in ([old0, new0, old1, new1], [old1, new1, old0, new0], [new0, old0, old1, new1]):
        assert ev.compare_replicas(order, tolerance=tolerance) == body["expected"]
    # A rank's own order is what decides: rank 1's lines the other way round make the new
    # attempt the first, and then the old one is the row set aside.
    swapped = ev.compare_replicas([old0, new0, new1, old1], tolerance=tolerance)
    assert swapped != body["expected"]
    assert [s.get("why") for s in swapped["steps"] if s["coverage"] == "not_established"] == ["attempt"]


def test_a_rank_in_several_files_takes_its_attempt_order():
    """A rank whose attempts are in two local files: `attempt_order` (`attempt_order_across` of
    its files, contract 1) orders them, whatever order the files are given in."""
    body, tolerance = _case("launch_read_from_an_earlier_start")
    old0, old1, new0, new1 = body["records"]
    for at, record in enumerate((old0, old1, new0, new1)):
        record["ts"] = 1000.0 + at
    # Rank 1's later file given first: without its order the new attempt would keep the launch.
    given = [old0, new0, new1, old1]
    assert ev.compare_replicas(given, tolerance=tolerance) != body["expected"]
    order = ev.attempt_order_across([[new1], [old1]])
    assert order[("r1", 1)] == [old1["attempt"], new1["attempt"]]
    assert ev.compare_replicas(given, tolerance=tolerance, attempt_order=order) == body["expected"]


def test_records_of_two_runs_are_refused():
    body, tolerance = _case("healthy_peers_with_distinct_attempts")
    body["records"][1]["run_id"] = "another"
    with pytest.raises(ValueError, match="more than one run"):
        ev.compare_replicas(body["records"], tolerance=tolerance)


def test_old_records_keep_todays_verdict_in_the_legacy_list():
    """Contract 6, by the rule the reader uses today (`detect_replica_drift`), on every record of
    every case that holds a gathered spread and no observation."""
    seen = 0
    for _, case, body, tolerance in ALL:
        for record, entry in zip([r for r in body["records"] if "replica_spread" in r and "replica" not in r],
                                 ev.compare_replicas(body["records"], tolerance=tolerance).get("legacy", ())):
            today = ev.detect_replica_drift([record], tolerance=tolerance)
            assert entry["drift"] is (today is not None), case
            assert entry.get("parameters", []) == sorted((today or {}).get("parameters", ())), case
            seen += 1
    assert seen == 2


# --- 2. The reader in the evidence ---------------------------------------------------------------


def _one_rank(records: list[dict], rank: int, tolerance: float) -> dict:
    """One rank's evidence, with its `last_step` as `compare` adds it."""
    own = [r for r in records if r["rank"] == rank]
    return {**ev.build_evidence(own, drift_tolerance=tolerance), "last_step": max(r["step"] for r in own)}


def _local_run(records: list[dict], tolerance: float) -> dict:
    """Every rank of a run kept as local files, put together as `compare` puts them: each rank's
    evidence, and every rank's records for the cross-rank comparison."""
    per_rank = {rank: _one_rank(records, rank, tolerance) for rank in sorted({r["rank"] for r in records})}
    return ev.combine_rank_evidence(per_rank, replica_records=records, drift_tolerance=tolerance)


def _replica_incidents(evidence: dict, key: str = "incidents") -> list[dict]:
    return [i for i in evidence.get(key) or () if i.get("kind") == "replica_drift"]


def test_a_lone_rank_file_of_a_larger_group_does_not_read_as_checked():
    """Fail before: on `main` the kind read `not_configured` with nothing on why, and the evidence
    had no word of the observations (no `replica`, no `coverage`)."""
    body, tolerance = _case("lone_rank_file")
    evidence = ev.build_evidence(body["records"], drift_tolerance=tolerance)
    check = evidence["kind_checks"]["replica_drift"]
    assert check["value"] == "not_configured" and check["enough_history"] is False
    assert check["coverage"]["missing_ranks"] == [1, 2] and check["coverage"]["compared"] == 0
    assert check["coverage"]["not_compared"][0]["coverage"] == "insufficient"
    assert "no observation of ranks 1-2 is in the records read" in check["coverage"]["detail"]
    assert evidence["replica"] == {"v": 1, "tolerance": tolerance, **body["expected"]}
    assert _replica_incidents(evidence) == []
    assert evidence["run_state"]["state"] == "healthy"  # one rank's evidence: a cross-rank kind is not degraded


def test_drift_between_ranks_is_raised_in_the_all_ranks_view_of_local_files():
    """Fail before: `combine_rank_evidence` took no records (TypeError on `replica_records`), and
    each rank's evidence alone raised nothing."""
    body, tolerance = _case("drift_between_two_ranks")
    combined = _local_run(body["records"], tolerance)
    [found] = _replica_incidents(combined, "run_incidents")
    [wanted] = body["expected"]["replica_drift"]
    assert found["class"] == "invariant" and found["source"] == "replica" and found["step"] == wanted["step"]
    assert (found["launch"], found["group"]) == (wanted["launch"], wanted["group"])
    assert sorted(found["parameters"]) == wanted["parameters"] and found["tolerance"] == tolerance
    assert all(set(detail) <= {"norm_spread", "sum_spread", "projection_spread", "non_finite"}
               for detail in found["parameters"].values())
    assert combined["replica"]["replica_drift"] == body["expected"]["replica_drift"]
    assert combined["kind_checks"]["replica_drift"]["value"] == "checked"
    assert combined["run_state"]["state"] == "unhealthy"
    # D1 is held: nothing names which rank drifted.
    assert "rank" not in found and "ranks" not in found
    text = ev._describe(found)
    assert text.startswith(f"Replicas diverged at step {wanted['step']}") and "observations of ranks 0-1" in text


def test_a_not_finite_fingerprint_is_said_in_the_incident():
    body, tolerance = _case("not_finite_on_one_rank")
    [found] = _replica_incidents(_local_run(body["records"], tolerance), "run_incidents")
    assert any(detail.get("non_finite") for detail in found["parameters"].values())
    assert "NaN or Inf" in ev._describe(found)


@pytest.mark.parametrize("case", ["drift_with_a_rank_missing", "rank_that_never_reports", "launch_not_known",
                                  "groups_differ", "every_parameter_unread", "different_device_types"])
def test_a_step_that_compared_nothing_does_not_read_as_checked(case):
    """Insufficient, not established, or complete with nothing compared: no incident, the kind
    is `unavailable` in the all-ranks view with why, and the run is degraded, not healthy.
    Fail before: TypeError on `replica_records`; with each rank alone the kind read
    `not_configured` and the run `healthy`."""
    body, tolerance = _case(case)
    combined = _local_run(body["records"], tolerance)
    check = combined["kind_checks"]["replica_drift"]
    assert _replica_incidents(combined, "run_incidents") == []
    assert check["value"] == "unavailable" and check["check"] == "replica_coverage", case
    assert check["coverage"]["compared"] == 0 and len(check["coverage"]["not_compared"]) == len(body["expected"]["steps"])
    missing = sorted({r for s in body["expected"]["steps"] for r in s.get("missing", ())})
    assert check["coverage"].get("missing_ranks", []) == missing
    if missing:
        assert f"rank{'s' if len(missing) > 1 else ''} {ev._rank_ranges(missing)} sent no observation" in check["reason"]
    assert combined["run_state"]["state"] == "degraded"
    assert any(r["code"] == "not_checked" for r in combined["run_state"]["reasons"])


def test_a_rank_that_never_reports_is_named_at_every_step():
    body, tolerance = _case("rank_that_never_reports")
    check = _local_run(body["records"], tolerance)["kind_checks"]["replica_drift"]
    assert [gap["missing"] for gap in check["coverage"]["not_compared"]] == [[2], [2]]
    line = ev.replica_line(_local_run(body["records"], tolerance))
    assert "0 of 2 step entries compared" in line and "missing rank 2" in line


def test_a_signature_only_difference_is_evidence_and_raises_nothing():
    """Fail before: TypeError on `replica_records`; nothing showed the signatures at all."""
    body, tolerance = _case("signature_only_disagreement")
    combined = _local_run(body["records"], tolerance)
    assert _replica_incidents(combined, "run_incidents") == []
    assert all(not group["kind"] == "replica_drift" for group in combined["incident_groups"])
    check = combined["kind_checks"]["replica_drift"]
    [step] = body["expected"]["steps"]
    assert check["value"] == "checked"
    assert check["signatures_differ"] == [{"step": step["step"], "launch": step["launch"], "group": step["group"],
                                           "parameters": step["signature_parameters"]}]
    assert combined["run_state"]["state"] == "healthy"
    assert "signatures differ at step" in ev.replica_line(combined) and "evidence only" in ev.replica_line(combined)


def test_old_records_keep_todays_verdict_in_the_evidence():
    """Contract 6: a record with `replica_spread` and no `replica` is read as before. A control:
    it passes on `main` too, and the evidence has no new key."""
    for case, drift in (("legacy_replica_spread_above_tolerance", True), ("legacy_replica_spread_below_tolerance", False)):
        body, tolerance = _case(case)
        evidence = ev.build_evidence(body["records"], drift_tolerance=tolerance)
        assert [i["parameters"] for i in _replica_incidents(evidence)] == (
            [ev.detect_replica_drift(body["records"], tolerance=tolerance)["parameters"]] if drift else [])
        assert "replica" not in evidence and "coverage" not in evidence["kind_checks"]["replica_drift"]
        assert evidence["kind_checks"]["replica_drift"]["value"] == "checked"


def test_two_sources_on_one_step_are_one_incident_that_names_the_other():
    """Contract 9 leaves open how a reader that shows one incident per kind presents two sources
    on one step. Here: the run's incident is the observations' (source `replica`), and names the
    gathered spread under `also`; the spread is also judged on each rank's own record, as before,
    and shows as that rank's incident group. Fail before: TypeError on `replica_records`."""
    body, tolerance = _case("explicit_spread_without_a_scope_and_both_drift")
    combined = _local_run(body["records"], tolerance)
    [found] = _replica_incidents(combined, "run_incidents")
    assert found["source"] == "replica" and found["also"] == [{"source": "replica_spread", "parameters": ["w"]}]
    [group] = [g for g in combined["incident_groups"] if g["kind"] == "replica_drift"]
    assert group["ranks"] == [0, 1] and "source" not in group["incident"]


def test_a_scoped_spread_decides_its_parameters_and_raises_once():
    body, tolerance = _case("explicit_spread_and_observations_on_one_step")
    combined = _local_run(body["records"], tolerance)
    assert _replica_incidents(combined, "run_incidents") == []  # the gathered spread decides, judged per rank
    assert [g["kind"] for g in combined["incident_groups"]] == ["replica_drift"]
    assert combined["kind_checks"]["replica_drift"]["value"] == "checked"


@pytest.mark.parametrize("file,case,body,tolerance", ALL, ids=IDS)
def test_the_all_ranks_view_of_local_files_carries_the_object(file, case, body, tolerance):
    combined = _local_run(body["records"], tolerance)
    if any("replica" in r for r in body["records"]):
        assert combined["replica"] == {"v": 1, "tolerance": tolerance, **body["expected"]}
        raised = [i for i in body["expected"]["replica_drift"] if i["source"] == "replica"]
        assert len(_replica_incidents(combined, "run_incidents")) == (1 if raised else 0), case
    else:
        assert "replica" not in combined and "replica_not_stated" not in combined


def test_records_without_observations_leave_the_evidence_as_it_was():
    """Golden files and every record set without `replica` keep their evidence (`test_golden.py`
    holds the rest): no `replica` key and the kind's entry as `kind_checks` counts it."""
    records = [{"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": s, "grad_norm": 1.0} for s in range(3)]
    evidence = ev.build_evidence(records)
    assert "replica" not in evidence
    assert evidence["kind_checks"]["replica_drift"] == {
        **ev.kind_checks(ev.input_counts(records), [])["replica_drift"]}


# --- 3. The all-ranks view of a server's store ---------------------------------------------------

RUN = "r1"


def _report(rank: int) -> dict:
    return {
        "schema_version": 1, "source": "tensorquill-server", "run_id": RUN, "rank": rank, "attempt": 1,
        "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
        "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, "kind_checks": ev.kind_checks(ev.input_counts([]), []),
    }


def _status(ranks: list[int], replica: object = None) -> dict:
    status = {
        "schema_version": 1, "run_id": RUN, "summary": "active", "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": len(ranks), "missing_ranks": [],
        "ranks": [{"rank": r, "attempt": 1, "last_step": 9, "phase": "train", "state": "active", "severity": None,
                   "stream": "open"} for r in ranks],
        "pauses": {}, "recommended_silent_seconds": None, "incidents": [], "ranks_compared": ranks,
    }
    if replica is not None:
        status["replica"] = replica
    return status


def _store(tmp_path: Path, ranks: list[int], replica: object = None) -> Path:
    """A run folder in a server's store: one current report per rank and the run's status."""
    run = tmp_path / "store" / "runs" / RUN
    for rank in ranks:
        folder = run / f"rank={rank}" / "attempt=1"
        folder.mkdir(parents=True)
        (folder / "part-00000000-00000009.parquet").write_bytes(b"")
        (folder.parent / "incidents.json").write_text(json.dumps(_report(rank)))
    (run / "watchdog.json").write_text(json.dumps(_status(ranks, replica)))
    return run


@pytest.fixture
def nothing_is_judged_here(monkeypatch):
    """The all-ranks view of a store opens no part and runs no comparison of its own."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("the store's view must take the server's result, not compute one")
    monkeypatch.setattr(ev, "_read_part", refuse)
    monkeypatch.setattr(ev, "_replica_judged", refuse)


def _server_object(case: str) -> dict:
    body, tolerance = _case(case)
    return {"v": 1, "tolerance": tolerance, **body["expected"]}


def test_the_store_view_takes_the_servers_result_from_its_status(tmp_path, nothing_is_judged_here):
    """Fail before: on `main` the view did not read `replica` from the status, so drift between
    ranks found by the server was in no incident and the run read healthy."""
    replica = _server_object("drift_on_one_of_three")
    evidence = ev.all_ranks_evidence(_store(tmp_path, [0, 1, 2], replica))
    assert evidence["replica"] == replica
    [found] = _replica_incidents(evidence, "run_incidents")
    [wanted] = replica["replica_drift"]
    assert found["class"] == "invariant" and found["source"] == "replica" and found["tolerance"] == replica["tolerance"]
    assert (found["step"], sorted(found["parameters"]), found["launch"], found["group"]) == (
        wanted["step"], wanted["parameters"], wanted["launch"], wanted["group"])
    assert evidence["kind_checks"]["replica_drift"]["value"] == "checked"
    assert evidence["run_state"]["state"] == "unhealthy"
    assert "Replicas diverged at step" in ev._all_ranks_summary(evidence)


def test_a_servers_result_with_a_rank_missing_does_not_read_as_checked(tmp_path, nothing_is_judged_here):
    evidence = ev.all_ranks_evidence(_store(tmp_path, [0, 1, 2], _server_object("rank_that_never_reports")))
    check = evidence["kind_checks"]["replica_drift"]
    assert check["value"] == "unavailable" and check["coverage"]["missing_ranks"] == [2]
    assert {"check": "replica_coverage", "kinds": ["replica_drift"], "reason": check["reason"]} in evidence["not_checked"]
    assert evidence["run_state"]["state"] == "degraded"
    assert "missing rank 2" in ev._all_ranks_summary(evidence)


def test_a_status_from_an_older_server_says_it_has_no_result(tmp_path, nothing_is_judged_here):
    """Fail before: nothing said that the replica observations were not compared."""
    evidence = ev.all_ranks_evidence(_store(tmp_path, [0, 1]))
    assert evidence["replica_not_stated"]["code"] == "status_without_replica"
    assert "older tensorquill-server" in evidence["replica_not_stated"]["detail"]
    assert "replica" not in evidence and "coverage" not in evidence["kind_checks"]["replica_drift"]
    assert ev.replica_line(evidence) is None  # said by the key: a line would be on every older server's run
    unreadable = ev.all_ranks_evidence(_store(tmp_path / "b", [0, 1], {"v": 1, "steps": "?"}))
    assert unreadable["replica_not_stated"]["code"] == "status_replica_unreadable"


def test_a_store_read_through_combine_takes_the_status_too():
    """`compare` puts a store whose ranks are not all current together with
    `combine_rank_evidence(status=...)`: the server's result comes from that status."""
    replica = _server_object("drift_between_two_ranks")
    combined = ev.combine_rank_evidence({}, status=_status([0, 1], replica), status_expected=True)
    assert combined["replica"] == replica and len(_replica_incidents(combined, "run_incidents")) == 1
    assert ev.combine_rank_evidence({}, status=_status([0, 1]), status_expected=True)["replica_not_stated"]["code"] \
        == "status_without_replica"


def test_the_watcher_writes_again_when_the_servers_result_changes(tmp_path):
    """Contract 7: the verdict is visible at the next look. Fail before: the status's stamp held
    nothing of `replica`, so a new result alone was no news and nothing was written."""
    run = _store(tmp_path, [0, 1], _server_object("delayed_rank_before_it_arrives"))
    written: list[dict] = []

    def update(evidence: dict, _path: Path) -> None:
        written.append(evidence)
        if len(written) == 1:  # the late rank's observation arrives and the server judges the step
            (run / "watchdog.json").write_text(json.dumps(_status([0, 1], _server_object("delayed_rank_after_it_arrives"))))

    ev.watch_all_ranks(run, tmp_path / "all.json", interval=0.0, polls=2, on_update=update)
    assert len(written) == 2
    assert [s["coverage"] for s in written[0]["replica"]["steps"]] == ["insufficient"]
    assert [s["coverage"] for s in written[1]["replica"]["steps"]] == ["complete"]
    assert written[1]["kind_checks"]["replica_drift"]["value"] == "checked"


# --- 4. L21, second part: an evidence file beside a file of several ranks says which it judged ----

import subprocess  # noqa: E402
import sys  # noqa: E402


def _shared_file(tmp_path: Path, ranks: tuple[int, ...] = (0, 1, 2)) -> Path:
    path = tmp_path / "job.jsonl"
    path.write_text("".join(
        json.dumps({"schema_version": 1, "run_id": "r1", "rank": rank, "attempt": 1, "step": step, "ts": 1.0 + step,
                    "grad_norm": 1.0 + rank}) + "\n"
        for step in range(3) for rank in ranks))
    return path


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = {**__import__("os").environ, "PYTHONPATH": str(Path(ev.__file__).resolve().parents[1])}
    return subprocess.run([sys.executable, "-m", "tensorquill", *args], capture_output=True, text=True, env=env,
                          timeout=120)


def test_l21_the_evidence_of_one_rank_of_a_shared_file_names_the_ranks_it_held(tmp_path):
    """Fail before: the evidence said `rank: 1` and nothing of ranks 0 and 2 in the same file, so a
    reader beside it (the run listing) could not tell it was one rank's of three."""
    path = _shared_file(tmp_path)
    done = _cli(str(path), "--rank", "1", "--detectors", "python")
    assert done.returncode == 0, done.stderr
    written = json.loads(path.with_suffix(".evidence.json").read_text())
    assert written["rank"] == 1 and written["ranks_in_source"] == [0, 1, 2]
    # The library call says the same, for a file and for a list.
    assert ev.build_evidence(path, rank=2)["ranks_in_source"] == [0, 1, 2]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert ev.build_evidence(records, rank=0)["ranks_in_source"] == [0, 1, 2]


def test_l21_a_file_of_one_rank_and_a_store_get_no_new_key(tmp_path):
    """A control: evidence of a file that holds only the rank judged is what it was."""
    path = _shared_file(tmp_path, ranks=(0,))
    assert "ranks_in_source" not in ev.build_evidence(path)
    assert "ranks_in_source" not in ev.build_evidence(path, rank=0)


def test_l21_a_shared_file_without_a_rank_is_a_usage_error_not_a_traceback(tmp_path):
    """Fail before: `ValueError` from `build_evidence` came out as a traceback."""
    done = _cli(str(_shared_file(tmp_path)), "--detectors", "python")
    assert done.returncode == 2 and "Traceback" not in done.stderr
    assert "holds ranks 0-2: choose one with --rank N" in done.stderr
