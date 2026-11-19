"""Lane V5 (Nov 19, 2026): the MCP server, the trigger's stall episodes and the TrainCheck
import, where the two outside reviews found a wrong answer. Each test failed on `main` at
66b52d7 and passes with the lane's fix; the row it is for is in its name or docstring.

Nothing here starts a process, opens a connection or sends anything. The tests that write
Parquet parts need pyarrow and skip without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tensorquill
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill.evidence import _progress_incidents
from tensorquill.invariants import LABEL, read_traincheck, traincheck_incident
from tensorquill.trigger import Trigger

from test_mcp_server import store  # noqa: F401  (a fixture: a run folder of two ranks, written by hand)


def _capture(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _row(step: int, *, rank: int = 0, attempt: int = 1, run: str = "r", **extra) -> dict:
    return {"schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": step,
            "grad_norm": 1.0, "grad_norms": {"w": 1.0}, "scalars": {"loss": 2.0}, **extra}


def _write_parts(root: Path, records: list[dict], name: str | None = None) -> None:
    """Records as tensorquill-server lays them out: rank=<r>/attempt=<a>/part-<first>-<last>.parquet."""
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from tensorquill.flight import records_to_batch

    groups: dict[tuple[int, int], list[dict]] = {}
    for record in records:
        groups.setdefault((record.get("rank", 0), record.get("attempt", 0)), []).append(record)
    for (rank, attempt), group in groups.items():
        directory = root / f"rank={rank}" / f"attempt={attempt}"
        directory.mkdir(parents=True, exist_ok=True)
        part = name or f"part-{group[0]['step']:08}-{group[-1]['step']:08}.parquet"
        pq.write_table(pa.Table.from_batches([records_to_batch(group)]), directory / part)


# --- E2 / CLI-07, the MCP half: progress after a resume is the run as it stands -------------------


def test_e2_mcp_progress_after_a_resume_is_the_resolved_records(tmp_path):
    """Steps 0, 1 and 10 of one attempt, then a new attempt resumed at step 1: the run as it
    stands is steps 0 and 1. main: `source.steps` said first 0, last 10, four records."""
    rows = [_row(0), _row(1), _row(10), _row(1, attempt=2)]
    capture = _capture(tmp_path / "r.jsonl", rows)
    listed = mcp_server.list_incidents(mcp_server.Source(capture), {})
    assert listed["source"]["steps"] == {"first": 0, "last": 1, "records": 2}
    # What was written before the resume is still said, apart and labelled as such.
    assert listed["source"]["steps_all_attempts"] == {"first": 0, "last": 10, "records": 4}
    # The same records resolve the same way for the detectors (`resolve_attempts`).
    assert [r["step"] for r in ev.resolve_attempts(rows)] == [0, 1]


def test_e2_a_run_with_one_attempt_says_no_more_than_before(tmp_path):
    capture = _capture(tmp_path / "r.jsonl", [_row(step) for step in range(5)])
    source = mcp_server.list_incidents(mcp_server.Source(capture), {})["source"]
    assert source["steps"] == {"first": 0, "last": 4, "records": 5} and "steps_all_attempts" not in source


# --- E8 / DET-19: the TrainCheck import -------------------------------------------------------


def _summary(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _violation(name: str, first: object = None, **extra) -> dict:
    return {"display_name": name, "relation_type": "Consistent", "first_step": first, "last_step": first,
            "occurrences": 1, **extra}


def test_det19_two_invariants_whose_names_share_their_first_200_characters_are_two(tmp_path):
    shared = "x" * LABEL
    found = read_traincheck(_summary(tmp_path / "violations_summary.json", {
        "first_violation_step": 4, "distinct_invariants_violated": 2,
        "violations": [_violation(shared + "A", 4), _violation(shared + "B", 5)],
    }))
    assert found["distinct_invariants_violated"] == 2  # main: 1
    # The labels shown are still cut to LABEL characters.
    assert all(len(v["display_name"]) <= LABEL for v in found["violations"])


def test_det19_the_first_step_is_the_earliest_genuine_one_when_the_summary_gives_none(tmp_path):
    found = read_traincheck(_summary(tmp_path / "violations_summary.json", {
        "first_violation_step": None, "distinct_invariants_violated": 2,
        "violations": [_violation("a", 3), _violation("b", 2)],
    }))
    assert found["first_violation_step"] == 2
    assert traincheck_incident(found, step=80)["step"] == 2  # main: 80, the fallback


def test_det19_the_earliest_step_of_several_summaries_and_their_entries(tmp_path):
    folder = tmp_path / "check_results"
    for sub, first, entries in (("a", 9, [_violation("a", 9)]), ("b", None, [_violation("b", 6)])):
        (folder / sub).mkdir(parents=True)
        _summary(folder / sub / "violations_summary.json",
                 {"first_violation_step": first, "distinct_invariants_violated": 1, "violations": entries})
    assert read_traincheck(folder)["first_violation_step"] == 6


@pytest.mark.parametrize("field", ["first_violation_step", "distinct_invariants_violated"])
def test_det19_a_boolean_is_not_taken_for_a_number(tmp_path, field):
    document = {"first_violation_step": 4, "distinct_invariants_violated": 1, "violations": [_violation("a", 4)]}
    document[field] = True
    with pytest.raises(ValueError, match="not a TrainCheck violations summary"):
        read_traincheck(_summary(tmp_path / "violations_summary.json", document))


def test_det19_an_entrys_step_that_is_not_a_number_is_no_step(tmp_path):
    found = read_traincheck(_summary(tmp_path / "violations_summary.json", {
        "first_violation_step": None, "distinct_invariants_violated": 2,
        "violations": [_violation("a", True), _violation("b", "7")],
    }))
    assert found["first_violation_step"] is None
    assert [v["first_step"] for v in found["violations"]] == [None, None]
    assert traincheck_incident(found, step=80)["step"] == 80  # no genuine step: the fallback, as documented


# --- DET-07: a job_stalled episode and a rank that finished past it ------------------------------


def _status(*ranks: dict) -> dict:
    return {"summary": "stalled", "ranks": list(ranks), "world_size": len(ranks)}


def _rank(n: int, state: str, step: int, severity: str | None = None) -> dict:
    return {"rank": n, "state": state, "last_step": step, "attempt": 7,
            **({"severity": severity} if severity else {})}


def _follow(opened: dict, later: dict) -> dict | None:
    trigger = Trigger(command="true", log=lambda message: None)
    trigger._episode = trigger._open_episode(opened, "d1")
    trigger._follow_episode(later)
    return trigger._episode


def test_det07_a_rank_that_steps_past_the_stall_and_finishes_ends_the_episode():
    fired = _progress_incidents(_status(_rank(0, "silent", 100, "critical"), _rank(1, "silent", 100, "critical")))[0]
    for state in ("finished", "paused"):
        later = _progress_incidents(_status(_rank(0, state, 150), _rank(1, "silent", 100, "critical")))[0]
        assert later["kind"] == "job_stalled" and later["step"] == 150
        assert _follow(fired, later) is None, state  # main: still open for "finished"


def test_det07_a_rank_that_finished_before_the_stall_does_not_hold_the_episode_open():
    """The rank that finished at 200 before the others stalled at 100: they step on to 150 and
    stall again. That is progress, as before the fix (the run's step is of the ranks listed)."""
    fired = _progress_incidents(_status(_rank(0, "finished", 200), _rank(1, "silent", 100, "critical"),
                                        _rank(2, "silent", 100, "critical")))[0]
    later = _progress_incidents(_status(_rank(0, "finished", 200), _rank(1, "silent", 150, "critical"),
                                        _rank(2, "silent", 150, "critical")))[0]
    assert _follow(fired, later) is None
    # And the same stall looked at again is the same episode.
    assert _follow(fired, fired) is not None


def test_det07_an_episode_from_a_state_file_with_no_stall_step_is_not_ended_by_it():
    fired = _progress_incidents(_status(_rank(0, "silent", 100, "critical"), _rank(1, "silent", 100, "critical")))[0]
    trigger = Trigger(command="true", log=lambda message: None)
    episode = trigger._open_episode(fired, "d1")
    episode.pop("stall_step", None)  # written before this version
    trigger._episode = episode
    trigger._follow_episode(fired)
    assert trigger._episode is not None


# --- CLI-15: get_incident's follow-up calls name the rank -----------------------------------------


def test_cli15_look_next_names_the_rank_the_incidents_numbers_are_from(store):  # noqa: F811
    source = mcp_server.Source(store)
    found = mcp_server.get_incident(source, {"id": "grad_spike@40"})
    assert found["ranks"]["numbers_are_from_rank"] == 0
    assert all(call["arguments"].get("rank") == 0 for call in found["look_next"])  # main: no rank
    # Asked for rank 1, which shows the incident: its own series.
    asked = mcp_server.get_incident(source, {"id": "grad_spike@40", "rank": 1})
    assert all(call["arguments"].get("rank") == 1 for call in asked["look_next"])
    # Rank 1 alone has the non-finite layer, and its numbers are rank 1's.
    alone = mcp_server.get_incident(source, {"id": "non_finite@55"})
    assert all(call["arguments"].get("rank") == 1 for call in alone["look_next"])


def test_cli15_a_server_started_on_a_rank_folder_answers_get_incident_for_that_rank(store):  # noqa: F811
    found = mcp_server.get_incident(mcp_server.Source(store / "rank=1"), {"id": "grad_spike@40"})
    assert "rank 0's, not rank 1's" in found["ranks"]["note"]  # main: no note, as if no rank were asked
    assert all(call["arguments"].get("rank") == 1 for call in found["look_next"])


def test_cli15_on_a_capture_of_two_ranks_look_next_reads_the_rank_asked_for(tmp_path):
    rows = []
    for step in range(60):
        rows.append(_row(step, rank=0))
        rows.append(_row(step, rank=1, grad_norms={"w": 50.0 if step == 40 else 1.0}, grad_norm=50.0 if step == 40 else 1.0))
    capture = _capture(tmp_path / "r.jsonl", rows)
    source = mcp_server.Source(capture)
    listed = mcp_server.list_incidents(source, {"rank": 1})
    spike = next(i for i in listed["incidents"] if i["kind"] == "grad_spike")
    found = mcp_server.get_incident(source, {"id": spike["id"], "rank": 1})
    call = next(c for c in found["look_next"] if c["arguments"].get("kind") == "parameter")
    assert call["arguments"]["rank"] == 1
    # Followed as given, it reads rank 1's spike, not rank 0's flat series.
    series = mcp_server.get_series(source, call["arguments"])
    assert series["rank"] == 1 and series["over_range"]["grad_norm"]["max"] == 50.0


# --- CLI-22: the previous sample of a compact file, and what the cache is keyed by ---------------


def _compact_two_windows(path: Path) -> Path:
    """A compact file written with `--compact-window 0`: samples kept at steps 10 and 50 only,
    of a run sampled at every step 0 to 99, with a MoE incident at each."""
    def sample(step: int, counts: list[int]) -> dict:
        return {"step": step, "layer": 0, "router": "r0", "routed_tokens": sum(counts), "counts": counts,
                "diversity": 0.5, "overloaded": [1], "dead": []}

    document = {
        "run_id": "m", "rank": 0, "run_state": {"state": "failing", "reasons": []},
        "incidents": [{"kind": "moe_imbalance", "step": 10, "layer": 0, "router": "r0", "overloaded": [1]},
                      {"kind": "moe_imbalance", "step": 50, "layer": 0, "router": "r0", "overloaded": [1]}],
        "compact": {
            "version": 1, "window": 0, "around": [{"kind": "moe_imbalance", "step": 10}, {"kind": "moe_imbalance", "step": 50}],
            "kept_steps": [[10, 10], [50, 50]], "steps": {"total": 100, "kept": 2, "omitted": 98},
            "samples": {"total": 100, "kept": 2, "omitted": 98}, "sampled_steps": [0, 99],
            "moe_flagged": {"samples": 2, "last_step": 50}, "moe_starved": {"samples": 0, "last_step": None},
        },
        "samples": [sample(10, [90, 10]), sample(50, [80, 20])],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_cli22_a_compact_files_previous_kept_sample_is_not_called_the_previous_sample(tmp_path):
    source = mcp_server.Source(_compact_two_windows(tmp_path / "m.evidence.compact.json"))
    sample = mcp_server.get_incident(source, {"id": "moe_imbalance@50"})["sample"]
    assert "previous_sample" not in sample  # main: step 10's sample, 40 steps and another incident back
    kept = sample["previous_kept_sample"]
    assert kept["step"] == 10 and "left out" in kept["note"]


def test_cli22_the_previous_sample_inside_one_kept_window_is_still_the_previous_sample(tmp_path):
    path = _compact_two_windows(tmp_path / "m.evidence.compact.json")
    document = json.loads(path.read_text())
    document["compact"]["kept_steps"] = [[10, 10], [49, 50]]
    document["samples"].insert(1, {**document["samples"][1], "step": 49, "counts": [70, 30]})
    path.write_text(json.dumps(document), encoding="utf-8")
    sample = mcp_server.get_incident(mcp_server.Source(path), {"id": "moe_imbalance@50"})["sample"]
    assert sample["previous_sample"]["step"] == 49 and "previous_kept_sample" not in sample


def test_cli22_evidence_is_never_kept_under_a_stamp_newer_than_the_records_it_was_built_from(tmp_path, monkeypatch):
    """A job appends a line just after the capture's stamp is taken, so the records held are the
    shorter file's. main: the evidence built from them was kept under the longer file's stamp,
    taken after, and served again once the records themselves had been read again."""
    capture = _capture(tmp_path / "r.jsonl", [_row(step) for step in range(5)])
    real = mcp_server._stamp
    stamped = []

    def stamping(path):
        found = real(path)
        stamped.append(path)
        if len(stamped) == 2:  # the job writes on, after this stamp was taken
            with capture.open("a", encoding="utf-8") as file:
                file.write(json.dumps(_row(5)) + "\n")
        return found

    monkeypatch.setattr(mcp_server, "_stamp", stamping)
    source = mcp_server.Source(capture)

    def judged() -> tuple[int, int]:  # the records the detectors judged, and the records served beside them
        opened = source.open(None)
        return opened["evidence"]["kind_checks"]["grad_spike"]["records"], opened["source"]["steps"]["records"]

    seen = [judged() for _ in range(4)]
    assert all(evidence == served for evidence, served in seen), seen  # main: (5, 6) once the records were read again
    assert seen[-1] == (6, 6)


def test_cli22_a_late_part_of_an_older_attempt_is_read_though_the_newest_part_is_unchanged(tmp_path):
    folder = tmp_path / "store" / "runs" / "r"
    # Attempt 1 wrote steps 0-19; attempt 2 resumed at 40 and is the newest part.
    _write_parts(folder, [_row(step) for step in range(20)])
    _write_parts(folder, [_row(step, attempt=2) for step in range(40, 60)])
    source = mcp_server.Source(folder)
    assert mcp_server.get_series(source, {"name": "loss"})["recorded"]["steps"] == 40
    # Attempt 1's steps 20-39 arrive late: the newest part, (2, 59), is the same.
    _write_parts(folder, [_row(step) for step in range(20, 40)])
    assert mcp_server.get_series(source, {"name": "loss"})["recorded"]["steps"] == 60  # main: 40, from the cache


# --- CLI-24: the tool text's kind count is the registry's -----------------------------------------


def test_cli24_the_all_ranks_tool_text_counts_the_servers_kinds():
    text = next(t for t in mcp_server.TOOLS if t["name"] == "get_all_ranks_status")["description"]
    assert f"each rank's own {len(ev.SERVER_KINDS)} kinds" in text  # main: 25, with 26 kinds


# --- L8: another rank, read from a store, beside one rank's evidence file -------------------------

ROUTER = "blocks.0.router"


def _moe_rows(rank: int, counts: list[int], steps: int = 3) -> list[dict]:
    return [{"schema_version": 1, "run_id": "m", "rank": rank, "attempt": 1, "step": step,
             "moe": {ROUTER: counts}, "moe_top_k": {ROUTER: 1}} for step in range(steps)]


def test_l8_another_ranks_router_is_read_from_the_store_with_no_flags(tmp_path):
    """Rank 0's evidence file, with `--records` naming the run folder: rank 1's router series is
    read from its own stored parts, and says that the file's samples and flags are not used."""
    folder = tmp_path / "store" / "runs" / "m"
    rank0, rank1 = _moe_rows(0, [100, 0, 0, 0]), _moe_rows(1, [0, 50, 25, 25])
    _write_parts(folder, rank0 + rank1)
    evidence = tensorquill.build_evidence(rank0, rank=0)
    assert any(s.get("router") == ROUTER for s in evidence["samples"])
    path = ev.write_evidence(evidence, tmp_path / "rank0.evidence.json")
    source = mcp_server.Source(path, records=folder)
    other = mcp_server.get_series(source, {"name": ROUTER, "rank": 1, "expert": 1})
    assert other["rank"] == 1 and other["values"]["expert_share"] == [0.0, 0.0, 0.0]
    assert "overloaded" not in other["values"] and "first_flagged_step" not in other
    assert "judged rank 0" in other["evidence_file"] and "with no flags" in other["evidence_file"]
    # Rank 0, the rank the file judged: its own samples and flags, with no note.
    own = mcp_server.get_series(source, {"name": ROUTER, "rank": 0, "expert": 1})
    assert own["values"]["expert_share"] == [1.0, 1.0, 1.0] and "evidence_file" not in own
    assert "overloaded" in own["values"]
