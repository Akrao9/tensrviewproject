"""Lane AD5 (wave AD): the readers and the MCP tools.

1. What an agent with only the MCP tools could not see (L29): `skipped_updates`,
   `skipped_updates_by_rank`, `checkpoint_tensors`, `checkpoint_tensors_by_rank`, and of a kind's
   entry in `kind_checks` its `parameters_not_judged`, the replica `coverage` and the no-peer
   `note`. Each answer of `list_incidents`, `get_incident` and `get_all_ranks_status` carries them
   under the evidence's own key and shape, only where the evidence has them. The inputs are the
   contracts' own cases (`fixtures/behaviour`, `fixtures/checkpoint`, `fixtures/replica`), read
   here and not changed.
2. The compact evidence file keeps `checkpoint_tensors` (it already did at `eb5a71b`); it is held
   here through the MCP answer read from a compact file, which is what did not carry it.
3. Steps newer than the last judgment (CLI-10): a `runs` row, and the MCP `list_runs` row, carry
   `freshness` when the evidence does not cover the newest step its records reach.
4. `first_divergence(..., resolved=True)`, which `compare` passes, and whether the overlap guard is
   still needed for a caller's rows.
5. One records file that holds two run IDs: `get_series` mixed their rows into one series; it now
   refuses them, as `list_incidents` does on the same file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from tensorquill import compact
from tensorquill import compare as compare_module
from tensorquill import divergence
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill import runs as runs_module
from tensorquill.divergence import first_divergence
from tensorquill.evidence import load_records, resolve_attempts, resolve_attempts_across

FIXTURES = Path(__file__).parent / "fixtures"
BEHAVIOUR = json.loads((FIXTURES / "behaviour" / "cases.json").read_text())["cases"]
CHECKPOINT = json.loads((FIXTURES / "checkpoint" / "cases.json").read_text())["cases"]
REPLICA = json.loads((FIXTURES / "replica" / "attribution_cases.json").read_text())
L29_KEYS = ("skipped_updates", "skipped_updates_by_rank", "checkpoint_tensors", "checkpoint_tensors_by_rank",
            "kind_checks")


def behaviour_rows(name: str) -> list[dict[str, Any]]:
    """A behaviour case's records as the engines read them (as lane AB1's tests read them)."""
    out = []
    for row in BEHAVIOUR[name]["given"]["records"]:
        row = {k: (float(v) if v in ("nan", "inf", "-inf") else v) for k, v in row.items()}
        if isinstance(row.get("grad_norms"), dict):
            row["grad_norms"] = {k: (float(v) if v == "nan" else v) for k, v in row["grad_norms"].items()}
        out.append({"schema_version": 1, "run_id": "r1", "rank": 0, "attempt": 1, **row})
    return out


def checkpoint_rows(name: str) -> list[dict[str, Any]]:
    return [dict(r) for r in CHECKPOINT[name]["records"]]


def write_capture(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def call(where: Path, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """A tool's answer as an agent gets it: through `call_tool`, so the size cut and strict JSON apply."""
    reply = mcp_server.call_tool(mcp_server.Source(where), 1, {"name": name, "arguments": arguments or {}})
    assert reply["result"]["isError"] is False, reply
    return json.loads(reply["result"]["content"][0]["text"])


def refused(where: Path, name: str, arguments: dict[str, Any] | None = None) -> str:
    reply = mcp_server.call_tool(mcp_server.Source(where), 1, {"name": name, "arguments": arguments or {}})
    assert reply["result"]["isError"] is True, reply
    return reply["result"]["content"][0]["text"]


def plain_rows(run: str = "r1", steps: range = range(30), rank: int = 0, **extra: Any) -> list[dict[str, Any]]:
    return [{"schema_version": 1, "run_id": run, "rank": rank, "attempt": 1, "step": s, "ts": 1000.0 + s,
             "grad_norm": 1.0, "grad_norms": {"w": 1.0 + s / 100}, **extra} for s in steps]


# --- 1. The evidence's blocks in the MCP answers ------------------------------------------------


def test_a_nan_gradient_kept_on_a_callers_skip_is_in_the_answer(tmp_path):
    """The register's example (L29): a NaN gradient on a skip the caller reported raises no
    incident and is kept in the evidence's `skipped_updates`. The agent now sees it."""
    rows = behaviour_rows("a_gradient_not_finite_on_a_skip_the_caller_reported")
    capture = write_capture(tmp_path / "r1.rank0.jsonl", rows)
    expected = ev.build_evidence(rows)["skipped_updates"]
    assert expected["non_finite_gradients"] == {"steps": 1, "first_step": 3, "gradients": ["blk.weight"]}
    answer = call(capture, "list_incidents")
    assert answer["incidents"] == []
    assert answer["skipped_updates"] == expected  # the evidence's key, unchanged in shape
    # A capture with no such record: none of the new keys, so its answer is as before.
    plain = write_capture(tmp_path / "p1.rank0.jsonl", plain_rows("p1"))
    assert not set(L29_KEYS) & set(call(plain, "list_incidents"))


def test_a_stall_names_the_parameters_it_did_not_judge(tmp_path):
    rows = behaviour_rows("a_stall_beside_an_unread_gradient")
    capture = write_capture(tmp_path / "r1.rank0.jsonl", rows)
    answer = call(capture, "list_incidents")
    assert answer["kind_checks"] == {"stalled_weights": {"parameters_not_judged": ["emb.weight"]}}
    assert "stalled_weights" in answer["kinds"]["checked"]  # `kinds` is unchanged: names by value
    [stall] = [i for i in answer["incidents"] if i["kind"] == "stalled_weights"]
    incident = call(capture, "get_incident", {"id": stall["id"]})
    assert incident["kind_checks"] == {"stalled_weights": {"parameters_not_judged": ["emb.weight"]}}
    assert "skipped_updates" not in incident  # the evidence has none


def test_a_lone_ranks_replica_observations_carry_the_no_peer_words_and_the_coverage(tmp_path):
    body = REPLICA["cases"]["group_of_one"]
    capture = write_capture(tmp_path / "r1.rank0.jsonl", [r for r in body["records"] if r["rank"] == 0])
    entry = ev.build_evidence(load_records(capture))["kind_checks"]["replica_drift"]
    answer = call(capture, "list_incidents")
    assert answer["kind_checks"]["replica_drift"] == {"coverage": entry["coverage"], "note": ev.NO_PEER_WORDS}
    assert answer["kind_checks"]["replica_drift"]["note"] == "one rank observed, zero peers compared"
    assert "replica_drift" in answer["kinds"]["not_configured"]


def test_checkpoint_tensors_in_the_answers_and_in_a_compact_file(tmp_path):
    rows = checkpoint_rows("a_resume_with_two_tensors_verified_missing")
    capture = write_capture(tmp_path / "r1.rank0.jsonl", rows)
    evidence = ev.build_evidence(load_records(capture))
    block = evidence["checkpoint_tensors"]
    assert block["entries"][0]["missing"] == {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}
    answer = call(capture, "list_incidents")
    assert answer["checkpoint_tensors"] == block
    [resume] = [i for i in answer["incidents"] if i["kind"] == "resume_incomplete"]
    incident = call(capture, "get_incident", {"id": resume["id"]})
    assert incident["checkpoint_tensors"] == block and "skipped_updates" not in incident
    # Item 2: the compact file keeps the block (it did on the base too), and the server started on
    # the compact file, as a trigger hands it to an agent, now gives it.
    written = compact.write_compact(evidence, tmp_path / "r1.rank0.evidence.json")
    on_disk = json.loads(written.read_text())
    assert on_disk["checkpoint_tensors"] == block and isinstance(on_disk["compact"], dict)
    assert call(written, "list_incidents")["checkpoint_tensors"] == block
    # A resume recorded before the check: no block anywhere.
    before = write_capture(tmp_path / "old.rank0.jsonl", checkpoint_rows("a_resume_recorded_before_this_check"))
    assert "checkpoint_tensors" not in call(before, "list_incidents")
    old = ev.build_evidence(load_records(before))
    assert "checkpoint_tensors" not in old and "checkpoint_tensors" not in compact.compact_evidence(old)


def store(root: Path, run: str, reports: dict[int, dict[str, Any]], status: dict[str, Any] | None = None) -> Path:
    """A run folder in a store with a current report for each rank (as lane AB5's tests write one)."""
    folder = root / "runs" / run
    for rank, extra in reports.items():
        parts = folder / f"rank={rank}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-00000000-00000009.parquet").write_bytes(b"")
        (parts.parent / "incidents.json").write_text(json.dumps({
            "schema_version": 1, "source": "tensorquill-server", "run_id": run, "rank": rank, "attempt": 1,
            "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
            "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN}, **extra,
        }))
    (folder / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run, "summary": "active", "server_running": True,
        "written_at_ms": int(time.time() * 1000), "world_size": len(reports), "missing_ranks": [], "incidents": [],
        "ranks": [{"rank": r, "attempt": 1, "last_step": 9, "phase": "train", "state": "active", "severity": None,
                   "stream": "open"} for r in reports],
        "pauses": {}, "recommended_silent_seconds": None, **(status or {}),
    }))
    return folder


def test_the_all_ranks_answers_carry_each_ranks_blocks_and_the_run_level_words(tmp_path):
    skips = {"caller": 1, "scaler": 0, "both": 0,
             "non_finite_gradients": {"steps": 1, "first_step": 3, "gradients": ["blk.weight"]}}
    tensors = ev.build_evidence(checkpoint_rows("a_resume_with_two_tensors_verified_missing"))["checkpoint_tensors"]
    # Rank 0 recorded batch hashes and rank 1 did not, and the server compared no rank: the run's
    # `duplicate_batches` saw one rank and had no peer for it (contract, section 2).
    with_data = ev.kind_checks({**dict.fromkeys(ev.INPUT_NEEDS, 0), "steps": 10, "data": 10}, [])
    without = ev.kind_checks({**dict.fromkeys(ev.INPUT_NEEDS, 0), "steps": 10}, [])
    folder = store(tmp_path, "run", {0: {"skipped_updates": skips, "kind_checks": with_data},
                                     1: {"checkpoint_tensors": tensors, "kind_checks": without}},
                   {"ranks_compared": []})
    view = ev.all_ranks_evidence(folder)
    assert view["skipped_updates_by_rank"] == [{"rank": 0, "skipped_updates": skips}]
    assert view["checkpoint_tensors_by_rank"] == [{"rank": 1, "checkpoint_tensors": tensors}]
    assert view["kind_checks"]["duplicate_batches"]["note"] == ev.NO_PEER_WORDS
    status = call(folder, "get_all_ranks_status")
    assert status["skipped_updates_by_rank"] == view["skipped_updates_by_rank"]
    assert status["checkpoint_tensors_by_rank"] == view["checkpoint_tensors_by_rank"]
    # `kind_checks` only for the kinds whose entry has one of the keys: here the no-peer words alone.
    assert status["kind_checks"] == {"duplicate_batches": {"note": ev.NO_PEER_WORDS}}
    # Both ranks with data and neither compared: the kind's other note is not the no-peer words, and
    # is not carried (nor is any kind's own `note`).
    both = store(tmp_path, "both", {0: {"kind_checks": with_data}, 1: {"kind_checks": with_data}},
                 {"ranks_compared": []})
    assert ev.all_ranks_evidence(both)["kind_checks"]["duplicate_batches"]["note"] != ev.NO_PEER_WORDS
    assert not set(L29_KEYS) & set(call(both, "get_all_ranks_status"))
    # With `rank`, that rank's entries only, and no key where the rank has none.
    one = call(folder, "get_all_ranks_status", {"rank": 1})
    assert one["checkpoint_tensors_by_rank"] == view["checkpoint_tensors_by_rank"] and "skipped_updates_by_rank" not in one
    listed = call(folder, "list_incidents", {"rank": 0})
    assert listed["skipped_updates_by_rank"] == view["skipped_updates_by_rank"]
    assert "checkpoint_tensors_by_rank" not in listed
    whole = call(folder, "list_incidents")
    assert whole["checkpoint_tensors_by_rank"] == view["checkpoint_tensors_by_rank"]
    # The kept evidence is not changed by an answer.
    assert ev.all_ranks_evidence(folder)["skipped_updates_by_rank"] == view["skipped_updates_by_rank"]


def test_the_all_ranks_lists_are_bounded_as_the_dropped_rows_ranks_are(tmp_path):
    ranks = mcp_server.PER_RANK_ROWS + 6
    skips = {"caller": 2, "scaler": 0, "both": 0}
    folder = store(tmp_path, "wide", {r: {"skipped_updates": skips} for r in range(ranks)})
    status = call(folder, "get_all_ranks_status")
    assert [e["rank"] for e in status["skipped_updates_by_rank"]] == list(range(mcp_server.PER_RANK_ROWS))
    assert status["skipped_updates_by_rank_omitted"] == 6
    last = call(folder, "get_all_ranks_status", {"rank": ranks - 1})
    assert last["skipped_updates_by_rank"] == [{"rank": ranks - 1, "skipped_updates": skips}]
    assert "skipped_updates_by_rank_omitted" not in last and "truncated" not in status


def test_each_tool_says_what_each_block_means_and_what_its_absence_means():
    described = {tool["name"]: tool["description"] for tool in mcp_server.TOOLS}
    for name in ("list_incidents", "get_all_ranks_status"):
        text = described[name]
        for key in ("skipped_updates", "non_finite_gradients", "checkpoint_tensors", "skipped_updates_by_rank",
                    "checkpoint_tensors_by_rank", "parameters_not_judged", "coverage", ev.NO_PEER_WORDS):
            assert key in text, (name, key)
        assert "With no key, no record says who skipped an update" in text
        assert "With no key, no load recorded its tensors" in text
        assert "With no `kind_checks`, no kind's entry has any of these" in text
    for key in ("skipped_updates", "checkpoint_tensors", "kind_checks", "parameters_not_judged"):
        assert key in described["get_incident"], key
    assert "refused" in described["get_series"] and "more than one run ID" in described["get_series"]
    assert "freshness" in described["list_runs"] and "nothing_judged" in described["list_runs"]


# --- 3. Steps newer than the last judgment --------------------------------------------------------


def older(path: Path, than: Path) -> None:
    """Date `path` before `than`, as an evidence file written before its records grew."""
    stamp = than.stat().st_mtime - 60
    os.utime(path, (stamp, stamp))


def newer(path: Path, than: Path) -> None:
    stamp = than.stat().st_mtime + 60
    os.utime(path, (stamp, stamp))


def evidence_beside(capture: Path, evidence: dict[str, Any]) -> Path:
    return ev.write_evidence(evidence, capture.with_suffix(".evidence.json"))


def only_row(root: Path) -> dict[str, Any]:
    [row] = runs_module.list_runs(root)
    return row


def test_a_row_whose_evidence_is_older_than_its_records_says_how_far_it_judged(tmp_path):
    capture = write_capture(tmp_path / "r1.rank0.jsonl", plain_rows())
    first = ev.build_evidence(plain_rows(steps=range(10)))
    written = evidence_beside(capture, first)
    older(written, capture)
    row = only_row(tmp_path)
    assert row["evidence"]["state"] == "older_than_records" and "detail" in row["evidence"]  # kept
    assert row["freshness"] == {"current": False, "judged_to_step": None, "records_to_step": 29, "why": "records_newer"}
    # `records_newer`: the evidence's result is shown as it is, beside the block.
    assert row["run_state"] == "healthy" and row["incidents"] == {} and row["kinds_checked"] is not None
    text = runs_module.listing_text([row], tmp_path)
    assert "r1: as of a step its evidence file does not name; records reach step 29" in text
    assert call(capture, "list_runs")["runs"][0]["freshness"] == row["freshness"]
    # Written after the records: covered, no block, the row as before.
    newer(written, capture)
    row = only_row(tmp_path)
    assert row["evidence"]["state"] == "current" and "freshness" not in row
    assert "records reach" not in runs_module.listing_text([row], tmp_path)
    assert "freshness" not in call(capture, "list_runs")["runs"][0]


def test_a_named_judged_step_dates_the_judgment(tmp_path):
    capture = write_capture(tmp_path / "r1.rank0.jsonl", plain_rows())
    judged = ev.build_evidence(plain_rows(steps=range(10)))
    judged["detection"] = {"source": "server", "state": "current", "reason": "", "rank": 0, "report_last_step": 9}
    written = evidence_beside(capture, judged)
    newer(written, capture)  # the file's time says nothing here: the step it names does
    row = only_row(tmp_path)
    assert row["freshness"] == {"current": False, "judged_to_step": 9, "records_to_step": 29, "why": "records_newer"}
    assert "r1: as of step 9; records reach step 29" in runs_module.listing_text([row], tmp_path)
    # A judgment that covers the newest step: no block, though the file is older than the records.
    judged["detection"]["report_last_step"] = 29
    written = evidence_beside(capture, judged)
    older(written, capture)
    row = only_row(tmp_path)
    assert "freshness" not in row and row["evidence"]["state"] == "older_than_records"


def test_evidence_that_judged_no_step_carries_no_result_over(tmp_path):
    capture = write_capture(tmp_path / "r1.rank0.jsonl", plain_rows())
    written = evidence_beside(capture, ev.build_evidence([]))
    older(written, capture)
    row = only_row(tmp_path)
    assert row["freshness"] == {"current": False, "judged_to_step": None, "records_to_step": 29, "why": "nothing_judged"}
    assert row["run_state"] == "unknown"
    assert row["incidents"] is None and row["first_incident"] is None and row["kinds_checked"] is None
    text = runs_module.listing_text([row], tmp_path)
    assert "r1: as of no step judged; records reach step 29; its evidence judged no step" in text
    cells = text.splitlines()[2]
    assert " unknown " in cells and " ? " in cells
    mcp_row = call(capture, "list_runs")["runs"][0]
    assert mcp_row["freshness"]["why"] == "nothing_judged" and mcp_row["incidents"] is None


def test_every_local_row_shape_that_reads_older_than_records_has_the_block(tmp_path):
    # A run of two rank files, rank 1's evidence older than its records.
    two = tmp_path / "two"
    files = {r: write_capture(two / f"m2.rank{r}.jsonl", plain_rows("m2", rank=r)) for r in (0, 1)}
    for r, path in files.items():
        written = evidence_beside(path, ev.build_evidence(plain_rows("m2", rank=r, steps=range(20 if r else 30))))
        (older if r else newer)(written, path)
    row = only_row(two)
    assert row["evidence"]["state"] == "older_than_records"
    assert row["freshness"] == {"current": False, "judged_to_step": None, "records_to_step": 29, "why": "records_newer"}
    assert row["run_state"] == "healthy"
    # One file of two ranks, its evidence one rank's and older than the records.
    shared = tmp_path / "shared"
    both = write_capture(shared / "s1.rank0.jsonl", [r for s in range(30) for r in (
        plain_rows("s1", steps=range(s, s + 1))[0], plain_rows("s1", steps=range(s, s + 1), rank=1)[0])])
    written = evidence_beside(both, ev.build_evidence(plain_rows("s1", steps=range(10))))
    older(written, both)
    row = only_row(shared)
    assert row["freshness"]["why"] == "records_newer" and row["freshness"]["records_to_step"] == 29
    # Every row that reads current has none.
    for folder in (two, shared):
        for path in folder.glob("*.evidence.json"):
            newer(path, next(folder.glob("*.jsonl")))
        for found in runs_module.list_runs(folder):
            assert "freshness" not in found, found


def test_freshness_is_absent_whenever_the_judgment_covers_the_newest_step():
    look_current, look_old = {"state": "current"}, {"state": "older_than_records"}
    some = ev.build_evidence(plain_rows(steps=range(5)))
    none = ev.build_evidence([])
    assert runs_module.freshness([(some, look_current, 29)], 29) is None
    assert runs_module.freshness([(none, look_old, None)], None) is None  # no step to judge
    assert runs_module.freshness([(None, {"state": "missing"}, 29)], 29) is None  # no evidence: the row says so
    assert runs_module.freshness([(some, look_old, 29), (none, look_current, 29)], 29)["why"] == "records_newer"
    assert runs_module.freshness([(none, look_current, 3), (none, look_old, 29)], 29)["why"] == "nothing_judged"
    assert set(runs_module.FRESHNESS_WHY) == {"records_newer", "thresholds_changed", "nothing_judged"}
    assert runs_module.freshness_text({"current": False, "judged_to_step": 40, "records_to_step": 55,
                                       "why": "records_newer"}) == "as of step 40; records reach step 55"


# --- 4. `first_divergence(..., resolved=True)` ---------------------------------------------------


def make(run_id: str = "a", steps: range = range(120), rank: int = 0, attempt: int | None = None,
         spike_from: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for s in steps:
        value = 1.0 + 0.001 * s
        if spike_from is not None and s >= spike_from:
            value *= 50
        rows.append({"run_id": run_id, "rank": rank, "step": s, "grad_norms": {"w": value}, "loss": 2.0 - 0.001 * s,
                     **({"attempt": attempt} if attempt is not None else {})})
    return rows


def test_resolved_rows_are_not_resolved_again_and_compare_says_they_are_resolved(tmp_path, monkeypatch):
    # `compare`'s rows for a rank's two files given newest first: the restart (steps 50-119) and what
    # the original kept (0-49). Resolving them again by first appearance drops the whole restart.
    original = make("b", range(81), attempt=100)
    restart = [{**row, "ts": 5000.0 + row["step"]} for row in make("b", range(50, 120), attempt=200, spike_from=90)]
    rows = resolve_attempts_across([restart, original])
    assert len(resolve_attempts(rows)) == 50  # what a second resolution would leave
    found = first_divergence(make(), rows, resolved=True)
    assert found["runs"]["b"]["records"] == 120 and "replaced_records" not in found["runs"]["b"]
    assert found["answer"] == "diverged" and found["divergence"]["step"] == 90
    # The direct call's guard still keeps them as well (lane AC4): the same answer without the keyword.
    assert json.dumps(first_divergence(make(), rows), sort_keys=True) == json.dumps(found, sort_keys=True)
    # Raw rows whose first attempt sits wholly above a later one: the guard reads them unresolved, as
    # the several files' rows above look the same. Given resolved, they are read rightly.
    upper = make("b", range(60, 101), attempt=100, spike_from=80)
    lower = make("b", range(0, 51), attempt=200)
    raw = upper + lower
    assert [r["attempt"] for r in load_records(raw)] == [200] * 51
    unresolved = first_divergence(make(), raw)
    assert unresolved["runs"]["b"]["records"] == 92 and unresolved["answer"] == "diverged"
    right = first_divergence(make(), load_records(raw), resolved=True)
    assert right["runs"]["b"]["records"] == 51 and right["answer"] != "diverged"
    # `compare` passes the keyword at each of its calls.
    seen: list[dict[str, Any]] = []
    real = divergence.first_divergence

    def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(divergence, "first_divergence", spy)
    write_capture(tmp_path / "a.rank0.jsonl", make("a"))
    write_capture(tmp_path / "b.rank0.jsonl", make("b", spike_from=90))
    part = compare_module.compare_runs("a", "b", source=tmp_path, paths=False)["divergence"]
    assert part["state"] == "diverged" and part["divergence"]["step"] == 90
    assert seen and all(kwargs.get("resolved") is True for kwargs in seen)
    seen.clear()
    compare_module.compare_runs("a", "b", source=tmp_path, paths=False, rank=0)
    assert seen and all(kwargs.get("resolved") is True for kwargs in seen)


# --- 5. One records file that holds two run IDs ---------------------------------------------------


def test_a_series_of_a_file_with_two_run_ids_is_refused_as_its_incidents_are(tmp_path):
    rows = plain_rows("r1", range(51)) + plain_rows("r2", range(40))
    capture = write_capture(tmp_path / "r1.rank0.jsonl", rows)
    said = refused(capture, "get_series", {"name": "w", "kind": "parameter"})
    assert "more than one run ('r1', 'r2')" in said and "cannot give a series" in said
    assert "more than one run ('r1', 'r2')" in refused(capture, "list_incidents")
    # An evidence file of r1 beside the mixed records: its series is read from them, and refused too.
    evidence_beside(capture, ev.build_evidence(plain_rows("r1", range(51))))
    beside = capture.with_suffix(".evidence.json")
    assert "more than one run ('r1', 'r2')" in refused(beside, "get_series", {"name": "w", "kind": "parameter"})
    # One run's file: the series as before, one point a step.
    one = write_capture(tmp_path / "solo" / "r3.rank0.jsonl", plain_rows("r3", range(51)))
    series = call(one, "get_series", {"name": "w", "kind": "parameter", "max_points": 1000})
    steps = series["steps"]
    assert steps == sorted(set(steps)) and len(steps) == 51



# --- 6. The keys of the detector rules contract (DETECTOR_RULES_CONTRACT.md) ---------------------
#
# The engine lanes build these keys after this lane, so each test gives the readers an evidence
# document that already holds them, taken from the cases' `expected` (`fixtures/rules/cases.json`,
# read here and not changed).

RULES = json.loads((FIXTURES / "rules" / "cases.json").read_text())["cases"]


def expected(name: str) -> Any:
    return json.loads(json.dumps(RULES[name]["expected"]))  # a copy


def document(run: str = "r1", **keys: Any) -> dict[str, Any]:
    """One rank's evidence as `build_evidence` writes it today, with the contract's keys put in."""
    found = ev.build_evidence(plain_rows(run, range(20)))
    found.update(keys)
    return found


def served(tmp_path: Path, doc: dict[str, Any], name: str = "r1") -> Path:
    return ev.write_evidence(doc, tmp_path / f"{name}.rank0.evidence.json")


def test_a_score_is_shown_in_the_cases_words_and_a_null_score_is_never_0(tmp_path):
    # The arithmetic cases: the words a reader shows for each.
    for name, body in RULES.items():
        if name.startswith("score_") and "display" in body["expected"]:
            assert mcp_server.score_display(body["expected"]) == body["expected"]["display"], name
    by_total = expected("score_on_grad_spike_by_the_total")["incident"]
    short = {**by_total, "step": 61, "score": None, "score_note": "short_history",
             "parameters": [{**by_total["parameters"][0], "score": None, "score_note": "no_dispersion"}]}
    activation = expected("score_on_activation_spike")["incident"]
    path = served(tmp_path, document(incidents=[by_total, short, activation]))
    found = call(path, "get_incident", {"id": "grad_spike@60"})
    assert found["incident"] == by_total  # the incident whole, `score` beside each `ratio`
    assert found["scores"] == [
        {"at": "incident", "ratio": 9.8, "score": by_total["score"], "display": "score 593.6"},
        {"at": "parameters: w", "ratio": by_total["parameters"][0]["ratio"], "score": by_total["parameters"][0]["score"],
         "display": "score 981.8"}]
    nulls = call(path, "get_incident", {"id": "grad_spike@61"})["scores"]
    assert [s["display"] for s in nulls] == ["no score: fewer than 10 earlier values",
                                             "no score: the earlier values had zero median absolute deviation"]
    assert all(s["score"] is None for s in nulls)
    assert call(path, "get_incident", {"id": "activation_spike@80"})["scores"] == [
        {"at": "modules: blk (abs_max)", "ratio": 10.2, "score": activation["modules"][0]["score"], "display": "score 620.5"}]
    # An incident written before the score: no `scores`.
    before = {**{k: v for k, v in by_total.items() if k != "score"},
              "parameters": [{k: v for k, v in by_total["parameters"][0].items() if k != "score"}]}
    old = served(tmp_path, document(incidents=[before]), "old")
    assert "scores" not in call(old, "get_incident", {"id": "grad_spike@60"})


def test_a_router_collapsed_since_step_0_is_said_to_be_reportable_later_in_every_reader(tmp_path):
    collapsed = expected("start_ten_samples_is_reported")["incident"]
    starting = expected("start_and_the_capture_ends")["starting_state"]
    doc = document(incidents=[collapsed])
    doc["kind_checks"]["moe_imbalance"] = {**doc["kind_checks"]["moe_imbalance"], "starting_state": starting}
    path = served(tmp_path, doc)
    listed = call(path, "list_incidents")
    [brief] = listed["incidents"]
    assert (brief["step"], brief["since_start"], brief["reportable_step"]) == (0, True, 9)
    assert brief["when"] == "collapsed since step 0, reportable at step 9"
    assert listed["kind_checks"]["moe_imbalance"] == {"starting_state": [{"router": "r", "samples": 5}]}
    one = call(path, "get_incident", {"id": brief["id"]})
    assert one["when"] == "collapsed since step 0, reportable at step 9" and one["incident"] == collapsed
    # `runs`: the first incident's cell, and `compare`'s line.
    capture = write_capture(tmp_path / "local" / "m1.rank0.jsonl", plain_rows("m1", range(20)))
    newer(evidence_beside(capture, {**document("m1"), "incidents": [collapsed]}), capture)
    row = only_row(tmp_path / "local")
    assert row["first_incident"] == {"kind": "moe_imbalance", "step": 0, "since_start": True, "reportable_step": 9}
    assert "moe_imbalance since step 0, reportable at step 9" in runs_module.listing_text([row], tmp_path)
    other = served(tmp_path, document("r2"), "r2")
    result = compare_module.compare_runs(str(other), str(path))
    [entry] = [e for e in result["incidents"]["entries"] if e["kind"] == "moe_imbalance"]
    assert entry["first_step"] == {"B": 0} and entry["reportable_step"] == {"B": 9}
    assert "first at B step 0 (collapsed since step 0, reportable at step 9)" in compare_module.summary_text(result)
    # A router judged as today has none of it.
    later = {**collapsed, "step": 40}
    for key in ("since_start", "reportable_step"):
        later.pop(key)
    assert not {"since_start", "when"} & set(call(served(tmp_path, document(incidents=[later]), "r3"),
                                                   "list_incidents")["incidents"][0])


def test_the_learning_rate_on_update_spike_and_the_measurements_are_carried_as_descriptions(tmp_path):
    spike = expected("update_spike_carries_the_learning_rate")["incident"]
    blocks = {key: expected(name)[key] for key, name in (
        ("gradient_trend", "gradient_trend_a_steady_rise"), ("attention_trends", "attention_trends_a_slow_rise"),
        ("update_trends", "update_trends_one_parameter_collapses"), ("weight_jumps", "weight_jumps_one_tensor_jumps"))}
    grad = expected("score_on_grad_spike_by_the_total")["incident"]
    path = served(tmp_path, document(incidents=[spike, grad], **blocks))
    listed = call(path, "list_incidents")
    for key, block in blocks.items():
        assert listed[key] == block, key
    on_update = call(path, "get_incident", {"id": f"update_spike@{spike['step']}"})
    assert (on_update["incident"]["lr"], on_update["incident"]["lr_before"]) == ([0.003, 0.0003], [0.001, 0.0001])
    assert on_update["update_trends"] == blocks["update_trends"] and on_update["weight_jumps"] == blocks["weight_jumps"]
    assert "gradient_trend" not in on_update
    assert call(path, "get_incident", {"id": "grad_spike@60"})["gradient_trend"] == blocks["gradient_trend"]
    # A case whose block is absent (`gradient_trend` under six points): no key.
    assert expected("gradient_trend_under_six_points")["gradient_trend"] is None
    # (On the merged tree `build_evidence` itself gives a `gradient_trend` for `document()`'s rows,
    # so the document that lacks the blocks is made by taking them out.)
    bare = {key: value for key, value in document().items() if key not in blocks}
    assert not set(blocks) & set(call(served(tmp_path, bare, "bare"), "list_incidents"))
    # The compact file keeps every one of these keys unchanged (it keeps every key but `samples`).
    whole = document(incidents=[spike, grad], **blocks)
    kept = compact.compact_evidence(whole)
    assert {k: kept[k] for k in blocks} == blocks and kept["incidents"] == [spike, grad]
    # Described, never worded as a finding.
    text = {tool["name"]: tool["description"] for tool in mcp_server.TOOLS}["list_incidents"]
    assert "are descriptions of the run: numbers with no threshold" in text
    assert "They are not findings or warnings" in text


def test_the_routing_source_is_said_in_the_cases_words(tmp_path):
    for name in ("moe_source_scores_are_what_the_router_asked_for", "moe_source_indices_at_an_unknown_point",
                 "moe_source_a_map_after_dispatch", "moe_source_absent"):
        want = expected(name)
        folder = tmp_path / name
        # From the records, as `get_series` reads a router's series.
        capture = write_capture(folder / "m.rank0.jsonl", [{"schema_version": 1, "run_id": "m", "rank": 0, "attempt": 1,
                                                           "step": r["step"], **{k: v for k, v in r.items() if k != "step"}}
                                                          for r in RULES[name]["records"]])
        series = call(capture, "get_series", {"name": "r", "kind": "router"})
        # From an evidence document holding the key, as the engines will write it.
        doc = document(**({"moe_source": want["moe_source"]} if want["moe_source"] is not None else {}),
                       incidents=[expected("start_ten_samples_is_reported")["incident"]])
        path = served(folder, doc)
        listed, incident = call(path, "list_incidents"), call(path, "get_incident", {"id": "moe_imbalance@0"})
        for answer in (series, listed, incident):
            if want["moe_source"] is None:
                assert "moe_source" not in answer and "moe_source_words" not in answer, name
            else:
                assert answer["moe_source"] == want["moe_source"] and answer["moe_source_words"] == want["words"], name
        said = json.dumps([series, listed, incident])
        assert ("dispatched assignments" in said) == (name == "moe_source_a_map_after_dispatch"), name


def test_layers_not_read_and_the_parameters_not_judged_by_rank(tmp_path):
    some = expected("layers_not_read_some_modules")
    doc = document(not_checked=some["not_checked"])
    for kind, entry in some["kind_checks"].items():
        doc["kind_checks"][kind] = {**doc["kind_checks"][kind], **entry}
    listed = call(served(tmp_path, doc), "list_incidents")
    assert some["not_checked"][0] in listed["not_checked"] and listed["not_checked"][0]["coverage"] == "partial"
    assert listed["kind_checks"] == {kind: {"modules_not_judged": ["blk1"]} for kind in some["kind_checks"]}
    # The all-ranks view's entry, from an all-ranks evidence file holding the contract's key.
    stated = ev.kind_checks({**dict.fromkeys(ev.INPUT_NEEDS, 0), "steps": 10}, [])
    view = ev.all_ranks_evidence(store(tmp_path / "st", "run", {0: {"kind_checks": stated}, 1: {"kind_checks": stated}}))
    by_rank = expected("the_all_ranks_view_names_them_by_rank_when_ranks_differ")["kind_checks"]["stalled_weights"]
    view["kind_checks"]["stalled_weights"] = {**view["kind_checks"]["stalled_weights"], **by_rank}
    path = ev.write_evidence(view, tmp_path / "run.all-ranks.evidence.json")
    assert call(path, "get_all_ranks_status")["kind_checks"] == {"stalled_weights": by_rank}


def test_where_capture_stopped_step_0_null_and_no_step_stay_three_things(tmp_path):
    stopped = {rank: expected(name)["status"]["capture_stopped"] for rank, name in enumerate(
        ("capture_stopped_at_step_0", "capture_stopped_before_any_step", "capture_stopped_from_an_older_sender"))}
    folder = store(tmp_path, "run", {0: {}, 1: {}, 2: {}, 3: {}})
    status = json.loads((folder / "watchdog.json").read_text())
    for entry in status["ranks"]:
        if entry["rank"] in stopped:
            entry["capture_stopped"] = stopped[entry["rank"]]
    (folder / "watchdog.json").write_text(json.dumps(status))
    answer = call(folder, "get_all_ranks_status")
    shown = {item["rank"]: item.get("capture_stopped", "absent") for item in answer["per_rank"]}
    assert shown == {0: {"reason": "sink_error", "step": 0}, 1: {"reason": "sink_error", "step": None},
                     2: {"reason": "sink_error"}, 3: "absent"}
    assert call(folder, "get_all_ranks_status", {"rank": 1})["rank"]["capture_stopped"] == {"reason": "sink_error", "step": None}
    assert "capture_stopped" not in call(folder, "get_all_ranks_status", {"rank": 3})["rank"]
    text = {tool["name"]: tool["description"] for tool in mcp_server.TOOLS}["get_all_ranks_status"]
    assert "`step` 0 is step 0; `step` null means it committed none; no `step` key" in text
