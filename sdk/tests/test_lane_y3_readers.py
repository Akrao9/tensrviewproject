"""Lane Y3 (wave Y): the run listing, `compare` and the MCP server, one test or more for each row
it fixes.

- XC4: in one rank file, a late row of an attempt that a later one replaced does not move the
  rank's last step: the listing reads the file whole, as `compare` does, when its last megabyte
  shows more than one attempt.
- XC5: a file named for one rank that holds another rank's rows is not counted as that rank
  alone: every whole line of the last megabyte is read for its rank, and a file whose middle
  is not read gives a definite count only when the run manifest's world size settles it.
- XC6: a world size above the ranks a reader counts (`MAX_EXPECTED_RANKS`) is not judged as
  the ranks that have files: the run is unknown, the ranks not covered are a count, and no
  list of them is made.
- XC7: the MCP server reads a capture's run manifest at every call, so one written or replaced
  after the first query is seen.
- L14: in `compare`, a rank a restart left judged by an earlier attempt (`earlier_attempt`) has
  its reason, with the parts read or not, and when it is the one rank read.
- L15, the MCP half: `list_incidents` gives the evidence's `head_trends` when it has them.

The records are built here, so nothing depends on another lane's test file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import compact as compact_module
from tensorquill import compare as cmp
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill import runs as runs_module

RUN = "lane-y3"
PAD = "x" * 20_000  # a row of about 20 KB, so a few dozen rows pass the listing's last megabyte


def row(step: int, *, rank: int = 0, attempt: int = 1, run: str = RUN, ts: float | None = None,
        pad: bool = False, head: bool = False) -> dict[str, Any]:
    found: dict[str, Any] = {
        "schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": step,
        "ts": 1_000.0 + step if ts is None else ts, "layers": {}, "grad_norm": 1.0, "grad_norms": {"w": 1.0},
        "lr": [0.1],
    }
    if head:
        found["probe"] = {"head": {"module": "lm_head", "norm": 10.0 + step, "mean": 0.1, "log_z": 3.0, "gap": 0.0,
                                   "gap_relative": 0.0, "positions": 8}}
    if pad:
        found["pad"] = PAD
    return found


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def manifest(folder: Path, run: str = RUN, **fields: Any) -> Path:
    path = folder / f"{run}.run.json"
    path.write_text(json.dumps({"schema_version": 1, "run_id": run, "project": "p", "name": run,
                                "created_at": "2026-11-20T10:00:00Z", **fields}), encoding="utf-8")
    return path


def judged(path: Path, rank: int | None = None) -> None:
    """The evidence file beside `path`, as `python -m tensorquill <path>` writes it, a second
    after the records so it reads as current."""
    ev.write_evidence(ev.build_evidence(path, rank=rank), path.with_suffix(".evidence.json"))
    later = path.stat().st_mtime + 1
    os.utime(path.with_suffix(".evidence.json"), (later, later))


def only_row(folder: Path) -> dict[str, Any]:
    (found,) = tensorquill.list_runs(folder)
    return found


# --- XC4 ----------------------------------------------------------------------------------------
#
# One file of one rank: attempt 900 writes steps 0 and 100, attempt 2 resumes at step 30, and a
# late row of attempt 900 (step 101) lands after it. Attempt 2 replaced 900 from step 30, so the
# rank stands at step 30; the file's last line said 101.

RESTART = [(900, 0), (900, 100), (2, 30), (900, 101)]


def restarted(path: Path, *, rank: int = 0, run: str = RUN, pad_before: int = 0) -> Path:
    """`RESTART`, with `pad_before` large rows of attempt 900 (steps 1, 2, ...) after its first."""
    pairs = [(*RESTART[0], False), *((900, s, True) for s in range(1, pad_before + 1)),
             *((attempt, step, False) for attempt, step in RESTART[1:])]
    return write(path, [row(step, rank=rank, attempt=attempt, run=run, ts=2_000.0 + at, pad=pad)
                        for at, (attempt, step, pad) in enumerate(pairs)])


def test_xc4_a_late_row_of_a_replaced_attempt_does_not_move_the_last_step(tmp_path):
    path = restarted(tmp_path / f"{RUN}.rank0.jsonl")
    kept = ev.resolve_attempts(ev.load_records(path, all_attempts=True))
    assert [r["step"] for r in kept] == [0, 30]  # as `compare` and the detectors read the file
    found = only_row(tmp_path)
    assert found["last_step"] == 30
    assert cmp.open_run(path)["last_step"] == 30


def test_xc4_holds_when_the_file_is_larger_than_the_part_the_listing_reads(tmp_path):
    # Attempt 900's first record is more than a megabyte before the restart; the last megabyte
    # names both attempts, so the file is read whole.
    path = restarted(tmp_path / f"{RUN}.rank0.jsonl", pad_before=60)
    assert path.stat().st_size > runs_module._TAIL_BYTES
    assert only_row(tmp_path)["last_step"] == 30


def test_xc4_in_a_run_of_several_ranks_the_restarted_rank_counts_from_its_current_attempt(tmp_path):
    restarted(tmp_path / f"{RUN}.rank0.jsonl")
    write(tmp_path / f"{RUN}.rank1.jsonl", [row(step, rank=1) for step in range(11)])
    found = only_row(tmp_path)
    assert found["ranks"] == [0, 1] and found["last_step"] == 30


def test_xc4_a_file_of_several_ranks_counts_each_rank_from_its_current_attempt(tmp_path):
    # Rank 1 has one attempt and stops at step 12; rank 0's late row of attempt 900 is its last.
    triples = [(0, 900, 0), (1, 1, 0), (0, 900, 100), (1, 1, 11), (0, 2, 30), (0, 900, 101), (1, 1, 12)]
    write(tmp_path / "both.jsonl", [row(step, rank=rank, attempt=attempt, ts=2_000.0 + at)
                                    for at, (rank, attempt, step) in enumerate(triples)])
    found = only_row(tmp_path)
    assert found["ranks"] == [0, 1] and found["last_step"] == 30


def test_xc4_a_resume_that_fills_the_last_megabyte_is_not_read_again(tmp_path, monkeypatch):
    # Attempt 1 writes steps 0 to 9; attempt 2 resumes at step 5 and its rows fill the last
    # megabyte. Its last line stands, so the file is not read whole on every listing.
    rows = [row(step, attempt=1, ts=2_000.0 + step) for step in range(10)]
    rows += [row(step, attempt=2, ts=3_000.0 + step, pad=True) for step in range(5, 125)]
    path = write(tmp_path / f"{RUN}.rank0.jsonl", rows)
    assert path.stat().st_size > 2 * runs_module._TAIL_BYTES
    monkeypatch.setattr(runs_module, "_rank_across_files", lambda *a, **k: pytest.fail("the file was read whole"))
    manifest(tmp_path, world_size=1)
    assert only_row(tmp_path)["last_step"] == 124


def test_xc4_a_file_of_one_attempt_is_still_read_from_its_ends_only(tmp_path, monkeypatch):
    path = write(tmp_path / f"{RUN}.rank0.jsonl", [row(step) for step in range(20)])
    monkeypatch.setattr(runs_module, "_rank_across_files", lambda *a, **k: pytest.fail("the file was read whole"))
    assert only_row(tmp_path)["last_step"] == 19
    assert path.exists()


# --- XC5 ----------------------------------------------------------------------------------------


def test_xc5_a_rank_file_holding_another_ranks_rows_is_not_counted_as_one_rank(tmp_path):
    write(tmp_path / "job.rank0.jsonl", [row(0, rank=0, run="job"), row(0, rank=1, run="job"), row(1, rank=0, run="job")])
    found = only_row(tmp_path)
    assert found["ranks"] == [0, 1]
    assert found["rank_count"] != 1
    if found["rank_count"] is None:
        assert "job.rank0.jsonl" in found["rank_count_not_known"]
    # The manifest's world size settles the count.
    manifest(tmp_path, "job", world_size=2)
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0, 1], 2)


def test_xc5_an_evidence_file_beside_such_a_file_names_the_rank_it_judged(tmp_path):
    # Rewritten by lane AA5 (Nov 20, 2026): it pinned `evidence.rank` None ("given to no rank").
    # The evidence names the rank it judged and the ranks the file held (lane Z2), so the row
    # says rank 0's, and the other rank is still not judged.
    path = write(tmp_path / "job.rank0.jsonl",
                 [row(0, rank=0, run="job"), row(0, rank=1, run="job"), row(1, rank=0, run="job")])
    judged(path, rank=0)
    found = only_row(tmp_path)
    assert found["evidence"]["rank"] == 0 and found["evidence"]["state"] == "ranks_unknown"
    assert found["evidence"]["ranks_in_source"] == [0, 1]
    assert found["run_state"] == "unknown"


def big_rank_file(path: Path, *, other_rank_at: int | None = None, run: str = RUN) -> Path:
    rows = [row(step, run=run, pad=True) for step in range(120)]
    if other_rank_at is not None:
        rows.insert(other_rank_at, row(other_rank_at, rank=1, run=run, pad=True))
    write(path, rows)
    assert path.stat().st_size > 2 * runs_module._TAIL_BYTES
    return path


def test_xc5_a_large_rank_file_with_no_world_size_is_counted_from_every_line(tmp_path):
    # Rewritten by lane AA5 (Nov 20, 2026): it pinned "1+" (`rank_count` None, "middle" in the
    # reason). Rank 1's row is in the middle, which the listing scans for each line's rank (L21):
    # every line is read, so the file holds ranks 0 and 1 and the count is settled without a
    # manifest. Where the scan cannot read every line the count is still not known
    # (`test_lane_aa5_readers.py`).
    big_rank_file(tmp_path / f"{RUN}.rank0.jsonl", other_rank_at=2)
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0, 1], 2) and "rank_count_not_known" not in found
    assert "1+" not in runs_module.listing_text([found], tmp_path)


def test_xc5_a_large_rank_file_is_counted_when_the_manifest_settles_it(tmp_path):
    big_rank_file(tmp_path / f"{RUN}.rank0.jsonl")
    manifest(tmp_path, world_size=1)
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0], 1) and "rank_count_not_known" not in found
    manifest(tmp_path, world_size=2)
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0, 1], 2)
    assert found["evidence"]["unknown_ranks"] >= 1  # rank 1 has no file: unknown, never clean


def test_xc5_a_small_rank_file_of_its_rank_alone_is_counted_without_a_manifest(tmp_path):
    write(tmp_path / f"{RUN}.rank0.jsonl", [row(step) for step in range(30)])
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0], 1) and "rank_count_not_known" not in found


# --- XC6 ----------------------------------------------------------------------------------------


def healthy_rank0(folder: Path) -> Path:
    path = write(folder / f"{RUN}.rank0.jsonl", [row(step) for step in range(40)])
    judged(path)
    return path


def test_xc6_a_world_size_above_the_cap_is_not_judged_as_the_ranks_with_files(tmp_path):
    healthy_rank0(tmp_path)
    assert only_row(tmp_path)["run_state"] == "healthy"
    world = ev.MAX_EXPECTED_RANKS + 1
    manifest(tmp_path, world_size=world)
    started = time.monotonic()
    found = only_row(tmp_path)
    assert time.monotonic() - started < 10
    assert found["run_state"] == "unknown"
    assert found["ranks"] == [0]  # the ranks seen; the others are a count, never a list
    assert found["evidence"]["state"] == "ranks_unknown"
    assert found["evidence"]["unknown_ranks"] == world - 1
    assert found["rank_count"] is None and f"{world:,}" in found["rank_count_not_known"]
    assert found["end"]["state"] == "not_closed"
    assert len(json.dumps(found)) < 20_000
    text = runs_module.listing_text([found], tmp_path)
    assert "unknown" in text and f"{world:,}" in text


def test_xc6_a_world_size_of_two_is_still_unknown(tmp_path):
    healthy_rank0(tmp_path)
    manifest(tmp_path, world_size=2)
    found = only_row(tmp_path)
    assert (found["run_state"], found["ranks"], found["rank_count"]) == ("unknown", [0, 1], 2)


def test_xc6_compare_does_not_judge_a_run_above_the_cap_as_whole(tmp_path):
    path = healthy_rank0(tmp_path)
    manifest(tmp_path, world_size=ev.MAX_EXPECTED_RANKS + 1)
    opened = cmp.open_run(RUN, tmp_path)
    assert opened["evidence"]["run_state"]["state"] == "unknown"
    assert opened["read_from"]["ranks_not_counted"] == ev.MAX_EXPECTED_RANKS
    assert len(json.dumps(opened["read_from"], default=str)) < 20_000
    assert path.exists()


# --- XC7 ----------------------------------------------------------------------------------------


def test_xc7_a_manifest_written_after_the_first_query_is_seen(tmp_path, monkeypatch):
    path = write(tmp_path / f"{RUN}.rank0.jsonl", [row(step) for step in range(10)])
    source = mcp_server.Source(path)
    assert source.open(None)["evidence"]["run"]["project"] == "uncategorized"
    built = []
    real = mcp_server.build_evidence
    monkeypatch.setattr(mcp_server, "build_evidence", lambda *a, **k: built.append(1) or real(*a, **k))
    stamp = path.stat()
    manifest(tmp_path, project="granite", name="baseline")
    assert source.open(None)["evidence"]["run"] == {"project": "granite", "name": "baseline", "run_id": RUN}
    # Replaced by a rename, as `write_manifest` writes it.
    scratch = tmp_path / "scratch.json"
    scratch.write_text(json.dumps({"run_id": RUN, "project": "llama", "name": "after"}), encoding="utf-8")
    os.replace(scratch, tmp_path / f"{RUN}.run.json")
    assert source.open(None)["evidence"]["run"]["project"] == "llama"
    assert source.capture_evidence(path, 0)["run"]["name"] == "after"
    (tmp_path / f"{RUN}.run.json").unlink()
    assert source.open(None)["evidence"]["run"]["project"] == "uncategorized"
    # The records did not change, so the detectors did not run again.
    assert (path.stat().st_mtime_ns, path.stat().st_size) == (stamp.st_mtime_ns, stamp.st_size)
    assert built == []


# --- L14 ----------------------------------------------------------------------------------------

STORE_RUN = "y3store"


def store(tmp_path: Path) -> Path:
    """A run folder in a store: rank 0 current, rank 1 current with a report the status shows to
    be an earlier attempt's (the job started again as attempt 2 and stored nothing yet)."""
    run = tmp_path / "store" / "runs" / STORE_RUN
    entries = [{"rank": 0, "attempt": 1, "last_step": 9},
               {"rank": 1, "attempt": 2, "last_step": None, "attempt_without_steps": True}]
    for entry in entries:
        folder = run / f"rank={entry['rank']}" / "attempt=1"
        folder.mkdir(parents=True)
        (folder / "part-00000000-00000009.parquet").write_bytes(b"")
        incidents = [{"kind": "non_finite", "step": 5, "layers": ["a"], "grads": []}] if entry["rank"] == 1 else []
        (folder.parent / "incidents.json").write_text(json.dumps({
            "schema_version": 1, "source": "tensorquill-server", "run_id": STORE_RUN, "rank": entry["rank"],
            "attempt": 1, "last_step": 9, "steps": 10, "incidents": incidents, "checked": list(ev.SERVER_KINDS),
            "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN},
        }))
    (run / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "run_id": STORE_RUN, "summary": "active", "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": 2, "missing_ranks": [], "incidents": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **e} for e in entries],
        "pauses": {}, "recommended_silent_seconds": None,
    }))
    return run


@pytest.mark.parametrize("read_parts", [True, False])
def test_l14_a_rank_judged_by_an_earlier_attempt_has_its_reason(tmp_path, read_parts):
    store(tmp_path)
    read = cmp.open_run(STORE_RUN, tmp_path / "store", read_parts=read_parts)["read_from"]
    assert (read["judged_ranks"], read["unknown_ranks"]) == ([0], [1])
    why = read["unknown_why"][1]
    assert isinstance(why, str) and why.startswith(f"{ev.EARLIER_ATTEMPT}: ") and "started again as attempt 2" in why


def test_l14_the_reason_is_in_the_comparison_and_its_summary(tmp_path):
    run = store(tmp_path)
    other = write(tmp_path / "other" / "other.rank0.jsonl", [row(step, run="other") for step in range(10)])
    result = tensorquill.compare_runs(run, other)
    read = result["runs"][0]["read_from"]
    assert read["unknown_why"][1].startswith(ev.EARLIER_ATTEMPT)
    assert "1 of 2 ranks" in cmp.summary_text(result)


def test_l14_the_one_rank_read_alone_has_its_reason_and_is_not_judged(tmp_path):
    store(tmp_path)
    opened = cmp.open_run(STORE_RUN, tmp_path / "store", rank=1)
    read = opened["read_from"]
    assert read["unknown_ranks"] == [1]
    assert read["unknown_why"][1].startswith(f"{ev.EARLIER_ATTEMPT}: ")
    assert opened["evidence"]["run_state"]["state"] == "unknown"
    # Rank 0 read alone is judged from its report, as before.
    assert "unknown_ranks" not in cmp.open_run(STORE_RUN, tmp_path / "store", rank=0)["read_from"]


# --- L15 ----------------------------------------------------------------------------------------


def test_l15_list_incidents_gives_the_head_trends_of_a_capture(tmp_path):
    path = write(tmp_path / f"{RUN}.rank0.jsonl", [row(step, head=True) for step in range(10)])
    source = mcp_server.Source(path)
    answer = mcp_server.list_incidents(source, {})
    expected = ev.head_logit_trends(ev.load_records(path))
    assert expected is not None and answer["head_trends"] == expected
    # Through the protocol too, as an agent reads it.
    reply = mcp_server.call_tool(source, 1, {"name": "list_incidents", "arguments": {}})
    assert json.loads(reply["result"]["content"][0]["text"])["head_trends"] == expected


@pytest.mark.parametrize("compact", [False, True])
def test_l15_list_incidents_gives_the_head_trends_of_an_evidence_file(tmp_path, compact):
    path = write(tmp_path / f"{RUN}.rank0.jsonl", [row(step, head=True) for step in range(10)])
    evidence = ev.build_evidence(path)
    if compact:
        target = compact_module.write_compact(evidence, path.with_suffix(".evidence.compact.json"))
    else:
        target = ev.write_evidence(evidence, path.with_suffix(".evidence.json"))
    answer = mcp_server.list_incidents(mcp_server.Source(target), {})
    assert answer["head_trends"] == evidence["head_trends"]


def test_l15_no_head_trends_key_where_the_evidence_has_none(tmp_path):
    path = write(tmp_path / f"{RUN}.rank0.jsonl", [row(step) for step in range(10)])
    assert "head_trends" not in mcp_server.list_incidents(mcp_server.Source(path), {})
