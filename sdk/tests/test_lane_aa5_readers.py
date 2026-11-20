"""Lane AA5 (wave AA): the run listing, `compare` and the MCP server, one test or more for each
item it takes.

- L21, the part left open ("1+"): a file named for a rank and larger than a megabyte is counted
  from every line, with no manifest. The part before its last megabyte is scanned for each
  line's rank and attempt anyway (L21's first part, lane Z5); the count now reads it. The count
  stays not known only where the scan could not read every line.
- L22, the part left open: `compare` and `compare_runs` show the rows a store run's server
  dropped for a null key, as an upper bound; a run that dropped none reads as before.
- From lane Z2's hand-over: the evidence of one rank picked out of a file of several ranks says
  which rank it judged (`rank`) and the ranks the file held (`ranks_in_source`); the listing's
  row for such a file names that rank.

The records and the store folders are built here (the status by hand, from the shape
`server/src/watchdog.rs` writes near line 1363), so nothing depends on another lane's test file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import compare as cmp
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill import runs as runs_module

RUN = "lane-aa5"
PAD = "x" * 20_000  # a row of about 20 KB, so a few dozen rows pass the listing's last megabyte
RECORDED = Path.home() / "Downloads" / "runs"


def row(step: int, *, rank: int = 0, attempt: int = 1, run: str = RUN, pad: bool = False) -> dict[str, Any]:
    found: dict[str, Any] = {
        "schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": step,
        "ts": 1_000.0 + step, "layers": {}, "grad_norm": 1.0, "grad_norms": {"w": 1.0}, "lr": [0.1],
    }
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


def only_row(folder: Path) -> dict[str, Any]:
    (found,) = tensorquill.list_runs(folder)
    return found


def big_rank_file(path: Path, *, other_rank_at: int | None = None, rows: int = 120) -> Path:
    """A file of rank 0 over two megabytes; `other_rank_at` puts one row of rank 1 near its start,
    in the part the listing does not parse."""
    found = [row(step, pad=True) for step in range(rows)]
    if other_rank_at is not None:
        found.insert(other_rank_at, row(other_rank_at, rank=1, pad=True))
    write(path, found)
    assert path.stat().st_size > 2 * runs_module._TAIL_BYTES
    return path


def judged(path: Path, rank: int | None = None) -> None:
    """The evidence file beside `path`, as `python -m tensorquill <path> [--rank N]` writes it,
    a second after the records so it reads as current."""
    ev.write_evidence(ev.build_evidence(path, rank=rank), path.with_suffix(".evidence.json"))
    later = path.stat().st_mtime + 1
    os.utime(path.with_suffix(".evidence.json"), (later, later))


# --- 1. L21: "1+" -------------------------------------------------------------------------------


def test_l21_a_large_rank_file_with_no_manifest_is_counted(tmp_path):
    """Fail before: `rank_count` None, "middle" in `rank_count_not_known`, and "1+" printed."""
    big_rank_file(tmp_path / f"{RUN}.rank0.jsonl")
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0], 1) and "rank_count_not_known" not in found
    text = runs_module.listing_text([found], tmp_path)
    assert "1+" not in text and "ranks not counted" not in text


def test_l21_a_rank_seen_only_in_the_middle_is_counted_and_the_file_is_one_of_several_ranks(tmp_path):
    """Fail before: ranks [0], `rank_count` None. Rank 1's only row is in the scanned part: the
    file holds ranks 0 and 1, as `compare` reads it, and its evidence file is one rank's of
    several."""
    path = big_rank_file(tmp_path / f"{RUN}.rank0.jsonl", other_rank_at=2)
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0, 1], 2) and "rank_count_not_known" not in found
    assert found["evidence"]["rank"] is None and "holds ranks 0, 1" in found["evidence"]["detail"]
    assert found["last_step"] == 119  # rank 0's last line; rank 1 has no row in the last megabyte
    opened = cmp.open_run(path, rank=None)
    assert opened["read_from"]["ranks"] == [0, 1]
    # The MCP listing gives the same count.
    (listed,) = mcp_server.list_runs(mcp_server.Source(path), {})["runs"]
    assert (listed["ranks"], listed["rank_count"]) == ([0, 1], 2) and "rank_count_not_known" not in listed


def test_l21_a_small_rank_file_holding_another_rank_is_counted_too(tmp_path):
    """Fail before: `rank_count` None (`test_lane_y3_readers.py::test_xc5_a_rank_file_holding_
    another_ranks_rows_is_not_counted_as_one_rank` allows either). A file under a megabyte is
    read on every line, so the count is the ranks its lines name."""
    write(tmp_path / "job.rank0.jsonl", [row(0, rank=0, run="job"), row(0, rank=1, run="job"), row(1, rank=0, run="job")])
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0, 1], 2) and "rank_count_not_known" not in found


def test_l21_a_world_size_below_the_ranks_found_does_not_hide_them(tmp_path):
    """Fail before: ranks [0], `rank_count` 1 (the manifest's 1 settled the count, and rank 1's
    row was not seen). A call for the author: the records name two ranks and are counted so; the
    manifest's world size is in the row's setup as written."""
    big_rank_file(tmp_path / f"{RUN}.rank0.jsonl", other_rank_at=2)
    manifest(tmp_path, world_size=1)
    found = only_row(tmp_path)
    assert (found["ranks"], found["rank_count"]) == ([0, 1], 2)


def test_l21_a_world_size_above_the_ranks_found_counts_the_rest_as_without_a_file(tmp_path):
    """Fail before: rank 1 was one of the ranks with no file (`missing`); it is in rank 0's file."""
    path = big_rank_file(tmp_path / f"{RUN}.rank0.jsonl", other_rank_at=2)
    known = runs_module._local_ranks({0: path}, {"world_size": 4})
    assert known["ranks"] == [0, 1, 2, 3] and sorted(known["missing"]) == [2, 3] and "why" not in known
    assert known["held"][path] == [0, 1] and known["several"] == [path]


def test_l21_a_scan_that_cannot_read_every_line_leaves_the_count_not_known(tmp_path, monkeypatch):
    """A control (passes before and after): where the scan stops short (a file cut or unreadable
    while it is read), the middle is not read, so with no manifest the count is not known."""
    big_rank_file(tmp_path / f"{RUN}.rank0.jsonl", other_rank_at=2)
    runs_module._SCANNED.clear()
    monkeypatch.setattr(runs_module, "_scan_lines", lambda file, start, before, found: start)
    found = only_row(tmp_path)
    assert found["ranks"] == [0] and found["rank_count"] is None
    assert "middle" in found["rank_count_not_known"]
    assert "1+" in runs_module.listing_text([found], tmp_path)


def test_l21_a_scan_reports_whether_it_read_every_line(tmp_path, monkeypatch):
    """Fail before: `runs._scan_file` did not exist."""
    path = big_rank_file(tmp_path / f"{RUN}.rank0.jsonl", other_rank_at=2)
    _, _, before = runs_module._tail_from(path)
    runs_module._SCANNED.clear()
    found, complete = runs_module._scan_file(path, before)
    assert complete and sorted(found) == [0, 1]
    assert runs_module._attempts_in_file(path, before) == found
    runs_module._SCANNED.clear()
    assert runs_module._scan_file(tmp_path / "gone.rank0.jsonl", before) == ({}, False)
    monkeypatch.setattr(runs_module, "_scan_lines", lambda file, start, before, found: start)
    runs_module._SCANNED.clear()
    assert runs_module._scan_file(path, before)[1] is False


def test_l21_the_count_costs_no_read_beyond_the_scan(tmp_path, monkeypatch):
    """A control (passes before and after): the count reads no line again. The file is scanned
    once a listing, as for L21's attempts, and never parsed whole."""
    big_rank_file(tmp_path / f"{RUN}.rank0.jsonl", other_rank_at=2)
    runs_module._SCANNED.clear()
    real = runs_module._scan_lines
    starts: list[int] = []

    def watched(file, start, before, found):
        starts.append(start)
        return real(file, start, before, found)

    monkeypatch.setattr(runs_module, "_scan_lines", watched)
    monkeypatch.setattr(runs_module, "_file_lines", lambda *a, **k: pytest.fail("the file was parsed whole"))
    only_row(tmp_path)
    assert starts == [0]


@pytest.mark.skipif(not RECORDED.is_dir(), reason="the recorded captures are not on this machine")
def test_l21_the_recorded_captures_named_for_a_rank_are_counted_as_one_rank(tmp_path):
    """Fail before: each listed as "1+" with no manifest. Each recorded capture (6 to 17 MB, one
    rank), linked here under a rank file's name: read, never written."""
    files = sorted(RECORDED.glob("*/*.jsonl"))
    assert files
    for path in files:
        first = json.loads(path.open("rb").readline())
        folder = tmp_path / path.parent.name / path.stem
        folder.mkdir(parents=True)
        (folder / f"{first['run_id']}.rank{first.get('rank', 0)}.jsonl").symlink_to(path)
        found = only_row(folder)
        assert (found["ranks"], found["rank_count"]) == ([first.get("rank", 0)], 1), path
        assert "rank_count_not_known" not in found, path
