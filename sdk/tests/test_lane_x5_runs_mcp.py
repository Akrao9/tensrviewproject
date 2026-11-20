"""Lane X5 (wave X): the run listing, the MCP server and the compact file, one test or more for
each row it fixes.

- CLI-13, open part: `runs` reads every record of a rank whose records are in several files, so
  an attempt seen only in the middle of a file orders the attempts as `compare` orders them
  (contract 1), and the `end` cell is taken from the file of the rank's newest attempt.
- DET-17, open part: `manifest.py`, `runs.py` and `mcp_server.py` read `rank=<n>` folder names
  with `evidence._rank_number` (ASCII digits only).
- CLI-14: the MCP server serves the probe's head and attention series.
- DET-13: the compact file's table at a window's first step is made when the full evidence has
  no earlier step that routed tokens either, and a later starvation onset gets a window.

The records are built here, so nothing depends on another lane's test file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import compare as cmp
from tensorquill import runs as runs_module
from tensorquill.sink import write_end_marker

RUN = "lane-x5"


def rows(attempt: int, steps: range, first_ts: float, *, rank: int = 0, run: str = RUN) -> list[dict[str, Any]]:
    return [{"schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": step, "ts": first_ts + at,
             "layers": {}, "grad_norm": 1.0, "grad_norms": {"w": 1.0}, "lr": [0.1]}
            for at, step in enumerate(steps)]


def write(path: Path, records: list[dict[str, Any]], *, tail: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail, encoding="utf-8")
    return path


def only_row(folder: Path) -> dict[str, Any]:
    (row,) = tensorquill.list_runs(folder)
    return row


# --- CLI-13 -----------------------------------------------------------------------------------
#
# One run and rank in two files. In `one.jsonl` attempt B is seen only in the middle: A, then B,
# then C. `two.jsonl` holds B's later rows. Every record: the order is A, B, C (one file orders
# all three), C replaces B from step 25, and the run ends at C's step 34. From the files' first
# and last records only, B is not seen in `one.jsonl`, so the order went by `ts` (A, C, B), B was
# taken as the newest, and the row said 99.

A, B, C = 900, 500, 2


def mid_file_case(folder: Path, first: str = "one.jsonl", second: str = "two.jsonl") -> tuple[Path, Path]:
    one = write(folder / first, rows(A, range(0, 60), 1000.0) + rows(B, range(20, 30), 3000.0)
                + rows(C, range(25, 35), 1500.0))
    two = write(folder / second, rows(B, range(30, 100), 3010.0))
    return one, two


def test_cli13_an_attempt_seen_only_mid_file_orders_the_attempts_as_compare_does(tmp_path):
    folder = tmp_path / "out"
    mid_file_case(folder)
    assert cmp.open_run(RUN, folder)["last_step"] == 34
    assert only_row(folder)["last_step"] == 34


def test_cli13_the_listing_agrees_with_compare_whichever_file_sorts_first(tmp_path):
    for first, second in (("one.jsonl", "two.jsonl"), ("z-one.jsonl", "a-two.jsonl")):
        folder = tmp_path / first
        mid_file_case(folder, first, second)
        assert only_row(folder)["last_step"] == cmp.open_run(RUN, folder)["last_step"] == 34


def test_cli13_a_copy_a_torn_line_and_files_that_disagree_read_as_compare_reads_them(tmp_path):
    # A copy of a file is read once.
    copy = tmp_path / "copy"
    one, _ = mid_file_case(copy)
    (copy / "one.bak.jsonl").write_bytes(one.read_bytes())
    assert only_row(copy)["last_step"] == cmp.open_run(RUN, copy)["last_step"] == 34
    # A torn line in the middle of a file (a killed writer, then a restart) is left out.
    torn = tmp_path / "torn"
    write(torn / "one.jsonl", rows(A, range(0, 60), 1000.0), tail='{"run_id": "lane-x5", "step": 7')
    with (torn / "one.jsonl").open("a") as file:
        file.write("\n" + "".join(json.dumps(r) + "\n" for r in rows(B, range(20, 30), 3000.0) + rows(C, range(25, 35), 1500.0)))
    write(torn / "two.jsonl", rows(B, range(30, 100), 3010.0))
    assert only_row(torn)["last_step"] == cmp.open_run(RUN, torn)["last_step"] == 34
    # Files that disagree on the order (here only through the attempt in the middle): by number,
    # and the row says so, as compare does.
    disagree = tmp_path / "disagree"
    write(disagree / "one.jsonl", rows(900, range(0, 10), 1000.0) + rows(2, range(5, 8), 2000.0)
          + rows(900, range(10, 12), 2100.0))
    write(disagree / "two.jsonl", rows(2, range(15, 20), 3000.0) + rows(900, range(12, 13), 4000.0))
    opened = cmp.open_run(RUN, disagree)
    row = only_row(disagree)
    assert opened["read_from"]["attempt_order"] == "numeric"
    assert row["last_step"] == opened["last_step"] == 12
    assert "disagree" in json.dumps(row)


def test_cli13_the_end_cell_is_the_newest_attempts_file(tmp_path):
    # The first file by name is the old attempt's, never closed; the resume's file closed.
    folder = tmp_path / "out"
    write(folder / "a-old.jsonl", rows(1, range(0, 50), 1000.0))
    new = write(folder / "b-new.jsonl", rows(2, range(20, 40), 2000.0))
    assert write_end_marker(new, run_id=RUN, rank=0, attempt=2)
    row = only_row(folder)
    assert row["last_step"] == 39
    assert row["end"]["state"] == "closed" and row["end"]["closed_ranks"] == [0]

    # The other way round: the old attempt's file closed (a raised session), the newest did not.
    other = tmp_path / "other"
    old = write(other / "a-old.jsonl", rows(1, range(0, 50), 1000.0))
    write(other / "b-new.jsonl", rows(2, range(20, 40), 2000.0))
    assert write_end_marker(old, run_id=RUN, rank=0, attempt=1, exception=RuntimeError())
    end = only_row(other)["end"]
    assert end["state"] == "not_closed" and end["not_closed_ranks"] == [0] and "raised_ranks" not in end
    assert "b-new.jsonl" in end["detail"]

    # A copy beside the newest attempt's file has no marker of its own: the marker of the file
    # the newest attempt wrote still says the rank closed.
    copy = tmp_path / "copy"
    closed = write(copy / f"{RUN}.rank0.jsonl", rows(1, range(0, 30), 1000.0))
    assert write_end_marker(closed, run_id=RUN, rank=0, attempt=1)
    (copy / "x.bak.jsonl").write_bytes(closed.read_bytes())
    assert only_row(copy)["end"]["state"] == "closed"
