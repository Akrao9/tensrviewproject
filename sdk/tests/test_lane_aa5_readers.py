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


def test_l21_a_split_ranks_file_of_its_own_name_is_not_scanned(tmp_path, monkeypatch):
    """Fail before (on b2b7a95): every file over a megabyte was scanned, a split rank's file of a
    name of its own too, whose lines its count does not read (it is read whole for the attempts
    anyway). A split rank's file named for a rank is scanned: its count reads it."""
    named = big_rank_file(tmp_path / f"{RUN}.rank0.jsonl")
    other = big_rank_file(tmp_path / "earlier.jsonl")  # rank 0 of the same run, a name of its own
    real = runs_module._scan_file
    scanned: list[str] = []

    def watched(path, before):
        scanned.append(Path(path).name)
        return real(path, before)

    monkeypatch.setattr(runs_module, "_scan_file", watched)
    runs_module._SCANNED.clear()
    found = only_row(tmp_path)
    assert other.name not in scanned and named.name in scanned
    assert found["ranks"] == [0]


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


# --- 2. L22: the rows a store run's server dropped, in `compare` ---------------------------------


def store(source: Path, run: str, drops: dict[int, Any] | None = None, ranks: int = 2) -> Path:
    """A run folder in a store at `source`, `ranks` ranks with current reports. `drops` puts a
    rank's `dropped_row_keys` (a dict) on its status entry, with `dropped_rows` its total, as
    `watchdog.rs` writes them; a value that is not a dict is written as the total alone."""
    folder = source / "runs" / run
    entries: list[dict[str, Any]] = [{"rank": r, "attempt": 1, "last_step": 9} for r in range(ranks)]
    for entry in entries:
        parts = folder / f"rank={entry['rank']}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-00000000-00000009.parquet").write_bytes(b"")
        (parts.parent / "incidents.json").write_text(json.dumps({
            "schema_version": 1, "source": "tensorquill-server", "run_id": run, "rank": entry["rank"],
            "attempt": 1, "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
            "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN},
        }))
        keys = (drops or {}).get(entry["rank"])
        if isinstance(keys, dict):
            entry["dropped_rows"] = sum(keys.values())
            entry["dropped_row_keys"] = keys
        elif keys is not None:
            entry["dropped_rows"] = keys
    (folder / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run, "summary": "active", "server_running": True,
        "written_at_ms": int(time.time() * 1000), "world_size": ranks, "missing_ranks": [], "incidents": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **e} for e in entries],
        "pauses": {}, "recommended_silent_seconds": None,
        # The carry state names the counts too; a reader that summed it as well would count twice.
        "restart": [{"rank": e["rank"], "dropped_rows": e.get("dropped_row_keys", {})} for e in entries],
    }))
    return folder


def without_time(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "written_at_ms"}


def test_l22_compare_shows_the_rows_a_store_run_dropped_as_an_upper_bound(tmp_path):
    """Fail before: no `dropped_rows` on the run's entry and no line in the summary."""
    source = tmp_path / "store"
    store(source, "base")
    store(source, "after", {0: {"step": 2, "run_id": 1}, 1: {"step": 4}})
    result = tensorquill.compare_runs("base", "after", source=source)
    base, after = result["runs"]
    assert "dropped_rows" not in base
    dropped = after["dropped_rows"]
    assert (dropped["rows"], dropped["by_key"], dropped["upper_bound"]) == (7, {"run_id": 1, "step": 6}, True)
    assert dropped["ranks"] == [{"rank": 0, "rows": 3, "by_key": {"run_id": 1, "step": 2}},
                                {"rank": 1, "rows": 4, "by_key": {"step": 4}}]
    assert "sent again" in dropped["note"]
    (line,) = [line for line in result["summary"].splitlines() if "dropped" in line]
    assert line.lstrip().startswith("B: ") and "at most 7 rows" in line and "upper bound" in line


def test_l22_compare_of_runs_that_dropped_nothing_reads_as_before(tmp_path):
    """A control (passes before and after), and fails before through its second half: a run
    that dropped no row has no key and no line, and a run that did differs by the key and the
    line alone."""
    source = tmp_path / "store"
    for run in ("base", "after"):
        store(source, run)
    clean = tensorquill.compare_runs("base", "after", source=source)
    assert all("dropped_rows" not in run for run in clean["runs"]) and "dropped" not in clean["summary"]
    status = source / "runs" / "after" / "watchdog.json"
    written = json.loads(status.read_text())
    written["ranks"][1].update({"dropped_rows": 4, "dropped_row_keys": {"step": 4}})
    status.write_text(json.dumps(written))
    some = tensorquill.compare_runs("base", "after", source=source)
    assert some["runs"][1]["dropped_rows"]["rows"] == 4
    some["runs"][1].pop("dropped_rows")
    assert [line for line in some["summary"].splitlines() if "dropped" not in line] == clean["summary"].splitlines()
    assert {**without_time(some), "summary": None} == {**without_time(clean), "summary": None}


def test_l22_compare_of_one_rank_shows_that_ranks_rows_alone(tmp_path):
    """Fail before: no `dropped_rows`. With `rank=`, the entry is that rank's; a rank that
    dropped none has no key though another rank of its run did."""
    source = tmp_path / "store"
    store(source, "base")
    store(source, "after", {1: {"step": 4}, 0: {"run_id": 2}})
    one = tensorquill.compare_runs("base", "after", source=source, rank=1)["runs"][1]["dropped_rows"]
    assert (one["rows"], one["by_key"], one["ranks"]) == (4, {"step": 4}, [{"rank": 1, "rows": 4, "by_key": {"step": 4}}])
    store(source, "other", {1: {"step": 4}})
    assert "dropped_rows" not in tensorquill.compare_runs("base", "other", source=source, rank=0)["runs"][1]


def test_l22_compare_of_one_rank_shows_the_rows_of_the_rank_read(tmp_path):
    """Fail before (on a5acac0): no `dropped_rows`. A run of one rank is read on the rank it has
    whatever `rank` asks (`_open_store`), so its count is that rank's, not the asked rank's."""
    source = tmp_path / "store"
    store(source, "base")
    store(source, "lone", {0: {"step": 3}}, ranks=1)
    entry = tensorquill.compare_runs("base", "lone", source=source, rank=1)["runs"][1]
    assert entry["read_from"]["rank"] == 0
    assert (entry["dropped_rows"]["rows"], entry["dropped_rows"]["ranks"]) == (3, [{"rank": 0, "rows": 3, "by_key": {"step": 3}}])


def test_l22_a_local_run_has_no_dropped_rows_in_compare(tmp_path):
    """A control (passes before and after): local files hold every record; only a store's
    server drops rows."""
    for run in ("a", "b"):
        write(tmp_path / f"{run}.rank0.jsonl", [row(step, run=run) for step in range(5)])
    result = tensorquill.compare_runs("a", "b", source=tmp_path)
    assert all("dropped_rows" not in run for run in result["runs"]) and "dropped" not in result["summary"]


def test_l22_the_mcp_compare_runs_answer_has_the_count_bounded(tmp_path):
    """Fail before: no `dropped_rows` in the answer, and the tool's description did not name it.
    A null key usually hits every rank: the ranks are listed up to `PER_RANK_ROWS`, the rest
    counted, through the server's own path (`call_tool`)."""
    source = tmp_path / "store"
    many = mcp_server.PER_RANK_ROWS + 6
    store(source, "base")
    store(source, "after", {r: {"step": 2} for r in range(many)}, ranks=many)
    reply = mcp_server.call_tool(mcp_server.Source(source / "runs" / "base"), 1,
                                 {"name": "compare_runs", "arguments": {"runs": ["base", "after"]}})
    assert reply["result"]["isError"] is False, reply
    answer = json.loads(reply["result"]["content"][0]["text"])
    assert "truncated" not in answer, answer.get("truncated")
    base, after = answer["runs"]
    assert "dropped_rows" not in base
    dropped = after["dropped_rows"]
    assert (dropped["rows"], dropped["by_key"], dropped["upper_bound"]) == (2 * many, {"step": 2 * many}, True)
    assert len(dropped["ranks"]) == mcp_server.PER_RANK_ROWS and dropped["ranks_omitted"] == 6
    (described,) = [tool["description"] for tool in mcp_server.TOOLS if tool["name"] == "compare_runs"]
    assert "dropped_rows" in described and "upper bound" in described


# --- 3. From lane Z2: the rank one rank's evidence of a shared file judged ------------------------


def shared(folder: Path, name: str = "both.jsonl", ranks: tuple[int, ...] = (0, 1, 2), spike_on: int | None = None) -> Path:
    """One file of several ranks, their rows interleaved; `spike_on` gives that rank a grad spike."""
    rows = []
    for step in range(60):
        for rank in ranks:
            found = row(step, rank=rank)
            if rank == spike_on and step == 40:
                found["grad_norm"] = 1e6
            rows.append(found)
    return write(folder / name, rows)


def test_z2_the_row_names_the_rank_the_evidence_of_a_shared_file_judged(tmp_path):
    """Fail before: `evidence.rank` None and "without saying which" in the detail, though the
    evidence says `rank: 1` and `ranks_in_source: [0, 1, 2]`."""
    path = shared(tmp_path, spike_on=1)
    judged(path, rank=1)
    found = only_row(tmp_path)
    cell = found["evidence"]
    assert (cell["state"], cell["rank"], cell["ranks_in_source"], cell["judged_ranks"]) == ("ranks_unknown", 1, [0, 1, 2], 1)
    assert "its evidence file is rank 1's of the ranks 0, 1, 2 the file held when it was written" in cell["detail"]
    assert "without saying which" not in cell["detail"]
    # The other ranks are still not judged, and the incidents are rank 1's.
    assert found["run_state"] == "unhealthy" and found["incidents"] == {"anomaly": 1}
    assert "rank 1's" in cell["detail"] and "the other ranks are not judged" in cell["detail"]
    # Rank 0's evidence, with no incident: unknown, never healthy.
    judged(path, rank=0)
    found = only_row(tmp_path)
    assert (found["evidence"]["rank"], found["run_state"]) == (0, "unknown")
    # The same for a file named for a rank, whose ranks the listing reads whole.
    named = shared(tmp_path / "named", f"{RUN}.rank0.jsonl", spike_on=2)
    judged(named, rank=2)
    found = only_row(tmp_path / "named")
    assert (found["rank_count"], found["evidence"]["rank"], found["evidence"]["ranks_in_source"]) == (3, 2, [0, 1, 2])
    assert found["incidents"] == {"anomaly": 1}


def test_z2_evidence_that_names_its_rank_without_the_ranks_in_source_still_names_it(tmp_path):
    """Fail before: `evidence.rank` None. Evidence written when the file held its rank alone (it
    has grown since) names the rank and no source ranks."""
    path = shared(tmp_path)
    evidence = ev.build_evidence(path, rank=2)
    evidence.pop("ranks_in_source")
    ev.write_evidence(evidence, path.with_suffix(".evidence.json"))
    later = path.stat().st_mtime + 1
    os.utime(path.with_suffix(".evidence.json"), (later, later))
    cell = only_row(tmp_path)["evidence"]
    assert cell["rank"] == 2 and "ranks_in_source" not in cell
    assert "its evidence file is rank 2's: the row's incidents are rank 2's" in cell["detail"]


@pytest.mark.parametrize("named", ["missing", None, True, "1", 7])
def test_z2_evidence_that_names_no_rank_of_the_file_is_read_as_before(tmp_path, named):
    """A control (passes before and after): evidence with no rank, a rank that is not a number,
    or a rank the file is read not to hold (a file named for a rank: every line is read), is
    given to no rank, as before."""
    path = shared(tmp_path, f"{RUN}.rank0.jsonl")
    evidence = ev.build_evidence(path, rank=1)
    if named == "missing":
        evidence.pop("rank")
    else:
        evidence["rank"] = named
    ev.write_evidence(evidence, path.with_suffix(".evidence.json"))
    later = path.stat().st_mtime + 1
    os.utime(path.with_suffix(".evidence.json"), (later, later))
    cell = only_row(tmp_path)["evidence"]
    assert cell["rank"] is None and "ranks_in_source" not in cell and "without saying which" in cell["detail"]
