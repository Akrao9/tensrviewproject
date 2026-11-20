"""Lane Z5 (wave SDK-22): the run listing and the MCP server, one test or more for each row it
fixes.

- L21, first part: an attempt that appears only in the middle of a rank file over a megabyte
  (the first record and the whole last megabyte are another attempt's) is seen by the listing.
  The listing scans the part before the last megabyte for the attempts each line names, without
  parsing the lines (`runs._attempts_in_file`), and reads the file whole only when an attempt
  newer than the last megabyte's is there.
  Fail before (on 89bef4e): `test_l21_an_attempt_seen_only_in_the_middle_of_a_large_file_is_seen`
  gave `last_step` 130 (the late row of the replaced attempt) where the file stands at 20, and the
  scan tests failed with `AttributeError: module 'tensorquill.runs' has no attribute
  '_attempts_in_file'`.
- L22: the rows the server dropped for a null key (`dropped_rows`, `dropped_row_keys` on a rank's
  entry of `watchdog.json`) are in the listing's row, its printed notes, and the MCP tools
  `list_runs` and `get_all_ranks_status`, said to be an upper bound. A rank that dropped none
  reads as before.
  Fail before (on 89bef4e): the row, MCP and description tests failed on behaviour (`KeyError:
  'dropped_rows'`, the printed note missing, the description without it); the six cases of
  `test_l22_a_count_that_is_not_a_count_is_left_out` failed only because `runs.dropped_rows` did
  not exist (`AttributeError`), not on behaviour, and the bounded-answer test was added after the
  first run. The controls (`..._reads_as_before`, `..._dropped_none_...`) pass on both.

The records and the store folders are built here (the status by hand, from the shape
`server/src/watchdog.rs` writes near line 1363), so nothing depends on another lane's test file.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import compare as cmp
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill import runs as runs_module

RUN = "lane-z5"
PAD = "x" * 20_000  # a row of about 20 KB, so a few dozen rows pass the listing's last megabyte


def row(step: int, *, rank: int = 0, attempt: int = 1, run: str = RUN, ts: float | None = None,
        pad: bool = False) -> dict[str, Any]:
    found: dict[str, Any] = {
        "schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": step,
        "ts": 1_000.0 + step if ts is None else ts, "layers": {}, "grad_norm": 1.0, "grad_norms": {"w": 1.0},
        "lr": [0.1],
    }
    if pad:
        found["pad"] = PAD
    return found


def write(path: Path, rows: list[dict[str, Any]], *, compact: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    separators = (",", ":") if compact else None  # `compact`: as the sink writes its records
    path.write_text("".join(json.dumps(r, separators=separators) + "\n" for r in rows), encoding="utf-8")
    return path


def manifest(folder: Path, run: str = RUN, **fields: Any) -> Path:
    path = folder / f"{run}.run.json"
    path.write_text(json.dumps({"schema_version": 1, "run_id": run, "project": "p", "name": run,
                                "created_at": "2026-11-20T10:00:00Z", **fields}), encoding="utf-8")
    return path


def only_row(folder: Path) -> dict[str, Any]:
    (found,) = tensorquill.list_runs(folder)
    return found


def whole_read(path: Path, rank: int = 0) -> int | None:
    """The rank's last step as every reader that reads the whole file gets it."""
    return runs_module._rank_across_files([path], rank, RUN)["last_step"]


# --- L21: an attempt seen only in the middle of a large file -----------------------------------


def hidden_attempt(path: Path, *, rank: int = 0, compact: bool = False) -> Path:
    """Attempt 1 writes steps 0 to 9; attempt 2 resumes at step 5 and writes to step 20, in the
    middle of the file only; then late rows of attempt 1 (steps 10 to 130, a zombie process that
    was not stopped) fill the whole last megabyte. Attempt 2 replaced attempt 1 from step 5, so
    the rank stands at step 20; the first record and the last megabyte both say attempt 1."""
    rows = [row(step, rank=rank, attempt=1, ts=2_000.0 + step) for step in range(10)]
    rows += [row(step, rank=rank, attempt=2, ts=3_000.0 + step) for step in range(5, 21)]
    rows += [row(step, rank=rank, attempt=1, ts=4_000.0 + step, pad=True) for step in range(10, 131)]
    write(path, rows, compact=compact)
    assert path.stat().st_size > 2 * runs_module._TAIL_BYTES
    # The last megabyte names attempt 1 alone: the reading before this lane missed attempt 2.
    lines, whole = runs_module._tail(path)
    assert not whole and {json.loads(line)["attempt"] for line in lines} == {1}
    return path


@pytest.mark.parametrize("compact", [False, True])
def test_l21_an_attempt_seen_only_in_the_middle_of_a_large_file_is_seen(tmp_path, compact):
    path = hidden_attempt(tmp_path / f"{RUN}.rank0.jsonl", compact=compact)
    manifest(tmp_path, world_size=1)
    kept = ev.resolve_attempts(ev.load_records(path, all_attempts=True))
    assert max(r["step"] for r in kept) == 20  # as `compare` and the detectors read the file
    assert whole_read(path) == 20
    found = only_row(tmp_path)
    assert found["last_step"] == 20  # was 130, the late row of the replaced attempt
    assert cmp.open_run(path)["last_step"] == 20


def test_l21_in_a_run_of_several_ranks_the_hidden_attempt_counts(tmp_path):
    hidden_attempt(tmp_path / f"{RUN}.rank0.jsonl")
    write(tmp_path / f"{RUN}.rank1.jsonl", [row(step, rank=1) for step in range(13)])
    manifest(tmp_path, world_size=2)
    found = only_row(tmp_path)
    assert found["ranks"] == [0, 1] and found["last_step"] == 20  # was 130


def test_l21_a_resume_whose_attempt_is_also_in_the_last_megabyte_is_not_read_whole(tmp_path, monkeypatch):
    # Attempt 1 then attempt 2, both before the last megabyte, and attempt 2 fills it: the newest
    # attempt is the last megabyte's, so its last line stands and no record is parsed whole.
    rows = [row(step, attempt=1, ts=2_000.0 + step, pad=True) for step in range(60)]
    rows += [row(step, attempt=2, ts=3_000.0 + step, pad=True) for step in range(30, 160)]
    path = write(tmp_path / f"{RUN}.rank0.jsonl", rows)
    manifest(tmp_path, world_size=1)
    assert path.stat().st_size > 3 * runs_module._TAIL_BYTES
    monkeypatch.setattr(runs_module, "_rank_across_files", lambda *a, **k: pytest.fail("the file was read whole"))
    assert only_row(tmp_path)["last_step"] == 159


def test_l21_a_large_file_of_one_attempt_is_not_parsed_whole(tmp_path, monkeypatch):
    path = write(tmp_path / f"{RUN}.rank0.jsonl", [row(step, pad=True) for step in range(200)])
    manifest(tmp_path, world_size=1)
    tail_lines = len(runs_module._tail(path)[0])
    parsed = []
    real = json.loads

    def counting(text, *args, **kwargs):
        found = real(text, *args, **kwargs)
        if isinstance(found, dict) and "step" in found:
            parsed.append(found["step"])
        return found

    monkeypatch.setattr(runs_module.json, "loads", counting)
    monkeypatch.setattr(runs_module, "_rank_across_files", lambda *a, **k: pytest.fail("the file was read whole"))
    assert only_row(tmp_path)["last_step"] == 199
    # The first record, and the last megabyte twice at most (the row and the count read it): never
    # the 200 records.
    assert len(parsed) <= 1 + 2 * tail_lines < 200


def full_parse_order(path: Path) -> dict[int, list[tuple[str, Any]]]:
    """Each rank's `(run, attempt)` keys in the order they first appear, as `compare` reads the
    file: every line that is a JSON object."""
    found: dict[int, list[tuple[str, Any]]] = {}
    for _, record in runs_module._file_lines(path):
        rank = runs_module._record_rank(record)
        if rank is None:
            continue
        key = (ev._run_of(record), record.get("attempt", 0))
        try:
            hash(key)
        except TypeError:
            continue
        if key not in found.setdefault(rank, []):
            found[rank].append(key)
    return found


def test_l21_the_scan_names_the_same_attempts_as_a_full_parse(tmp_path):
    rng = random.Random(5)
    lines: list[str] = []
    for at in range(400):
        rank, attempt = rng.choice([0, 1, 2]), rng.choice([1, 2, "a\"b", 7])
        record = row(at, rank=rank, attempt=attempt, run=rng.choice([RUN, "other", "es\\c\"apedé"]))
        shape = at % 9
        if shape == 1:  # keys in another order
            keys = list(record)
            rng.shuffle(keys)
            record = {k: record[k] for k in keys}
        elif shape == 2:
            del record["attempt"]  # written before attempts existed: attempt 0
        elif shape == 3:
            del record["rank"]  # rank 0
        elif shape == 4:
            record["run_id"] = None  # of no named run
        elif shape == 5:
            record["attempt"] = 1.5  # not an attempt the prefix reads: parsed whole
        lines.append(json.dumps(record, separators=(",", ":") if at % 2 else None, ensure_ascii=at % 3 == 0))
        if at % 50 == 7:
            lines.append("[1, 2, 3]")  # JSON, not an object
            lines.append("not json at all")
            lines.append("")
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    scanned = runs_module._attempts_in_file(path, path.stat().st_size)
    assert scanned == full_parse_order(path)


def test_l21_a_torn_line_may_add_an_attempt_to_the_scan_and_never_hides_one(tmp_path):
    # A killed process's half line, with the next process's first record written onto it: no
    # reader parses that line, and the prefix scan reads the half line's attempt. The scan may name
    # more attempts than a full parse (the file is then read whole: slower, never wrong), never
    # fewer, and in the same order.
    half = json.dumps(row(3, attempt=1))[:60]
    lines = [json.dumps(row(0, attempt=1)), half + json.dumps(row(5, attempt=9)), json.dumps(row(6, attempt=2))]
    path = tmp_path / "torn.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    scanned = runs_module._attempts_in_file(path, path.stat().st_size)[0]
    full = full_parse_order(path)[0]
    assert [k for k in scanned if k in full] == full and set(full) <= set(scanned)


RECORDED = Path.home() / "Downloads" / "runs"


@pytest.mark.skipif(not RECORDED.is_dir(), reason="the recorded GPU captures are not on this machine")
def test_l21_the_scan_agrees_with_a_full_parse_on_the_recorded_captures():
    files = sorted(RECORDED.glob("*/*.jsonl"))
    assert files
    for path in files:
        assert runs_module._attempts_in_file(path, path.stat().st_size) == full_parse_order(path), path.name


def test_l21_a_listing_again_scans_only_what_was_appended(tmp_path, monkeypatch):
    # Within one process (the MCP server lists runs on every call) a file that only grew is
    # scanned from where the last scan stopped, not from its start.
    path = hidden_attempt(tmp_path / f"{RUN}.rank0.jsonl")
    manifest(tmp_path, world_size=1)
    runs_module._SCANNED.clear()
    assert only_row(tmp_path)["last_step"] == 20
    first = dict(runs_module._SCANNED)
    assert first
    with path.open("a", encoding="utf-8") as file:
        for step in range(131, 200):
            file.write(json.dumps(row(step, attempt=1, ts=5_000.0 + step, pad=True)) + "\n")
    starts: list[int] = []
    real = runs_module._scan_lines

    def watched(file, start, before, found):
        starts.append(start)
        return real(file, start, before, found)

    monkeypatch.setattr(runs_module, "_scan_lines", watched)
    assert only_row(tmp_path)["last_step"] == whole_read(path) == 20
    assert starts and min(starts) > 0
    # A file replaced by another of the same size or more is scanned from its start again.
    starts.clear()
    hidden = path.read_bytes()
    path.unlink()
    write(path, [row(step, attempt=3, pad=True) for step in range(len(hidden) // 20_000 + 10)])
    found = only_row(tmp_path)
    assert 0 in starts and found["last_step"] == whole_read(path)


# --- L22: the rows the server dropped -----------------------------------------------------------

STORE_RUN = "z5store"


def store(tmp_path: Path, drops: dict[int, Any] | None = None, run: str = STORE_RUN) -> Path:
    """A run folder in a store, two ranks with current reports. `drops` puts a rank's
    `dropped_row_keys` (a dict) on its status entry, with `dropped_rows` its total, as
    `watchdog.rs` writes them; a value that is not a dict is written as the total alone."""
    folder = tmp_path / "store" / "runs" / run
    entries = [{"rank": 0, "attempt": 1, "last_step": 9}, {"rank": 1, "attempt": 1, "last_step": 9}]
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
        "written_at_ms": int(time.time() * 1000), "world_size": 2, "missing_ranks": [], "incidents": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **e} for e in entries],
        "pauses": {}, "recommended_silent_seconds": None,
        # The carry state names the counts too; a reader that summed it as well would count twice.
        "restart": [{"rank": e["rank"], "dropped_rows": e.get("dropped_row_keys", {})} for e in entries],
    }))
    return folder


def test_l22_the_listing_row_has_the_dropped_rows_by_rank_and_key(tmp_path):
    store(tmp_path, {0: {"step": 2, "run_id": 1}, 1: {"step": 4}})
    found = only_row(tmp_path / "store")
    dropped = found["dropped_rows"]
    assert dropped["rows"] == 7
    assert dropped["by_key"] == {"run_id": 1, "step": 6}
    assert dropped["ranks"] == [{"rank": 0, "rows": 3, "by_key": {"run_id": 1, "step": 2}},
                                {"rank": 1, "rows": 4, "by_key": {"step": 4}}]
    assert dropped["upper_bound"] is True and "sent again" in dropped["note"]


def test_l22_the_printed_listing_says_the_count_is_an_upper_bound(tmp_path):
    store(tmp_path, {1: {"step": 4}})
    rows = tensorquill.list_runs(tmp_path / "store")
    text = runs_module.listing_text(rows, tmp_path / "store")
    (line,) = [line for line in text.splitlines() if "dropped" in line]
    assert line.startswith(f"{STORE_RUN}: ") and "at most 4 rows" in line and "rank 1: 4" in line
    assert "upper bound" in line and "sent again" in line


def test_l22_a_run_that_dropped_no_row_reads_as_before(tmp_path):
    clean = only_row(store(tmp_path / "a").parent.parent)
    assert "dropped_rows" not in clean
    text = runs_module.listing_text([clean], tmp_path)
    assert "dropped" not in text
    # One rank's drops change the row by that key alone.
    some = only_row(store(tmp_path / "b", {1: {"step": 4}}).parent.parent)
    assert {k: v for k, v in some.items() if k not in ("dropped_rows", "path")} == \
        {k: v for k, v in clean.items() if k != "path"}


@pytest.mark.parametrize("bad", ["3", -2, True, 0, None, 2.5])
def test_l22_a_count_that_is_not_a_count_is_left_out(tmp_path, bad):
    folder = store(tmp_path, {0: bad} if bad is not None else None)
    status = json.loads((folder / "watchdog.json").read_text())
    status["ranks"][1]["dropped_row_keys"] = {"step": bad, "x": "y"} if bad is not None else []
    (folder / "watchdog.json").write_text(json.dumps(status))
    assert "dropped_rows" not in only_row(tmp_path / "store")
    assert runs_module.dropped_rows(status) is None
    assert runs_module.dropped_rows({"ranks": "no"}) is None and runs_module.dropped_rows(None) is None


def test_l22_a_total_with_no_keys_is_still_counted(tmp_path):
    store(tmp_path, {0: 5})
    dropped = only_row(tmp_path / "store")["dropped_rows"]
    assert dropped["rows"] == 5 and dropped["ranks"] == [{"rank": 0, "rows": 5}] and "by_key" not in dropped


def test_l22_the_mcp_list_runs_row_has_the_count(tmp_path):
    store(tmp_path, {0: {"step": 2}})
    answer = mcp_server.list_runs(mcp_server.Source(tmp_path / "store" / "runs" / STORE_RUN), {})
    (listed,) = answer["runs"]
    assert listed["dropped_rows"]["rows"] == 2 and listed["dropped_rows"]["upper_bound"] is True


def test_l22_the_mcp_all_ranks_status_has_the_count_for_the_run_and_each_rank(tmp_path):
    folder = store(tmp_path, {0: {"step": 2, "run_id": 1}})
    source = mcp_server.Source(folder)
    answer = mcp_server.get_all_ranks_status(source, {})
    assert answer["dropped_rows"]["rows"] == 3 and "sent again" in answer["dropped_rows"]["note"]
    per_rank = {entry["rank"]: entry for entry in answer["per_rank"]}
    assert per_rank[0]["dropped_rows"] == 3 and "dropped_rows" not in per_rank[1]
    one = mcp_server.get_all_ranks_status(source, {"rank": 0})
    assert one["dropped_rows"]["rows"] == 3
    assert one["rank"]["dropped_rows"] == {"rows": 3, "by_key": {"run_id": 1, "step": 2}}
    assert "dropped_rows" not in mcp_server.get_all_ranks_status(source, {"rank": 1})["rank"]


def test_l22_the_mcp_list_incidents_of_one_rank_has_its_count(tmp_path):
    folder = store(tmp_path, {1: {"step": 4}})
    answer = mcp_server.list_incidents(mcp_server.Source(folder), {"rank": 1})
    assert answer["rank"]["dropped_rows"] == {"rows": 4, "by_key": {"step": 4}}
    assert "dropped_rows" not in mcp_server.list_incidents(mcp_server.Source(folder), {"rank": 0})["rank"]


def test_l22_the_mcp_answers_of_a_run_that_dropped_none_are_as_before(tmp_path):
    folder = store(tmp_path)
    source = mcp_server.Source(folder)
    status = mcp_server.get_all_ranks_status(source, {})
    assert "dropped_rows" not in status and all("dropped_rows" not in e for e in status["per_rank"])
    assert "dropped_rows" not in mcp_server.list_incidents(source, {"rank": 0})["rank"]
    assert "dropped_rows" not in mcp_server.list_runs(source, {})["runs"][0]


def test_l22_a_large_run_whose_every_rank_dropped_rows_has_a_bounded_answer(tmp_path):
    # A null key usually hits every rank: the answers list the ranks up to `PER_RANK_ROWS` and
    # count the rest, through the server's own path (`call_tool`, which cuts what does not fit).
    many = mcp_server.PER_RANK_ROWS + 36
    folder = tmp_path / "store" / "runs" / STORE_RUN
    folder.mkdir(parents=True)
    (folder / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "run_id": STORE_RUN, "summary": "active", "written_at_ms": int(time.time() * 1000),
        "world_size": many, "missing_ranks": [], "incidents": [], "pauses": {},
        "ranks": [{"rank": r, "state": "active", "stream": "open", "attempt": 1, "last_step": 9,
                   "dropped_rows": 2, "dropped_row_keys": {"step": 2}} for r in range(many)],
    }))
    source = mcp_server.Source(folder)
    for tool, arguments in (("get_all_ranks_status", {}), ("list_runs", {})):
        reply = mcp_server.call_tool(source, 1, {"name": tool, "arguments": arguments})
        assert reply["result"]["isError"] is False, reply
        answer = json.loads(reply["result"]["content"][0]["text"])
        assert "truncated" not in answer, answer.get("truncated")
        dropped = answer["dropped_rows"] if tool == "get_all_ranks_status" else answer["runs"][0]["dropped_rows"]
        assert dropped["rows"] == 2 * many and dropped["by_key"] == {"step": 2 * many}
        assert len(dropped["ranks"]) == mcp_server.PER_RANK_ROWS and dropped["ranks_omitted"] == 36
    one = mcp_server.get_all_ranks_status(source, {"rank": many - 1})
    assert one["rank"]["dropped_rows"] == {"rows": 2, "by_key": {"step": 2}}


def test_l22_the_tool_descriptions_say_the_count_is_an_upper_bound():
    tools = {tool["name"]: tool["description"] for tool in mcp_server.TOOLS}
    for name in ("list_runs", "get_all_ranks_status"):
        assert "dropped_rows" in tools[name] and "upper bound" in tools[name], name
