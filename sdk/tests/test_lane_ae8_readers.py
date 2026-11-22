"""Lane AE8: the readers' side of WAVE_AE_CONTRACT.md, sections 5 and 6.

Section 5, an input that holds several run IDs. `first_divergence` checks each input on its own,
over every row of the rank, and answers `no_answer` with `several_run_ids` for an input that is not
one run; `runs` and the MCP `list_runs` list a file whose rank holds several run IDs as one row in
the state `several_run_ids`, with no run state, last step or incident, whatever evidence file lies
beside it; `compare` names that file and its IDs when a run is not found. The author's test: a file
whose first and last rows say `r1` and whose middle says `r2`, through every reader, small (the last
megabyte, parsed) and over a megabyte (the scan of the part before it, in the sink's key order and
in another).

Section 6, rows not stored because their step was not newer (L30). A rank's entry in
`watchdog.json` carries `rows_not_newer` and `late_steps_confirmed`, each only when above zero
(`server/src/watchdog.rs` of lane AE1). The statuses here are written by hand in that shape.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import compare as compare_module
from tensorquill import divergence as dv
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill import runs as runs_module

CASES = json.loads((Path(__file__).parent / "fixtures" / "wave_ae" / "cases.json").read_text())["cases"]
SECTION_5 = [(name, body) for name, body in CASES.items() if body["section"] == 5]


def record(run: str | None, step: int, *, sink_order: bool = True, gn: float = 1.0) -> dict[str, Any]:
    """A step record. In the sink's order the four keys the scan reads lead the line
    (`runs._RECORD_HEAD`); otherwise `step` comes first and the line is parsed whole."""
    head: dict[str, Any] = {"schema_version": 1, **({"run_id": run} if run is not None else {}), "rank": 0, "attempt": 0}
    body = {"step": step, "ts": 1000.0 + step, "grad_norm": gn, "lr": [0.1],
            "layers": {"blk": {"mean": 0.0, "std": 1.0, "abs_max": 1.0, "zero_frac": 0.0}}, "grad_norms": {"w": gn + step}}
    return {**head, **body} if sink_order else {**body, **head}


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def call(where: Path, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    reply = mcp_server.call_tool(mcp_server.Source(where), 1, {"name": name, "arguments": arguments or {}})
    assert reply["result"]["isError"] is False, reply
    return json.loads(reply["result"]["content"][0]["text"])


def refused(where: Path, name: str, arguments: dict[str, Any] | None = None) -> str:
    reply = mcp_server.call_tool(mcp_server.Source(where), 1, {"name": name, "arguments": arguments or {}})
    assert reply["result"]["isError"] is True, reply
    return reply["result"]["content"][0]["text"]


# --- Section 5: first_divergence ------------------------------------------------------------------


@pytest.mark.parametrize("name,body", SECTION_5, ids=[name for name, _ in SECTION_5])
def test_each_case_of_section_5_gives_its_expected(name, body):
    """Fails before for the three new cases: the rows of two run IDs were read as one run."""
    result = dv.first_divergence(body["given"]["records_a"], body["given"]["records_b"])
    expected = body["expected"]
    assert result["answer"] == expected["answer"]
    assert result.get("several_run_ids") == expected["several_run_ids"]
    if expected["several_run_ids"] is None:
        assert "several_run_ids" not in result  # the key only when its condition holds
    else:
        for side, ids in expected["several_run_ids"].items():
            assert f"run {side.upper()}'s rows of rank 0 hold {len(ids)} run IDs" in result["reason"]
            assert ", ".join(repr(run) for run in ids) in result["reason"]
        assert len(expected["several_run_ids"]) == result["reason"].count("run IDs (")


def test_a_row_with_no_run_id_beside_named_rows_is_another_run():
    clean = [record("r1", s) for s in range(40)]
    mixed = [record("r1", s) for s in range(20)] + [record(None, s) for s in range(20, 39)] + [record("r1", 39)]
    result = dv.first_divergence(clean, mixed)
    assert result["answer"] == "no_answer" and result["several_run_ids"] == {"b": [None, "r1"]}
    assert "(no run_id, 'r1')" in result["reason"]
    # Rows that all name none are one run, as `check_one_run` reads them.
    assert "several_run_ids" not in dv.first_divergence(clean, [record(None, s) for s in range(40)])


def test_each_input_is_checked_over_every_row_of_the_chosen_rank_only():
    """A row of another rank, of another run ID, is not read for rank 0."""
    a = [record("r1", s) for s in range(40)]
    b = [record("r2", s) for s in range(40)] + [{**record("r9", 5), "rank": 1}]
    assert dv.first_divergence(a, b)["answer"] == "no_divergence"
    assert dv.first_divergence(a, b, rank=1)["answer"] == "no_answer"  # A has no rank 1: as before
    mixed = [record("r1", s) for s in range(40)] + [{**record("r9", 5), "rank": 1}, {**record("r8", 6), "rank": 1}]
    assert dv.first_divergence(mixed, [{**r, "rank": 1} for r in a], rank=1)["several_run_ids"] == {"a": ["r8", "r9"]}


def test_a_row_replaced_by_a_later_attempt_still_counts_for_the_check():
    """The check is over every row of the rank, before attempts are resolved."""
    # The earlier attempt is 0 and the later 1: an attempt of -1 is refused since wave AF (section 1).
    b = [{**record("r2", s), "attempt": 1} for s in range(40)] + [record("r9", 10)]
    assert dv.first_divergence([record("r1", s) for s in range(40)], b)["several_run_ids"] == {"b": ["r2", "r9"]}


# --- Section 5: the author's test through every reader ------------------------------------------

KINDS = ["small", "large_sink_order", "large_other_order"]


def mixed_rows(kind: str) -> list[dict[str, Any]]:
    """`r1` at the first and last rows and `r2` only in the middle. Over a megabyte, `r2` is only in
    the part before the last megabyte, so only the scan of that part can see it."""
    if kind == "small":
        return [record("r1", s) for s in range(15)] + [record("r2", s) for s in range(15, 30)] + \
               [record("r1", s) for s in range(30, 45)]
    order = kind == "large_sink_order"
    rows = [record("r1", s, sink_order=order) for s in range(300)]
    rows += [record("r2", s, sink_order=order) for s in range(300, 400)]
    rows += [record("r1", s, sink_order=order) for s in range(400, 11000)]
    return rows


@pytest.fixture(params=KINDS)
def mixed(request, tmp_path: Path) -> dict[str, Any]:
    folder = tmp_path / "out"
    folder.mkdir()
    rows = mixed_rows(request.param)
    path = write(folder / "r1.rank0.jsonl", rows)
    assert rows[0]["run_id"] == rows[-1]["run_id"] == "r1" and {r["run_id"] for r in rows} == {"r1", "r2"}
    if request.param != "small":
        size = path.stat().st_size
        assert size > 2 * runs_module._TAIL_BYTES
        with path.open("rb") as file:  # the last megabyte holds no r2 line
            file.seek(size - runs_module._TAIL_BYTES)
            assert b'"r2"' not in file.read()
    # An evidence file of r1's rows alone, written after the records: it is not read.
    evidence_path = path.with_suffix(".evidence.json")
    ev.write_evidence(ev.build_evidence([r for r in rows if r["run_id"] == "r1"], detectors="python"), evidence_path)
    later = path.stat().st_mtime + 5
    os.utime(evidence_path, (later, later))
    write(folder / "other.rank0.jsonl", [record("other", s) for s in range(45)])
    return {"kind": request.param, "folder": folder, "path": path, "rows": rows}


def test_runs_lists_the_file_as_several_run_ids_with_no_state_and_no_incident(mixed):
    """Fails before: the row was r1's, `current` and as healthy as its evidence file said."""
    rows = {row["run_id"]: row for row in runs_module.list_runs(mixed["folder"])}
    row = rows["r1"]
    assert row["evidence"]["state"] == "several_run_ids"
    assert row["evidence"]["run_ids"] == ["r1", "r2"]
    assert row["evidence"]["files"] == [{"path": str(mixed["path"]), "ranks": [0], "run_ids": ["r1", "r2"]}]
    assert row["run_state"] is None and row["incidents"] is None and row["first_incident"] is None
    assert row["kinds_checked"] is None and row["last_step"] is None and "freshness" not in row
    assert "path" not in row["evidence"]  # the evidence file beside it was not read
    text = runs_module.listing_text(list(rows.values()), mixed["folder"])
    assert "r1.rank0.jsonl holds, for rank 0, the records of more than one run (r1, r2)" in text
    assert rows["other"]["evidence"]["state"] == "missing"  # a file of one run ID is listed as before


def test_mcp_list_runs_carries_the_state_and_the_ids(mixed):
    listed = {row["run_id"]: row for row in call(mixed["path"], "list_runs")["runs"]}
    assert listed["r1"]["evidence"]["state"] == "several_run_ids" and listed["r1"]["evidence"]["run_ids"] == ["r1", "r2"]
    assert listed["r1"]["run_state"] is None and listed["r1"]["incidents"] is None


def test_compare_names_the_file_that_holds_a_run_not_found(mixed):
    """Fails before: "no run r2", and nothing said r2 was in r1's file."""
    with pytest.raises(runs_module.RunLookupError) as raised:
        tensorquill.compare_runs("r2", "other", source=mixed["folder"], paths=False)
    message = str(raised.value)
    assert "no run with the ID or name 'r2'" in message
    assert "r1.rank0.jsonl holds, for rank 0, the records of runs r1, r2" in message
    text = refused(mixed["path"], "compare_runs", {"runs": ["r2", "other"]})
    assert "r1.rank0.jsonl holds, for rank 0, the records of runs r1, r2" in text
    # The run the file is named for is refused by the detectors, as before.
    with pytest.raises(ValueError, match="more than one run"):
        tensorquill.compare_runs("r1", "other", source=mixed["folder"], paths=False)


def test_first_divergence_on_the_file_has_no_answer(mixed):
    """Fails before: the mixed rows were compared as one run."""
    other = [record("other", s) for s in range(len(mixed["rows"]))]
    result = dv.first_divergence(mixed["rows"], other)
    assert result["answer"] == "no_answer" and result["several_run_ids"] == {"a": ["r1", "r2"]}
    loaded = ev.load_records(str(mixed["path"]))
    assert dv.first_divergence(other, loaded)["several_run_ids"] == {"b": ["r1", "r2"]}


def test_the_readers_that_refused_already_still_refuse(mixed):
    """Passes on the base too: `get_series`, `list_incidents` and `build_evidence` read every row."""
    assert "more than one run" in refused(mixed["path"], "get_series", {"name": "w", "kind": "parameter"})
    assert "more than one run" in refused(mixed["path"], "list_incidents")
    with pytest.raises(ValueError, match="more than one run"):
        ev.build_evidence(str(mixed["path"]), detectors="python")


def test_a_split_rank_whose_restart_file_holds_another_run_in_the_middle(tmp_path):
    """A rank in two files, the second of a name of its own, read whole (`_rank_across_files`),
    not scanned: its run IDs are read there."""
    folder = tmp_path / "out"
    folder.mkdir()
    write(folder / "r1.rank0.jsonl", [record("r1", s) for s in range(20)])
    restart = [{**r, "attempt": 1} for r in [record("r1", s) for s in range(15, 25)] + [record("r2", s) for s in range(25, 30)]
               + [record("r1", s) for s in range(30, 40)]]
    write(folder / "restart.jsonl", restart)
    (row,) = runs_module.list_runs(folder)
    assert row["evidence"]["state"] == "several_run_ids"
    assert row["evidence"]["files"] == [{"path": str(folder / "restart.jsonl"), "ranks": [0], "run_ids": ["r1", "r2"]}]
    assert row["run_state"] is None and row["last_step"] is None


def test_one_rank_each_of_its_own_run_id_in_one_file_is_listed_as_before(tmp_path):
    """The page's condition is a rank that holds more than one run ID; this file's ranks hold one each."""
    folder = tmp_path / "out"
    folder.mkdir()
    write(folder / "mixed.jsonl", [record("r1", s) for s in range(10)] + [{**record("r2", s), "rank": 1} for s in range(10)])
    (row,) = runs_module.list_runs(folder)
    assert row["evidence"]["state"] != "several_run_ids"


def test_a_run_that_is_found_reads_nothing_more(tmp_path, monkeypatch):
    """The file scan for the IDs runs only for a run not found."""
    folder = tmp_path / "out"
    folder.mkdir()
    write(folder / "a.rank0.jsonl", [record("a", s) for s in range(45)])
    write(folder / "b.rank0.jsonl", [record("b", s) for s in range(45)])
    looked: list[str] = []
    monkeypatch.setattr(runs_module, "files_holding_run", lambda source, wanted: looked.append(wanted) or [])
    tensorquill.compare_runs("a", "b", source=folder, paths=False)
    assert looked == []
    with pytest.raises(runs_module.RunLookupError, match="no run with the ID or name 'c'"):
        tensorquill.compare_runs("a", "c", source=folder, paths=False)
    assert looked == ["c"]


# --- Section 6: rows not stored because their step was not newer (L30) --------------------------

ROWS_0, LATE_0 = 60, 12  # rank 0: the page's 60-row part
ROWS_1 = 7  # rank 1: rows not newer, none confirmed late


def page_sentence(rows: int, late: int) -> str:
    """WAVE_AE_CONTRACT.md, section 6, "What a reader says", with N and M put in."""
    text = (f"The server received {rows} row(s) whose step was not above a step it had already kept for their attempt,"
            f" and did not store them. A row sent again is counted again, so {rows} is not a number of steps lost. ")
    if late:
        text += (f"{late} distinct step(s) among them are confirmed to have arrived out of order inside one stored part,"
                 f" and are not stored. {late} is a lower bound: a late step that crosses a part boundary is not in it,"
                 f" so the same rows can give another {late} when the parts are cut elsewhere. ")
    else:
        text += "None is confirmed as a step that arrived out of order. "
    return text.rstrip() if late >= rows else text + "The other rows are not told apart from rows sent again."


def store(source: Path, run: str, entries: dict[int, dict[str, Any]] | None = None, ranks: int = 3) -> Path:
    """A run folder in a store at `source`, `ranks` ranks with current reports; `entries` adds keys
    to a rank's status entry (as lane AB5's test writes it)."""
    folder = source / "runs" / run
    rows: list[dict[str, Any]] = [{"rank": r, "attempt": 1, "last_step": 9} for r in range(ranks)]
    for entry in rows:
        parts = folder / f"rank={entry['rank']}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-00000000-00000009.parquet").write_bytes(b"")
        (parts.parent / "incidents.json").write_text(json.dumps({
            "schema_version": 1, "source": "tensorquill-server", "run_id": run, "rank": entry["rank"],
            "attempt": 1, "last_step": 9, "steps": 10, "incidents": [], "checked": list(ev.SERVER_KINDS),
            "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN},
        }))
        entry.update((entries or {}).get(entry["rank"], {}))
    keys = ("dropped_rows", "dropped_row_keys", "rows_not_newer", "late_steps_confirmed")
    (folder / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run, "summary": "active", "server_running": True,
        "written_at_ms": int(time.time() * 1000), "world_size": ranks, "missing_ranks": [], "incidents": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **e} for e in rows],
        "pauses": {}, "recommended_silent_seconds": None,
        # The carry names the counts too; a reader that summed it as well would count twice.
        "restart": [{"rank": e["rank"], **{k: e[k] for k in keys if k in e}} for e in rows],
    }))
    return folder


def late_entries() -> dict[int, dict[str, Any]]:
    return {0: {"rows_not_newer": ROWS_0, "late_steps_confirmed": LATE_0, "dropped_rows": 3, "dropped_row_keys": {"step": 3}},
            1: {"rows_not_newer": ROWS_1}}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    found = tmp_path / "store"
    store(found, "base")
    store(found, "after", late_entries())
    return found


def test_the_block_carries_both_counts_and_the_pages_sentence():
    status = {"ranks": [{"rank": 0, **late_entries()[0]}, {"rank": 1, **late_entries()[1]}, {"rank": 2}]}
    found = runs_module.rows_not_newer(status)
    assert found == {"rows": ROWS_0 + ROWS_1, "late_steps_confirmed": LATE_0,
                     "ranks": [{"rank": 0, "rows": ROWS_0, "late_steps_confirmed": LATE_0}, {"rank": 1, "rows": ROWS_1}],
                     "includes_replays": True, "note": page_sentence(ROWS_0 + ROWS_1, LATE_0)}
    one = runs_module.rows_not_newer({"ranks": [{"rank": 1, "rows_not_newer": ROWS_1}]})
    assert one["note"] == page_sentence(ROWS_1, 0) and "late_steps_confirmed" not in one
    assert "None is confirmed as a step that arrived out of order." in one["note"]


@pytest.mark.parametrize("entry", [
    {}, {"rows_not_newer": 0}, {"rows_not_newer": True}, {"rows_not_newer": "3"}, {"rows_not_newer": 2.0},
    {"rows_not_newer": -1}, {"late_steps_confirmed": 4}, {"rows_not_newer": None, "late_steps_confirmed": 4},
])
def test_a_status_without_a_count_above_zero_gives_no_block_and_never_raises(entry):
    assert runs_module.rows_not_newer({"ranks": [{"rank": 0, **entry}]}) is None
    for status in (None, {}, {"ranks": None}, {"ranks": ["x", {"rank": "0", "rows_not_newer": 3}]}):
        assert runs_module.rows_not_newer(status) is None


@pytest.mark.parametrize("late", [0, True, "5", 2.0, -3])
def test_a_late_count_that_is_not_a_count_above_zero_is_left_out(late):
    found = runs_module.rows_not_newer({"ranks": [{"rank": 0, "rows_not_newer": 9, "late_steps_confirmed": late}]})
    assert "late_steps_confirmed" not in found and found["note"] == page_sentence(9, 0)


def test_the_listing_row_and_its_text(source):
    """Fails before: no reader said these rows."""
    rows = {row["run_id"]: row for row in runs_module.list_runs(source)}
    after = rows["after"]
    assert after["rows_not_newer"]["rows"] == ROWS_0 + ROWS_1
    assert after["rows_not_newer"]["late_steps_confirmed"] == LATE_0
    assert after["dropped_rows"]["rows"] == 3  # not added to the rows dropped
    assert "rows_not_newer" not in rows["base"]
    text = runs_module.listing_text([after], source)
    assert f"after: {page_sentence(ROWS_0 + ROWS_1, LATE_0)} (rank 0: 60, 12 confirmed late, rank 1: 7)" in text


def test_compare_carries_the_block_and_its_line(source):
    result = tensorquill.compare_runs("base", "after", source=source)
    base, after = result["runs"]
    assert "rows_not_newer" not in base
    assert after["rows_not_newer"]["rows"] == ROWS_0 + ROWS_1 and after["rows_not_newer"]["late_steps_confirmed"] == LATE_0
    assert after["dropped_rows"]["rows"] == 3
    (line,) = [line for line in result["summary"].splitlines() if "B: The server received" in line]
    assert page_sentence(ROWS_0 + ROWS_1, LATE_0) in line
    one = tensorquill.compare_runs("base", "after", source=source, rank=1)["runs"][1]["rows_not_newer"]
    assert one == {"rows": ROWS_1, "ranks": [{"rank": 1, "rows": ROWS_1}], "includes_replays": True,
                   "note": page_sentence(ROWS_1, 0)}
    assert "rows_not_newer" not in tensorquill.compare_runs("base", "after", source=source, rank=2)["runs"][1]


def test_the_mcp_answers_carry_both_keys(source):
    where = source / "runs" / "after"
    status = call(where, "get_all_ranks_status")
    assert status["rows_not_newer"]["rows"] == ROWS_0 + ROWS_1 and status["rows_not_newer"]["late_steps_confirmed"] == LATE_0
    assert status["rows_not_newer"]["note"] == page_sentence(ROWS_0 + ROWS_1, LATE_0)
    assert status["dropped_rows"]["rows"] == 3
    per_rank = {entry["rank"]: entry for entry in status["per_rank"]}
    assert (per_rank[0]["rows_not_newer"], per_rank[0]["late_steps_confirmed"]) == (ROWS_0, LATE_0)
    assert per_rank[1]["rows_not_newer"] == ROWS_1 and "late_steps_confirmed" not in per_rank[1]
    assert "rows_not_newer" not in per_rank[2] and "late_steps_confirmed" not in per_rank[2]
    one = call(where, "get_all_ranks_status", {"rank": 0})["rank"]["rows_not_newer"]
    assert one == {"rows": ROWS_0, "late_steps_confirmed": LATE_0, "note": page_sentence(ROWS_0, LATE_0)}
    listed = call(where, "list_incidents", {"rank": 1})["rank"]["rows_not_newer"]
    assert listed == {"rows": ROWS_1, "note": page_sentence(ROWS_1, 0)}
    assert "rows_not_newer" not in call(where, "list_incidents", {"rank": 2})["rank"]
    rows = {row["run_id"]: row for row in call(where, "list_runs")["runs"]}
    assert rows["after"]["rows_not_newer"]["late_steps_confirmed"] == LATE_0 and "rows_not_newer" not in rows["base"]
    compared = call(where, "compare_runs", {"runs": ["base", "after"]})["runs"]
    assert compared[1]["rows_not_newer"]["rows"] == ROWS_0 + ROWS_1 and "rows_not_newer" not in compared[0]


def test_the_mcp_answers_bound_the_ranks(tmp_path):
    source = tmp_path / "store"
    many = mcp_server.PER_RANK_ROWS + 5
    store(source, "base")
    store(source, "after", {r: {"rows_not_newer": 2, "late_steps_confirmed": 1} for r in range(many)}, ranks=many)
    status = call(source / "runs" / "after", "get_all_ranks_status")
    assert status["rows_not_newer"]["rows"] == 2 * many and status["rows_not_newer"]["late_steps_confirmed"] == many
    assert len(status["rows_not_newer"]["ranks"]) == mcp_server.PER_RANK_ROWS and status["rows_not_newer"]["ranks_omitted"] == 5
    for answer in (call(source / "runs" / "after", "list_runs")["runs"],
                   call(source / "runs" / "after", "compare_runs", {"runs": ["base", "after"]})["runs"]):
        (found,) = [row["rows_not_newer"] for row in answer if "rows_not_newer" in row]
        assert found["ranks_omitted"] == 5


def test_a_status_without_the_keys_reads_as_the_same_run_with_them_taken_out(source):
    """A control: the two keys change nothing but their own key and line."""
    plain = source.parent / "plain"
    store(plain, "base")
    store(plain, "after", {0: {k: v for k, v in late_entries()[0].items() if k.startswith("dropped")}})
    a = json.loads(json.dumps(tensorquill.compare_runs("base", "after", source=source)).replace(str(source), "<S>"))
    b = json.loads(json.dumps(tensorquill.compare_runs("base", "after", source=plain)).replace(str(plain), "<S>"))
    a["runs"][1].pop("rows_not_newer")
    assert [line for line in a["summary"].splitlines() if "The server received" not in line] == b["summary"].splitlines()
    for result in (a, b):
        result.pop("written_at_ms"), result.pop("summary")
    assert a == b
    for tool, arguments in (("get_all_ranks_status", {}), ("get_all_ranks_status", {"rank": 0}),
                            ("list_incidents", {"rank": 0}), ("list_runs", {})):
        with_keys = json.dumps(call(source / "runs" / "after", tool, arguments)).replace(str(source), "<S>")
        without = json.dumps(call(plain / "runs" / "after", tool, arguments)).replace(str(plain), "<S>")
        with_keys = json.loads(with_keys)
        _drop_late(with_keys)
        assert _without_times(with_keys) == _without_times(json.loads(without)), tool


def _drop_late(value: Any) -> None:
    if isinstance(value, dict):
        for key in ("rows_not_newer", "late_steps_confirmed"):
            value.pop(key, None)
        for item in value.values():
            _drop_late(item)
    elif isinstance(value, list):
        for item in value:
            _drop_late(item)


def _without_times(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_times(v) for k, v in value.items() if k not in ("written_at_ms", "age_seconds", "seconds")}
    if isinstance(value, list):
        return [_without_times(v) for v in value]
    return value


def test_each_tool_description_says_what_each_number_is_and_is_not():
    tools = {tool["name"]: tool["description"] for tool in mcp_server.TOOLS}
    for name in ("list_incidents", "get_all_ranks_status", "list_runs", "compare_runs"):
        said = tools[name]
        assert "rows_not_newer" in said and "late_steps_confirmed" in said, name
        assert "a row sent again is counted again, so it is not a number of steps lost" in said, name
        assert "It is a lower bound: a late step that crosses a part boundary is not in it" in said, name
        assert "The other rows, if any, are not told apart from rows sent again." in said, name
        assert "Neither count is added to `dropped_rows`." in said, name
    assert "several_run_ids" in tools["list_runs"]
