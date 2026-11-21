"""Lane AB5 (wave AB): the readers' follow-ups of wave AA.

1. Refused batches. Since wave AA the server writes `refused_batches` and `refused_batch_keys` on
   a rank's entry of `watchdog.json` for a batch it refused whole because a key column was absent
   or of another type (RS10). They are shown wherever dropped rows are (`runs.dropped_rows` and the
   listing's note, `compare`, the MCP answers), under their own key and in their own words, and
   are never added to the rows dropped. One fixture holds a refused batch, rows dropped for a null
   key and rows dropped for an unsupported `schema_version`, with counts whose sums appear nowhere,
   so no two of them can be combined by accident. A run with none of them reads as before.
2. The MCP wording: no tool description says every dropped row had a null key; the descriptions
   are pinned by an invariant over what each tool answers, so they cannot go stale silently.
3. The store path of `compare`, end to end against a real `tensorquill-server` built in this
   tree: rows sent over Flight (a null-key row, a row of schema version 7, a batch with no `step`
   column and one with a text `attempt` column), the server's own `watchdog.json` and reports,
   then `runs`, `compare` (from the reports alone, and with a rank's parts judged here), the
   command line, the MCP `compare_runs`, and `tensorquill-server detect --run` over the stored rows.

The hand-written statuses follow `server/src/watchdog.rs` (near line 1391), and the end-to-end test
checks that the real server writes the same keys, so the fixture cannot drift from the server.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import tensorquill
from tensorquill import evidence as ev
from tensorquill import mcp_server
from tensorquill import runs as runs_module

# Counts chosen so that no sum of two or three of them is any one of them, or any other count here.
NULL_ROWS = 3  # rank 0, rows dropped for a null `step`
VERSION_ROWS = 7  # rank 0, rows dropped for an unsupported schema version
REFUSED_0 = 2  # rank 0, batches refused for a missing `step` column
REFUSED_1 = 11  # rank 1, batches refused for a mistyped `attempt` column; rank 1 dropped no row
SUMS = {NULL_ROWS + VERSION_ROWS + REFUSED_0, NULL_ROWS + REFUSED_0, VERSION_ROWS + REFUSED_0,
        NULL_ROWS + VERSION_ROWS + REFUSED_0 + REFUSED_1, REFUSED_0 + REFUSED_1 + NULL_ROWS + VERSION_ROWS,
        NULL_ROWS + VERSION_ROWS + REFUSED_1, REFUSED_1 + NULL_ROWS, REFUSED_1 + VERSION_ROWS}
assert not SUMS & {NULL_ROWS, VERSION_ROWS, REFUSED_0, REFUSED_1, NULL_ROWS + VERSION_ROWS, REFUSED_0 + REFUSED_1}

UNSUPPORTED = runs_module.UNSUPPORTED_VERSION_KEY


def store(source: Path, run: str, entries: dict[int, dict[str, Any]] | None = None, ranks: int = 3) -> Path:
    """A run folder in a store at `source`, `ranks` ranks with current reports. `entries` adds keys
    to a rank's status entry, as `watchdog.rs` writes them."""
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
    (folder / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run, "summary": "active", "server_running": True,
        "written_at_ms": int(time.time() * 1000), "world_size": ranks, "missing_ranks": [], "incidents": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **e} for e in rows],
        "pauses": {}, "recommended_silent_seconds": None,
        # The carry state names the counts too; a reader that summed it as well would count twice.
        "restart": [{"rank": e["rank"], **{k: e[k] for k in ("dropped_row_keys", "refused_batches") if k in e}}
                    for e in rows],
    }))
    return folder


def three_reasons() -> dict[int, dict[str, Any]]:
    """Rank 0: rows dropped for a null key and for their version, and batches refused; rank 1:
    batches refused only; rank 2: nothing."""
    return {
        0: {"dropped_rows": NULL_ROWS + VERSION_ROWS, "dropped_row_keys": {"step": NULL_ROWS, UNSUPPORTED: VERSION_ROWS},
            "refused_batches": REFUSED_0, "refused_batch_keys": {"step": REFUSED_0}},
        1: {"refused_batches": REFUSED_1, "refused_batch_keys": {"attempt": REFUSED_1}},
    }


def numbers(value: Any) -> set[int]:
    """Every whole number in `value`, and in the text of its strings."""
    text = json.dumps(value, default=str)
    return {int(found.replace(",", "")) for found in re.findall(r"(?<![\w.])\d[\d,]*(?![\w.])", text)}


def counts(value: Any) -> list[Any]:
    """Every value under a `dropped_rows` or `refused_batches` key in `value`, at any depth."""
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += [item] if key in ("dropped_rows", "refused_batches") else counts(item)
    elif isinstance(value, list):
        for item in value:
            found += counts(item)
    return found


def call(where: Path, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    reply = mcp_server.call_tool(mcp_server.Source(where), 1, {"name": name, "arguments": arguments or {}})
    assert reply["result"]["isError"] is False, reply
    return json.loads(reply["result"]["content"][0]["text"])


@pytest.fixture
def source(tmp_path: Path) -> Path:
    found = tmp_path / "store"
    store(found, "base")
    store(found, "after", three_reasons())
    return found


# --- 1. Refused batches, apart from dropped rows ----------------------------------------------------


def test_the_readers_keep_refused_batches_apart_from_both_kinds_of_dropped_row(source):
    """Fail before: no reader had `refused_batches`."""
    status = ev.read_watchdog(source / "runs" / "after" / "watchdog.json")
    dropped, refused = runs_module.dropped_rows(status), runs_module.refused_batches(status)
    assert dropped == {
        "rows": NULL_ROWS + VERSION_ROWS, "by_key": {"step": NULL_ROWS, UNSUPPORTED: VERSION_ROWS},
        "ranks": [{"rank": 0, "rows": NULL_ROWS + VERSION_ROWS, "by_key": {"step": NULL_ROWS, UNSUPPORTED: VERSION_ROWS}}],
        "upper_bound": True, "note": runs_module.DROPPED_ROWS_NOTE_WITH_VERSIONS}
    assert refused == {
        "batches": REFUSED_0 + REFUSED_1, "by_key": {"attempt": REFUSED_1, "step": REFUSED_0},
        "ranks": [{"rank": 0, "batches": REFUSED_0, "by_key": {"step": REFUSED_0}},
                  {"rank": 1, "batches": REFUSED_1, "by_key": {"attempt": REFUSED_1}}],
        "note": runs_module.REFUSED_BATCHES_NOTE}
    assert "upper_bound" not in refused and "batches, not rows" in refused["note"]
    assert not numbers([dropped, refused]) & SUMS


def test_the_listing_row_and_its_text_say_each_count_in_its_own_words(source):
    rows = {row["run_id"]: row for row in runs_module.list_runs(source, None)}
    assert "refused_batches" not in rows["base"] and "dropped_rows" not in rows["base"]
    after = rows["after"]
    assert after["dropped_rows"]["rows"] == NULL_ROWS + VERSION_ROWS
    assert after["refused_batches"]["batches"] == REFUSED_0 + REFUSED_1
    text = runs_module.listing_text(list(rows.values()), source, None)
    (dropped_line,) = [line for line in text.splitlines() if "dropped at most" in line]
    (refused_line,) = [line for line in text.splitlines() if "refused" in line]
    assert dropped_line.startswith("after: ") and refused_line.startswith("after: ")
    assert f"at most {NULL_ROWS + VERSION_ROWS} rows with a null key or an unsupported schema version" in dropped_line
    assert "refused" not in dropped_line and "batch" not in dropped_line.split(";")[0]
    assert (f"refused {REFUSED_0 + REFUSED_1} batches whole because a key column was missing or of another type"
            in refused_line)
    assert f"rank 0: {REFUSED_0}, rank 1: {REFUSED_1}; by key column: attempt {REFUSED_1}, step {REFUSED_0}" in refused_line
    assert "not in any count of rows dropped" in refused_line and "upper bound" not in refused_line
    assert not numbers([dropped_line, refused_line]) & SUMS


def test_compare_shows_refused_batches_beside_the_dropped_rows(source):
    result = tensorquill.compare_runs("base", "after", source=source)
    base, after = result["runs"]
    assert "refused_batches" not in base and "dropped_rows" not in base
    assert after["refused_batches"]["batches"] == REFUSED_0 + REFUSED_1
    assert after["dropped_rows"]["rows"] == NULL_ROWS + VERSION_ROWS
    lines = [line for line in result["summary"].splitlines() if "B: the server" in line]
    assert len(lines) == 2 and "dropped at most 10 rows" in lines[0] and "refused 13 batches" in lines[1]
    assert not numbers([after["dropped_rows"], after["refused_batches"], lines]) & SUMS


@pytest.mark.parametrize("rank, rows, batches", [(0, NULL_ROWS + VERSION_ROWS, REFUSED_0), (1, None, REFUSED_1),
                                                 (2, None, None)])
def test_compare_of_one_rank_shows_that_ranks_counts_alone(source, rank, rows, batches):
    after = tensorquill.compare_runs("base", "after", source=source, rank=rank)["runs"][1]
    assert (after.get("dropped_rows") or {}).get("rows") == rows
    assert (after.get("refused_batches") or {}).get("batches") == batches
    if batches is not None:
        assert [item["rank"] for item in after["refused_batches"]["ranks"]] == [rank]


def test_one_ranks_dropped_rows_note_names_that_ranks_own_reasons(tmp_path):
    """Fail before: with `rank=`, the note was the whole run's, so a rank whose rows were dropped
    for a null key alone was said to have dropped some for their schema version too."""
    source = tmp_path / "store"
    store(source, "base")
    store(source, "after", {0: {"dropped_rows": 2, "dropped_row_keys": {UNSUPPORTED: 2}},
                            1: {"dropped_rows": 4, "dropped_row_keys": {"step": 4}}})
    one = tensorquill.compare_runs("base", "after", source=source, rank=1)["runs"][1]["dropped_rows"]
    assert one["note"] == runs_module.DROPPED_ROWS_NOTE
    both = tensorquill.compare_runs("base", "after", source=source, rank=0)["runs"][1]["dropped_rows"]
    assert both["note"] == runs_module.DROPPED_ROWS_NOTE_WITH_VERSIONS


def test_the_mcp_answers_carry_refused_batches_on_their_own_key(source):
    where = source / "runs" / "after"
    status = call(where, "get_all_ranks_status")
    assert status["refused_batches"]["batches"] == REFUSED_0 + REFUSED_1
    assert status["dropped_rows"]["rows"] == NULL_ROWS + VERSION_ROWS
    per_rank = {entry["rank"]: entry for entry in status["per_rank"]}
    assert (per_rank[0].get("dropped_rows"), per_rank[0].get("refused_batches")) == (NULL_ROWS + VERSION_ROWS, REFUSED_0)
    assert (per_rank[1].get("dropped_rows"), per_rank[1].get("refused_batches")) == (None, REFUSED_1)
    assert "dropped_rows" not in per_rank[2] and "refused_batches" not in per_rank[2]
    one = call(where, "get_all_ranks_status", {"rank": 1})["rank"]
    assert one["refused_batches"] == {"batches": REFUSED_1, "by_key": {"attempt": REFUSED_1}} and "dropped_rows" not in one
    listed = call(where, "list_incidents", {"rank": 0})["rank"]
    assert listed["refused_batches"] == {"batches": REFUSED_0, "by_key": {"step": REFUSED_0}}
    assert listed["dropped_rows"]["rows"] == NULL_ROWS + VERSION_ROWS
    assert "refused_batches" not in call(where, "list_incidents", {"rank": 2})["rank"]
    rows = {row["run_id"]: row for row in call(where, "list_runs")["runs"]}
    assert rows["after"]["refused_batches"]["batches"] == REFUSED_0 + REFUSED_1 and "refused_batches" not in rows["base"]
    compared = call(where, "compare_runs", {"runs": ["base", "after"]})["runs"]
    assert compared[1]["refused_batches"]["batches"] == REFUSED_0 + REFUSED_1 and "refused_batches" not in compared[0]
    assert not numbers(counts([status, one, listed, rows, compared])) & SUMS


def test_the_mcp_answers_bound_the_ranks_that_refused(tmp_path):
    source = tmp_path / "store"
    many = mcp_server.PER_RANK_ROWS + 5
    store(source, "base")
    store(source, "after", {r: {"refused_batches": 1, "refused_batch_keys": {"step": 1}} for r in range(many)}, ranks=many)
    status = call(source / "runs" / "after", "get_all_ranks_status")
    assert status["refused_batches"]["batches"] == many and len(status["refused_batches"]["ranks"]) == mcp_server.PER_RANK_ROWS
    assert status["refused_batches"]["ranks_omitted"] == 5
    for answer in (call(source / "runs" / "after", "list_runs")["runs"],
                   call(source / "runs" / "after", "compare_runs", {"runs": ["base", "after"]})["runs"]):
        (refused,) = [row["refused_batches"] for row in answer if "refused_batches" in row]
        assert refused["ranks_omitted"] == 5 and len(refused["ranks"]) == mcp_server.PER_RANK_ROWS


@pytest.mark.parametrize("value", [0, -1, True, "3", 2.5, None, [1]])
def test_a_refused_count_that_is_not_a_count_above_zero_is_left_out(tmp_path, value):
    source = tmp_path / "store"
    store(source, "after", {0: {"refused_batches": value, "refused_batch_keys": {"step": value}}})
    assert runs_module.refused_batches(ev.read_watchdog(source / "runs" / "after" / "watchdog.json")) is None


def test_a_run_with_no_dropped_row_and_no_refused_batch_reads_as_before(source):
    """A control, and fails before through its second half: the refused counts change nothing but
    their own key and line. The answers of the run with them, with those taken out, are the
    answers of the same run without them."""
    plain = source.parent / "plain"
    store(plain, "base")
    store(plain, "after", {0: {k: v for k, v in three_reasons()[0].items() if k.startswith("dropped")}})
    a = json.loads(json.dumps(tensorquill.compare_runs("base", "after", source=source)).replace(str(source), "<S>"))
    b = json.loads(json.dumps(tensorquill.compare_runs("base", "after", source=plain)).replace(str(plain), "<S>"))
    a["runs"][1].pop("refused_batches")
    assert [line for line in a["summary"].splitlines() if "the server refused" not in line] == b["summary"].splitlines()
    for result in (a, b):
        result.pop("written_at_ms"), result.pop("summary")
    assert a == b
    store(source, "other")
    clean = tensorquill.compare_runs("base", "other", source=source)
    assert all("refused_batches" not in run and "dropped_rows" not in run for run in clean["runs"])
    assert "the server refused" not in clean["summary"] and "the server dropped" not in clean["summary"]
    status = call(source / "runs" / "base", "get_all_ranks_status")
    assert "refused_batches" not in status and all("refused_batches" not in entry for entry in status["per_rank"])
    row = [r for r in runs_module.list_runs(source, None) if r["run_id"] == "base"][0]
    assert "refused_batches" not in row and "the server refused" not in runs_module.listing_text([row], source, None)


# --- 2. The MCP wording ------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.;:])\s+", text) if part]


def test_no_tool_description_says_every_dropped_row_had_a_null_key():
    """Fail before: four descriptions said the server dropped rows "because a key of theirs was
    null" or "with a null key", and nothing else."""
    for tool in mcp_server.TOOLS:
        for sentence in _sentences(tool["description"]):
            if "dropped" in sentence and "null" in sentence:
                assert "schema version" in sentence, (tool["name"], sentence)


def test_every_tool_that_answers_with_these_counts_names_both_reasons_and_the_refused_batches(source):
    """Pinned by what the tools answer, not by a copy of their text: a tool whose answer on the
    fixture carries `dropped_rows` or `refused_batches` must say both reasons a row is dropped and
    name `refused_batches` as a count of batches never added to the rows. A tool added later that
    answers with them fails here until its description says so."""
    where = source / "runs" / "after"
    answers = {
        "get_all_ranks_status": call(where, "get_all_ranks_status"),
        "list_incidents": call(where, "list_incidents", {"rank": 0}),
        "list_runs": call(where, "list_runs"),
        "compare_runs": call(where, "compare_runs", {"runs": ["base", "after"]}),
    }
    for tool in mcp_server.TOOLS:
        if tool["name"] in answers:
            continue
        properties = (tool.get("inputSchema") or {}).get("properties") or {}
        required = (tool.get("inputSchema") or {}).get("required") or []
        if not required:
            try:
                answers[tool["name"]] = call(where, tool["name"], {"rank": 0} if "rank" in properties else {})
            except AssertionError:  # an error answer carries no count
                continue
    carrying = {name for name, answer in answers.items()
                if re.search(r'"(dropped_rows|refused_batches)"', json.dumps(answer))}
    assert carrying >= {"get_all_ranks_status", "list_incidents", "list_runs", "compare_runs"}
    described = {tool["name"]: tool["description"] for tool in mcp_server.TOOLS}
    for name in carrying:
        text = described[name]
        assert "dropped_rows" in text and "refused_batches" in text, name
        assert "null" in text and "schema version" in text, name
        assert "key column was missing or of another type" in text, name
        assert "never added to" in text and "upper bound" in text, name


def test_the_mcp_module_says_no_dropped_row_had_only_a_null_key():
    """The docstrings and comments too: every line of `mcp_server.py` that speaks of rows dropped
    for a null key names the schema version within its paragraph."""
    text = Path(mcp_server.__file__).read_text(encoding="utf-8")
    for paragraph in re.split(r'\n\s*\n|"""', text):
        if re.search(r"dropped[^.]*null key|null key[^.]*dropped", paragraph):
            assert "schema_version" in paragraph or "schema version" in paragraph, paragraph[:300]


# --- 3. The store path of `compare`, end to end against a real server --------------------------


def _server_binary() -> Path | None:
    named = os.environ.get("TENSORQUILL_SERVER_BIN")
    if named:
        return Path(named)
    target = Path(__file__).resolve().parents[2] / "server" / "target"
    built = [p for p in (target / "debug" / "tensorquill-server", target / "release" / "tensorquill-server") if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


BINARY = _server_binary()


class _Server:
    def __init__(self, store_path: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"}
        self.process = subprocess.Popen(
            [str(BINARY), "--listen", "127.0.0.1:0", "--store", str(store_path), "--watchdog-seconds", "0.2",
             "--flush-seconds", "0.3"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        line = self.process.stdout.readline()  # "tensorquill-server listening on 127.0.0.1:PORT"
        self.endpoint = f"grpc://127.0.0.1:{int(line.rsplit(':', 1)[1])}"

    def stop(self) -> int:
        self.process.send_signal(signal.SIGTERM)
        code = self.process.wait(20)
        self.process.stdout.close()
        return code


def _put(endpoint: str, run: str, rank: int, schema: Any, rows: list[dict[str, Any]], end: int) -> Any:
    """One stream of one batch, with the metadata the sender writes: the ack's document, or the
    error the stream ended with."""
    import pyarrow as pa
    import pyarrow.flight as flight

    client = flight.FlightClient(endpoint)
    try:
        writer, reader = client.do_put(flight.FlightDescriptor.for_path("runs", run, f"rank={rank}"), schema)
        steps = [row["step"] for row in rows if isinstance(row.get("step"), int)]
        metadata = {"schema_version": 1, "run_id": run, "rank": rank, "min_step": min(steps, default=None),
                    "max_step": max(steps, default=None), "end_byte_offset": end}
        writer.write_with_metadata(pa.RecordBatch.from_pylist(rows, schema=schema), pa.py_buffer(json.dumps(metadata).encode()))
        try:
            ack = reader.read()
            writer.done_writing()
            while reader.read() is not None:
                pass
            writer.close()
        except pa.ArrowException as error:  # the status that ended the stream (a refusal is ArrowInvalid)
            return error
        return json.loads(ack.to_pybytes())
    finally:
        client.close()


def _rows(run: str, rank: int, steps: range, **fields: Any) -> list[dict[str, Any]]:
    return [{"schema_version": 1, "run_id": run, "rank": rank, "attempt": 1, "step": step, "ts": 1_000.0 + step,
             "grad_norm": 1.0, "lr": [0.1], **fields} for step in steps]


def _wait(until: Any, what: str, seconds: float = 30.0) -> Any:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = until()
        if found:
            return found
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")
def test_compare_of_a_live_servers_store_shows_its_dropped_rows_and_refused_batches(tmp_path):
    """Fail before: the compare, listing and MCP answers had no `refused_batches`. The rest is a
    test of the path itself, which had none against a live server: the rows go to a real server
    over Flight, the server writes the status and the reports, and the readers read that store."""
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.flight")
    from tensorquill.flight import flight_schema

    root = tmp_path / "store"
    server = _Server(root)
    schema = flight_schema()
    try:
        for rank in (0, 1):
            assert isinstance(_put(server.endpoint, "base", rank, schema, _rows("base", rank, range(12)), 100), dict)
            assert isinstance(_put(server.endpoint, "after", rank, schema, _rows("after", rank, range(12)), 100), dict)
        # Rank 0: a batch that keeps two rows and drops one for a null step and one for its version.
        nullable = pa.schema([field.with_nullable(True) if field.name == "step" else field for field in schema])
        mixed = [*_rows("after", 0, range(12, 14)), {**_rows("after", 0, range(1))[0], "step": None},
                 {**_rows("after", 0, range(14, 15))[0], "schema_version": 7}]
        assert isinstance(_put(server.endpoint, "after", 0, nullable, mixed, 200), dict)
        # Rank 0: a batch with no `step` column, refused; sent twice, as a sender retries it: counted once.
        no_step = pa.schema([field for field in schema if field.name != "step"])
        for _ in range(2):
            refused = _put(server.endpoint, "after", 0, no_step, [{k: v for k, v in _rows("after", 0, range(1))[0].items()
                                                                   if k != "step"}], 300)
            assert "column `step` is missing" in str(refused)
        # Rank 1: a batch whose `attempt` is text, refused; rank 1 drops no row.
        text_attempt = pa.schema([pa.field("attempt", pa.string(), nullable=False) if field.name == "attempt" else field
                                  for field in schema])
        refused = _put(server.endpoint, "after", 1, text_attempt, [{**_rows("after", 1, range(12, 13))[0], "attempt": "1"}], 200)
        assert "`attempt`" in str(refused)

        status_path = root / "runs" / "after" / "watchdog.json"

        def counted() -> Any:
            status = ev.read_watchdog(status_path)
            entries = {e.get("rank"): e for e in (status or {}).get("ranks") or ()}
            return status if (entries.get(0, {}).get("refused_batches") and entries.get(1, {}).get("refused_batches")
                              and entries.get(0, {}).get("dropped_rows")) else None

        status = _wait(counted, "the server's status to count the refused batches")
        for run in ("base", "after"):
            for rank, last in ((0, 13 if run == "after" else 11), (1, 11)):
                # Current, and of every stored row: rank 0 of `after` stored two more steps later.
                def judged(run: str = run, rank: int = rank, last: int = last) -> bool:
                    look = ev.read_server_report(root / "runs" / run / f"rank={rank}")
                    return look["state"] == "current" and (look.get("report") or {}).get("last_step") == last

                _wait(judged, f"{run} rank {rank}'s report to be current at step {last}")
    finally:
        assert server.stop() == 0

    # The keys the hand-written fixture above assumes are the ones the server writes.
    entries = {e["rank"]: e for e in status["ranks"]}
    assert (entries[0]["dropped_rows"], entries[0]["dropped_row_keys"]) == (2, {"step": 1, UNSUPPORTED: 1})
    assert (entries[0]["refused_batches"], entries[0]["refused_batch_keys"]) == (1, {"step": 1})
    assert (entries[1]["refused_batches"], entries[1]["refused_batch_keys"]) == (1, {"attempt": 1})
    assert "dropped_rows" not in entries[1]
    assert set(three_reasons()[0]) <= set(entries[0]) and set(three_reasons()[1]) <= set(entries[1])

    expected_dropped = {"rows": 2, "by_key": {"step": 1, UNSUPPORTED: 1},
                        "ranks": [{"rank": 0, "rows": 2, "by_key": {"step": 1, UNSUPPORTED: 1}}],
                        "upper_bound": True, "note": runs_module.DROPPED_ROWS_NOTE_WITH_VERSIONS}
    expected_refused = {"batches": 2, "by_key": {"attempt": 1, "step": 1},
                        "ranks": [{"rank": 0, "batches": 1, "by_key": {"step": 1}},
                                  {"rank": 1, "batches": 1, "by_key": {"attempt": 1}}],
                        "note": runs_module.REFUSED_BATCHES_NOTE}

    # The listing.
    rows = {row["run_id"]: row for row in runs_module.list_runs(root, None)}
    assert (rows["after"]["dropped_rows"], rows["after"]["refused_batches"]) == (expected_dropped, expected_refused)
    assert "dropped_rows" not in rows["base"] and "refused_batches" not in rows["base"]

    # `compare` from the server's reports alone (the MCP path's way, `read_parts=False`), and as the
    # library and the command line read it.
    for read_parts in (False, True):
        result = tensorquill.compare_runs("base", "after", source=root, read_parts=read_parts)
        base, after = result["runs"]
        assert after["read_from"]["kind"] == "server store" and after["read_from"]["judged_ranks"] == [0, 1]
        assert "judged_here" not in after["read_from"]
        assert (after["dropped_rows"], after["refused_batches"]) == (expected_dropped, expected_refused)
        assert "dropped_rows" not in base and "refused_batches" not in base
        assert after["last_step"] == 13  # the dropped rows were never stored: 12 and 13 are rank 0's last kept rows
    one = tensorquill.compare_runs("base", "after", source=root, rank=1)["runs"][1]
    assert "dropped_rows" not in one and one["refused_batches"]["ranks"] == [expected_refused["ranks"][1]]
    answer = call(root / "runs" / "base", "compare_runs", {"runs": ["base", "after"]})
    assert (answer["runs"][1]["dropped_rows"], answer["runs"][1]["refused_batches"]) == (expected_dropped, expected_refused)
    status_answer = call(root / "runs" / "after", "get_all_ranks_status")
    assert status_answer["refused_batches"] == expected_refused and status_answer["dropped_rows"] == expected_dropped

    out = tmp_path / "compare.json"
    done = subprocess.run([sys.executable, "-m", "tensorquill", "compare", "base", "after", "--source", str(root),
                           "-o", str(out)], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(Path(tensorquill.__file__).resolve().parents[1])})
    assert done.returncode == 0, done.stderr
    assert "B: the server dropped at most 2 rows with a null key or an unsupported schema version" in done.stdout
    assert "B: the server refused 2 batches whole because a key column was missing or of another type" in done.stdout
    written = json.loads(out.read_text())["runs"][1]
    assert (written["dropped_rows"], written["refused_batches"]) == (expected_dropped, expected_refused)

    # A rank whose report is not current is judged here from its stored parts, through
    # `combine_rank_evidence` (the store path with no `watchdog` block, L22's open part): the
    # counts are read from the status all the same.
    (root / "runs" / "after" / "rank=1" / "incidents.json").unlink()
    judged = tensorquill.compare_runs("base", "after", source=root)["runs"][1]
    assert judged["read_from"]["judged_here"] == [1]
    assert (judged["dropped_rows"], judged["refused_batches"]) == (expected_dropped, expected_refused)
    lone = tensorquill.compare_runs("base", "after", source=root, rank=1)["runs"][1]
    assert lone["read_from"]["judged_by"].startswith("the Python detectors, here, on the stored parts")
    assert lone["refused_batches"]["batches"] == 1 and "dropped_rows" not in lone
    with pytest.raises(ValueError, match="no current report"):
        tensorquill.compare_runs("base", "after", source=root, rank=1, read_parts=False)

    # `tensorquill-server detect --run` over the stored rows of the run: it reads no store and no
    # status, so it judges only what was stored. None of the dropped or refused rows is there.
    from tensorquill.server_detect import server_run_report

    records = ev.load_records(root / "runs" / "after", all_attempts=True)
    assert sorted({(r["rank"], r["step"]) for r in records}) == [(0, s) for s in range(14)] + [(1, s) for s in range(12)]
    document = server_run_report(records, binary=str(BINARY))
    assert [report["rank"] for report in document["reports"]] == [0, 1]
    assert [report["last_step"] for report in document["reports"]] == [13, 11]
    assert "dropped_rows" not in json.dumps(document) and "refused" not in json.dumps(document)
