"""`evidence.all_ranks_evidence`: the all-ranks view as one public call, returned and not written.

It is the look the all-ranks watcher makes first (`watch(run_folder, rank="all", polls=1)`),
built from the same pieces, so the two cannot drift: every case here holds them equal. The
MCP server calls it, where it used to put the view together from five private helpers.

Needs `pyarrow` (the `flight` extra) and the server binary
(`cargo build --manifest-path server/Cargo.toml`, or `TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.flight")

from tensorquill import evidence as ev  # noqa: E402
from tensorquill import mcp_server  # noqa: E402
from tensorquill.evidence import all_ranks_evidence, watch  # noqa: E402
from tensorquill.server_detect import BINARY_ENV  # noqa: E402

import test_watch_all_ranks  # noqa: E402
from test_flight_e2e import BINARY  # noqa: E402
from test_server_detectors import FAST, everything  # noqa: E402
from test_server_reports import no_part_is_read  # noqa: E402
from test_traincheck import _summary, _violation  # noqa: E402
from test_watch_all_ranks import Run, steady  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")


@pytest.fixture(autouse=True)
def nothing_but_the_reports_is_used(monkeypatch, tmp_path):
    """No part is opened, and neither the server binary nor any detector is called for."""
    no_part_is_read(monkeypatch)
    monkeypatch.setenv(BINARY_ENV, str(tmp_path / "no-such-binary"))

    def refuse(*args, **kwargs):
        raise AssertionError("the all-ranks look ran the detectors")

    monkeypatch.setattr(ev, "build_evidence", refuse)
    yield
    for server in test_watch_all_ranks.STARTED:
        if server.process.poll() is None:
            server.process.kill()
            server.process.wait(20)
        if not server.process.stdout.closed:
            server.process.stdout.close()
    test_watch_all_ranks.STARTED.clear()


def written_and_returned(folder: Path, out: Path, **options) -> tuple[dict, dict]:
    """What the watcher writes on its first look, and what the public call returns."""
    path = watch(folder, out, interval=0.0, polls=1, rank="all", **options)
    assert path is not None, "the watcher wrote nothing"
    theirs = json.loads(path.read_text(encoding="utf-8"))
    ours = json.loads(json.dumps(all_ranks_evidence(folder, **options)))
    assert theirs.pop("written_at_ms") <= ours.pop("written_at_ms")
    return theirs, ours


def mixed_run(tmp_path: Path) -> Run:
    """Five ranks: one clean, two sharing an incident, one with its own, one whose report is
    gone; and a sixth the status expects that never stored anything."""
    run = Run(tmp_path, *FAST)
    run.send(0, steady(60))
    run.send(1, steady(60, nan_at=(30, "blk2")))
    run.send(2, steady(60, nan_at=(30, "blk2"), salt=2))
    run.send(3, steady(60, spike=(40, 8.0)))
    run.send(4, steady(60))
    assert run.server.terminate() == 0
    run.report_file(4).unlink()
    return run


def test_the_public_look_is_what_the_watcher_writes_first(tmp_path):
    run = mixed_run(tmp_path)
    theirs, ours = written_and_returned(run.folder, tmp_path / "all.json")
    assert theirs == ours
    assert ours["view"] == "all_ranks" and ours["judged_ranks"] == [0, 1, 2, 3] and ours["unknown_ranks"] == [4]
    assert [(g["kind"], g["ranks"]) for g in ours["incident_groups"]] == [("grad_spike", [3]), ("non_finite", [1, 2])]
    assert ours["run_state"]["state"] == "unhealthy" and ours["thresholds"] == ev.THRESHOLD_DEFAULTS
    # It takes a path as text too, and gives a fresh document each time.
    again = all_ranks_evidence(str(run.folder))
    assert again is not all_ranks_evidence(run.folder) and again["incident_groups"] == ours["incident_groups"]


def test_the_looks_options_are_the_watchers(tmp_path):
    run = mixed_run(tmp_path)
    # A threshold given that the server did not apply: every rank with a report is refused.
    theirs, ours = written_and_returned(run.folder, tmp_path / "t.json", starve_window=20)
    assert theirs == ours and ours["ranks_by_state"] == {"other_thresholds": [0, 1, 2, 3], "missing": [4]}
    # TrainCheck's results are read once for the run.
    results = tmp_path / "violations_summary.json"
    results.write_text(json.dumps(_summary([_violation("zero_grad() resets .grad", 3, 5, "APIContainRelation")], first=3)))
    theirs, ours = written_and_returned(run.folder, tmp_path / "tc.json", traincheck=results)
    assert theirs == ours and [i["kind"] for i in ours["run_incidents"]] == ["traincheck_violation"]
    # Another status file than the run's own.
    status = json.loads((run.folder / "watchdog.json").read_text())
    elsewhere = tmp_path / "status.json"
    elsewhere.write_text(json.dumps({**status, "world_size": 7}))
    theirs, ours = written_and_returned(run.folder, tmp_path / "w.json", watchdog=elsewhere)
    assert theirs == ours and ours["world_size"] == 7 and ours["never_reporting_ranks"] == [5, 6]


def test_what_is_not_a_run_folder_is_refused_and_an_empty_one_says_nothing_is_stored(tmp_path):
    run = mixed_run(tmp_path)
    with pytest.raises(ValueError, match="is one rank's folder"):
        all_ranks_evidence(run.rank_folder(0))
    with pytest.raises(ValueError, match="needs a run folder"):
        all_ranks_evidence(run.file(0))
    empty = tmp_path / "empty"
    empty.mkdir()
    nothing = all_ranks_evidence(empty)
    assert nothing["rank_count"] == 0 and nothing["judged_ranks"] == [] and nothing["thresholds"] is None
    assert ev.incident_summary(nothing).startswith("No rank has stored anything for this run yet")
    assert nothing["run_state"]["state"] == "unknown"


def test_a_first_look_has_no_memory_so_a_report_behind_its_store_is_behind(tmp_path):
    run = Run(tmp_path, *FAST)
    run.send(0, steady(60))
    run.send(1, steady(60))
    assert run.server.terminate() == 0
    # A part the report does not cover, as the moment after a part is stored.
    newest = sorted(run.rank_folder(1).glob("attempt=*/part-*.parquet"))[-1]
    first, last = (int(n) for n in newest.stem.split("-")[1:])
    (newest.parent / f"part-{last + 1:012d}-{last + 50:012d}.parquet").write_bytes(b"")
    theirs, ours = written_and_returned(run.folder, tmp_path / "all.json")
    assert theirs == ours and ours["ranks_by_state"] == {"current": [0], "behind": [1]}


def test_the_mcp_server_calls_it_and_composes_the_view_from_no_private_helper(tmp_path):
    run = mixed_run(tmp_path)
    ours = json.loads(json.dumps(mcp_server.all_ranks_look(run.folder)))
    direct = json.loads(json.dumps(all_ranks_evidence(run.folder)))
    assert ours.pop("written_at_ms") <= direct.pop("written_at_ms") and ours == direct
    for private in ("_all_ranks_evidence", "_all_ranks_run_state", "_ranks_without_folder", "_status_age"):
        assert not hasattr(mcp_server, private), f"mcp_server still imports {private}"
    assert mcp_server.all_ranks_evidence is all_ranks_evidence


def test_the_watcher_and_the_look_share_one_body(tmp_path):
    # Not two copies: a change to what a look reads has to reach both.
    import inspect

    for function in (ev.watch_all_ranks, ev.all_ranks_evidence):
        source = inspect.getsource(function)
        assert "_look_at_ranks(" in source and "_all_ranks_whole(" in source
        assert "_all_ranks_evidence(" not in source and "_rank_look(" not in source
    assert everything  # (imported for the record sets other files share)
