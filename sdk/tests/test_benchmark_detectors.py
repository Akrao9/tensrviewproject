import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.parquet")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import benchmark_detectors as bd  # noqa: E402

from tensorquill import evidence  # noqa: E402

SHAPE = {"blocks": 3, "experts": 8, "top_k": 2}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "store"
    template = bd.capture_template(**SHAPE)
    manifest = bd.write_store(path, template, ranks=3, steps=150, workers=1)
    return path, template, manifest


def test_the_template_is_a_real_capture_with_the_weight_check_fields(store):
    _, template, _ = store
    check, plain = template["check"], template["plain"]
    assert {"weight_norms", "weights_unchanged", "replica_spread", "optimizer_ownership"} <= set(check)
    assert "weight_norms" not in plain
    assert len(check["moe"]) == SHAPE["blocks"] and len(next(iter(check["moe"].values()))) == SHAPE["experts"]
    assert template["check_phase"] == check["step"] % template["check_every"]


def test_the_store_has_the_servers_layout_and_reads_back_as_the_records_written(store):
    path, template, manifest = store
    parts = sorted(path.rglob("part-*.parquet"))
    assert len(parts) == manifest["parts"] == 3 * 3
    assert parts[0].relative_to(path).parts[:3] == ("runs", bd.RUN_ID, "rank=0")
    assert parts[0].name == "part-00000000-00000049.parquet"
    one = evidence.load_records(path / "runs" / bd.RUN_ID / "rank=1", all_attempts=True)
    assert [r["step"] for r in one] == list(range(150)) and {r["rank"] for r in one} == {1}
    expected = bd.synth_record(template, 1, 1, 99)  # rank 1 shares the stream of variant 1
    assert one[99] == json.loads(json.dumps(expected))
    # A run of healthy records: no detector fires.
    assert evidence.build_evidence(one)["incidents"] == []


def test_a_timed_update_builds_what_watch_builds(store, tmp_path):
    path, template, _ = store
    view = bd.make_view(path, range(3), 150, tmp_path / "view", template["attempt"])
    row = bd.watch_update(view, 1, 100, tmp_path / "timed.json")
    assert row["steps"] == 150 and row["new_parts"] == 3 and row["incidents"] == []
    assert row["total"] == pytest.approx(row["poll"] + row["build"] + row["write"])
    # The benchmark times Python's detectors, so the watcher is told not to look for a server's report.
    out = evidence.watch(view, tmp_path / "watched.json", interval=0.0, rank=1, polls=1, server_reports=False)
    assert out.read_bytes() == (tmp_path / "timed.json").read_bytes()


def test_the_measurements_and_the_report_run_end_to_end(store, tmp_path, monkeypatch, capsys):
    path, template, _ = store
    attempt = template["attempt"]
    monkeypatch.setattr(bd, "child_memory", lambda *a, **k: {"peak_mb": 10.0, "baseline_mb": 100.0, "records": 150})
    one = bd.measure_one_rank(path, [100, 150], 1, attempt, tmp_path / "a")
    updates = bd.measure_updates(path, 3, [100, 150], 2, attempt, tmp_path / "b")
    allr = bd.measure_all_ranks(path, 3, 150, attempt, tmp_path / "c")
    profile = bd.profile_stage(path, 3, 150, updates[150], attempt, tmp_path / "p", top=5)
    assert allr["ranks"] == 3 and len(updates[150]["rows"]) == 2 and profile["stage"] in ("poll", "build", "write")
    report = {
        "machine": {"cpus": 8, "load_average_at_start": [0.1, 0.1, 0.1]},
        "store": {"ranks": 3, "steps": 150, "shape": SHAPE, "parts": 9, "bytes": 1_000_000},
        "template": template, "step_seconds": bd.STEP_SECONDS, "one_rank": one, "updates": updates,
        "all_ranks": allr, "memory": {"x": {"peak_mb": 10.0, "records": 150}}, "profile": profile,
    }
    report["budget"] = bd.budget_summary(report, 3)
    assert report["budget"]["incremental_floor"]["total"] > 0 and set(report["budget"]["fit"]) >= {"build", "total"}
    bd.print_report(report)
    printed = capsys.readouterr().out
    assert "(a) One rank" in printed and "(b) The --watch update" in printed and "floor if the detectors" in printed
    assert "Profile of the slowest stage" in printed


def test_one_memory_scenario_runs_in_a_fresh_process(store):
    path, template, _ = store
    args = bd.parse_args(["--store", str(path)])
    result = bd.child_memory("rank_build", path, args, 100, 1, template["attempt"])
    assert "error" not in result and result["records"] == 100, result
