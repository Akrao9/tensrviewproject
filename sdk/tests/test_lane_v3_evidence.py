"""Lane V3 (wave V): the evidence core's open rows, each held by a test that failed before its fix.

- E3: a resume incident judged from the loss scale alone, while `kind_checks` said the kind was
  never configured.
- E6 / DET-15: the Python detectors took thresholds the Rust detectors refuse.
- DET-12: a watcher whose default rank has no record wrote nothing and said nothing.
- DET-14: the Rerun export had no expert-load table at the first step, and did not let go of
  the recording when it failed.
- DET-17: the folder name `rank=<n>` was read by two different rules.
- DET-18: not a defect (the stamp matches Flight Recorder files as PyTorch's analyzer does); held.
- L6: Python's readers did not look at the status's `attempt_without_steps`.
- C16, the Python reader: batch hashes of two encodings were compared with each other.

Most tests need neither `pyarrow` nor the server binary; those that run the Rust detectors are
skipped without them.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import pytest

import tensorquill
from tensorquill import data_checks as dc
from tensorquill import evidence as ev
from tensorquill import server_detect as sd

ROOT = Path(__file__).resolve().parents[2]


def real_binary() -> Path | None:
    named = os.environ.get(sd.BINARY_ENV)
    if named:
        return Path(named)
    target = ROOT / "server" / "target"
    built = [p for p in (target / "debug" / sd.BINARY_NAME, target / "release" / sd.BINARY_NAME) if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


REAL = real_binary()
try:
    import pyarrow  # noqa: F401

    HAVE_ARROW = True
except ImportError:
    HAVE_ARROW = False
needs_rust = pytest.mark.skipif(REAL is None or not REAL.is_file() or not HAVE_ARROW,
                                reason="needs the server binary and pyarrow")


def moe_records(steps: int = 120) -> list[dict]:
    """One router that collapses at step 60: `moe_imbalance` and `starved_experts` at the defaults."""
    out = []
    for s in range(steps):
        counts = [30, 30, 30, 30] if s < 60 else [118, 1, 1, 0]
        out.append({"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": s, "ts": float(s),
                    "grad_norm": 1.0, "grad_norms": {"w": 1.0}, "moe": {"router": counts},
                    "moe_top_k": {"router": 1}})
    return out


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def run_main(monkeypatch, capsys, *argv) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["tensorquill", *map(str, argv)])
    status = 0
    try:
        ev.main()
    except SystemExit as stop:
        status = stop.code if isinstance(stop.code, int) else 1
    out = capsys.readouterr()
    return status, out.out, out.err


# --- E6 / DET-15: one options contract, the one `Thresholds::parse` in server/src/detect.rs keeps ---

# Each value the Rust rules refuse, with what their message says.
REFUSED = [
    ({"starve_window": 0}, "starve_window must be at least 1"),
    ({"starve_window": -1}, "starve_window must be"),
    ({"starve_window": 2.5}, "starve_window must be"),
    ({"starve_window": True}, "starve_window must be"),
    ({"update_window": 0}, "update_window must be"),
    ({"update_window": True}, "update_window must be"),
    ({"top_k": 0}, "top_k must be at least 1"),
    ({"top_k": -1}, "top_k must be at least 1"),
    ({"top_k": {"router": 0}}, "top_k must be at least 1"),
    ({"top_k": True}, "top_k must be"),
    ({"top_k": 2.5}, "top_k must be"),
    ({"overload_threshold": -1.0}, "overload_threshold must be a finite number of at least 0"),
    ({"overload_threshold": float("inf")}, "overload_threshold must be a finite number"),
    ({"overload_threshold": True}, "overload_threshold must be"),
    ({"overload_threshold": "0.5"}, "overload_threshold must be"),
    ({"drift_tolerance": -1.0}, "drift_tolerance must be a finite number of at least 0"),
    ({"starve_ratio": -0.5}, "starve_ratio must be a finite number of at least 0"),
    ({"starve_ratio": float("nan")}, "starve_ratio must be a finite number"),
]
# The smallest values the Rust rules take, and whole numbers where a fraction is asked for.
ACCEPTED = [
    {"overload_threshold": 0.0}, {"overload_threshold": 1}, {"drift_tolerance": 0.0}, {"starve_ratio": 0},
    {"starve_window": 1}, {"update_window": 1}, {"top_k": 1}, {"top_k": {"router": 1}}, {"top_k": None},
]


@pytest.mark.parametrize("option, reason", REFUSED)
def test_e6_the_python_detectors_refuse_what_the_rust_ones_refuse(option, reason):
    with pytest.raises(ValueError, match=reason):
        tensorquill.build_evidence(moe_records(), detectors="python", **option)


@pytest.mark.parametrize("option", ACCEPTED)
def test_e6_the_smallest_values_the_rust_rules_take_are_taken(option):
    tensorquill.build_evidence(moe_records(), detectors="python", **option)


@needs_rust
@pytest.mark.parametrize("option, reason", REFUSED)
def test_e6_the_rust_detectors_refuse_every_value_the_python_ones_refuse(option, reason):
    # The other half of the contract: nothing Python refuses is taken by Rust.
    with pytest.raises(ValueError):
        tensorquill.build_evidence(moe_records(), detectors="rust", server_binary=REAL, **option)


@needs_rust
@pytest.mark.parametrize("option", ACCEPTED)
def test_e6_the_rust_detectors_take_every_value_the_python_ones_take(option):
    tensorquill.build_evidence(moe_records(), detectors="rust", server_binary=REAL, **option)


@pytest.mark.parametrize("flag", [["--starve-window", "0"], ["--update-window", "0"], ["--top-k", "0"],
                                  ["--starve-ratio", "-0.5"], ["--overload-threshold", "nan"]])
def test_e6_the_command_line_refuses_them_as_a_usage_error_not_a_traceback(tmp_path, monkeypatch, capsys, flag):
    path = write_jsonl(tmp_path / "r.jsonl", moe_records())
    status, _, err = run_main(monkeypatch, capsys, path, "--detectors", "python", "-o", tmp_path / "out.json", *flag)
    assert status == 2 and "refuse these thresholds" in err
    assert not (tmp_path / "out.json").exists()


def test_e6_a_watcher_refuses_them_before_its_first_look(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", moe_records())
    with pytest.raises(ValueError, match="starve_window must be at least 1"):
        ev.watch(path, tmp_path / "out.json", interval=0.0, polls=1, starve_window=0, detectors="python")
    assert not (tmp_path / "out.json").exists()


# --- E3: a kind that reports an incident is never `not_configured` ----------------------------------


def scaler_reset() -> list[dict]:
    """The second audit's case: the loss scale goes 64 -> 32, and the resumed attempt starts at 64
    again, with no checkpoint manifest or optimizer snapshot recorded."""
    out = []
    for s in range(10):
        out.append({"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": s, "ts": float(s),
                    "grad_norm": 1.0, "grad_scale": 64.0 if s < 5 else 32.0, "step_skipped": False})
    for s in range(10, 15):
        out.append({"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 2, "step": s, "ts": float(s),
                    "grad_norm": 1.0, "grad_scale": 64.0, "step_skipped": False})
    return out


def _e3_check(evidence: dict) -> None:
    kinds = [i["kind"] for i in evidence["incidents"]]
    assert "resume_incomplete" in kinds
    check = evidence["kind_checks"]["resume_incomplete"]
    assert check["value"] == "checked", check
    # What was reported is not left out of what the evidence says was checked.
    assert "resume_incomplete" not in (ev.kinds_by_value(evidence) or {}).get("not_configured", [])


def test_e3_a_resume_judged_from_the_loss_scale_is_checked_under_the_python_detectors():
    _e3_check(tensorquill.build_evidence(scaler_reset(), detectors="python"))


@needs_rust
def test_e3_a_resume_judged_from_the_loss_scale_is_checked_under_the_rust_detectors():
    _e3_check(tensorquill.build_evidence(scaler_reset(), detectors="rust", server_binary=REAL))


def test_e3_a_kind_without_its_input_and_without_an_incident_stays_not_configured():
    checks = tensorquill.build_evidence(moe_records(), detectors="python")["kind_checks"]
    assert checks["resume_incomplete"]["value"] == "not_configured"


# --- DET-17: one reading of `rank=<n>` -----------------------------------------------------------


@pytest.mark.parametrize("stray", ["rank=²", "rank=--1", "rank=٣", "rank=+1", "rank= 1"])
def test_det17_a_stray_rank_folder_does_not_break_or_split_the_readers(tmp_path, stray):
    run = tmp_path / "runs" / "r"
    for name in ("rank=0", stray):
        (run / name / "attempt=1").mkdir(parents=True)
        (run / name / "attempt=1" / "part-00000000-00000004.parquet").write_bytes(b"")
    # Reading rank 0 enters no other rank's folder, and a name the server never writes is not a rank.
    found = ev._parquet_parts(run, rank=0)
    assert run / "rank=0" / "attempt=1" / "part-00000000-00000004.parquet" in found
    assert ev._discover_rank_folders(run) == {0: run / "rank=0"}
    # Every reader of the name gives the same answer.
    (run / "rank=17").mkdir()
    with os.scandir(run) as entries:
        for entry in entries:
            number = ev._rank_number(entry.name)
            assert number == ev._own_rank(Path(entry.path)), entry.name
            assert (number is not None and number != 0) == ev._other_rank_folder(entry, 0), entry.name
            assert number == {"rank=0": 0, "rank=17": 17}.get(entry.name), entry.name


# --- DET-18: the stamp matches Flight Recorder files as the analyzer reads them ---------------------


def test_det18_a_dump_with_the_prefix_inside_its_name_changes_the_stamp(tmp_path):
    # PyTorch's analyzer (torch/distributed/flight_recorder/components/loader.py, `read_dir`, since
    # pytorch#135158 in Sept 2024) reads every file whose name contains the prefix. A dump named
    # `host1_nccl_trace_rank_0` is read, so a change to it must change the stamp: `startswith`
    # would miss it.
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    dump = dumps / "host1_nccl_trace_rank_0"
    dump.write_bytes(b"a")
    before = ev._other_sources_stamp(traincheck=None, flight_recorder=dumps, flight_recorder_prefix="nccl_trace_rank_")
    dump.write_bytes(b"ab")
    after = ev._other_sources_stamp(traincheck=None, flight_recorder=dumps, flight_recorder_prefix="nccl_trace_rank_")
    assert before != after


# --- DET-12: a default rank with no record is said, once ---------------------------------------------


def _ranks(records: list[dict], rank: int) -> list[dict]:
    return [{**r, "rank": rank} for r in records]


def test_det12_a_shared_file_without_the_default_rank_says_so_once(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", _ranks(moe_records(5), 1) + _ranks(moe_records(5), 2))
    with pytest.warns(RuntimeWarning, match=r"rank 0 has no records here yet, and ranks 1-2 do") as said:
        written = ev.watch(path, tmp_path / "out.json", interval=0.0, polls=3, default_rank=0, detectors="python")
    assert written is None and not (tmp_path / "out.json").exists()  # nothing is judged for a rank with no step
    assert len([w for w in said if issubclass(w.category, RuntimeWarning)]) == 1


def test_det12_a_store_without_the_default_ranks_folder_says_so(tmp_path):
    run = tmp_path / "runs" / "r"
    for rank in (1, 2):
        (run / f"rank={rank}" / "attempt=1").mkdir(parents=True)
        (run / f"rank={rank}" / "attempt=1" / "part-00000000-00000004.parquet").write_bytes(b"")
    with pytest.warns(RuntimeWarning, match=r"rank 0 has no records here yet, and ranks 1-2 do"):
        assert ev.watch(run, tmp_path / "out.json", interval=0.0, polls=1, default_rank=0, detectors="python") is None


def test_det12_an_empty_source_stays_silent(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ev.watch(path, tmp_path / "out.json", interval=0.0, polls=2, default_rank=0, detectors="python") is None


# --- DET-14: the Rerun export ---------------------------------------------------------------------


def _rerun_evidence(steps: int) -> dict:
    return tensorquill.build_evidence(moe_records(steps), detectors="python")


def test_det14_the_expert_load_table_is_logged_from_the_first_step(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("rerun")
    tabled: list[int] = []
    table = ev.expert_load_table
    monkeypatch.setattr(ev, "expert_load_table", lambda e, step, layer: tabled.append(step) or table(e, step, layer))
    ev.export_rerun(_rerun_evidence(3), tmp_path / "three.rrd")
    assert tabled == [0, 1, 2]
    tabled.clear()
    ev.export_rerun(_rerun_evidence(1), tmp_path / "one.rrd")
    assert tabled == [0]  # a one-step recording has its table too


def test_det14_a_failed_export_lets_go_of_the_recording(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    rr = pytest.importorskip("rerun")
    let_go: list[bool] = []
    disconnect = rr.disconnect
    monkeypatch.setattr(rr, "disconnect", lambda *a, **k: let_go.append(True) or disconnect(*a, **k))
    with pytest.raises(KeyError):
        ev.export_rerun(_rerun_evidence(3), tmp_path / "bad.rrd", records=[{"no": "step"}])
    assert let_go == [True]


def test_det14_a_layer_name_with_a_path_character_is_escaped(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    rr = pytest.importorskip("rerun")
    logged: list[str] = []
    log = rr.log
    monkeypatch.setattr(rr, "log", lambda entity, *a, **k: logged.append(entity) or log(entity, *a, **k))
    records = [{"run_id": "r", "rank": 0, "step": 0, "layers": {"blocks/a b": {"abs_max": 1.0}}}]
    ev.export_rerun(_rerun_evidence(1), tmp_path / "esc.rrd", records=records)
    want = f"/layers/{rr.escape_entity_path_part('blocks/a b')}/{rr.escape_entity_path_part('abs_max')}"
    assert want in logged


# --- L6: a rank whose new attempt has stored nothing is not judged by its earlier attempt -----------

RUN = "l6run"


def _report(rank: int, attempt: int, last: int) -> dict:
    return {
        "schema_version": 1, "source": "tensorquill-server", "run_id": RUN, "rank": rank, "attempt": attempt,
        "last_step": last, "steps": last + 1, "incidents": [], "checked": list(ev.SERVER_KINDS),
        "thresholds": {"spike_sustain": ev.SPIKE_SUSTAIN},
    }


def _status(ranks: list[dict]) -> dict:
    return {
        "schema_version": 1, "run_id": RUN, "summary": "active", "server_running": True, "refresh_seconds": 30.0,
        "written_at_ms": int(time.time() * 1000), "world_size": len(ranks), "missing_ranks": [], "incidents": [],
        "ranks": [{"phase": "train", "state": "active", "severity": None, "stream": "open", **r} for r in ranks],
        "pauses": {}, "recommended_silent_seconds": None,
    }


def _store(tmp_path: Path, entries: list[dict]) -> Path:
    """A run folder with one attempt-1 part and a current report per rank, and `entries` as the status."""
    run = tmp_path / "store" / "runs" / RUN
    for entry in entries:
        folder = run / f"rank={entry['rank']}" / "attempt=1"
        folder.mkdir(parents=True)
        (folder / "part-00000000-00000009.parquet").write_bytes(b"")
        (folder.parent / "incidents.json").write_text(json.dumps(_report(entry["rank"], 1, 9)))
    (run / "watchdog.json").write_text(json.dumps(_status(entries)))
    return run


STARTED_AGAIN = {"attempt": 2, "last_step": 9, "attempt_without_steps": True}


def test_l6_run_state_reads_a_rank_started_again_with_nothing_stored_as_unknown():
    status = _status([{"rank": 0, **STARTED_AGAIN}])
    found = ev.run_state(incidents=[], steps_judged=True, status=status, judged_ranks=[0], judged_attempts={0: 1})
    assert found["state"] == "unknown" and found["reasons"][0]["ranks"] == [0]
    # Judged at the attempt the status shows, the same rank is current.
    status = _status([{"rank": 0, "attempt": 1, "last_step": 9}])
    found = ev.run_state(incidents=[], steps_judged=True, status=status, judged_ranks=[0], judged_attempts={0: 1})
    assert found["state"] == "healthy"
    # Another attempt than the one judged, with steps stored, is not current either.
    status = _status([{"rank": 0, "attempt": 3, "last_step": 12}])
    found = ev.run_state(incidents=[], steps_judged=True, status=status, judged_ranks=[0], judged_attempts={0: 1})
    assert found["state"] == "unknown"
    # A status that names no attempt (a server older than attempts) is not compared.
    status = _status([{"rank": 0, "last_step": 9}])
    found = ev.run_state(incidents=[], steps_judged=True, status=status, judged_ranks=[0], judged_attempts={0: 1})
    assert found["state"] == "healthy"


def test_l6_the_single_rank_watcher_on_a_current_report(tmp_path):
    run = _store(tmp_path, [{"rank": 0, **STARTED_AGAIN}])
    out = tmp_path / "out.json"
    ev.watch(run / "rank=0", out, interval=0.0, polls=1)
    evidence = json.loads(out.read_text())
    assert evidence["detection"]["state"] == "current"
    assert evidence["run_state"]["state"] == "unknown"
    assert "ranks_unknown" in [r["code"] for r in evidence["run_state"]["reasons"]]


def test_l6_the_all_ranks_view(tmp_path):
    run = _store(tmp_path, [{"rank": 0, "attempt": 1, "last_step": 9}, {"rank": 1, **STARTED_AGAIN}])
    out = tmp_path / "all.json"
    ev.watch(run, out, interval=0.0, polls=1, rank="all")
    evidence = json.loads(out.read_text())
    assert evidence["run_state"]["state"] == "unknown"
    unknown = [r for r in evidence["run_state"]["reasons"] if r["code"] == "ranks_unknown"]
    assert unknown and unknown[0]["ranks"] == [1]


def test_l6_detectors_run_here_over_the_earlier_attempts_records(tmp_path):
    status = _status([{"rank": 0, **STARTED_AGAIN}])
    records = [{**r, "rank": 0} for r in moe_records(10)]  # attempt 1, steps 0..9
    evidence = tensorquill.build_evidence(records, watchdog=status, detectors="python")
    assert evidence["run_state"]["state"] == "unknown"


def test_l6_the_flag_changing_is_news_to_a_watcher():
    before = _status([{"rank": 0, "attempt": 1, "last_step": 9}])
    after = _status([{"rank": 0, **STARTED_AGAIN}])
    assert ev._status_stamp(before, None) != ev._status_stamp(after, None)


# --- C16: the Python reader compares only batch hashes of one kind --------------------------------

OLD = ["5330bfb822ed0aaf", "cbfa972bd31669cd", "a7e6d0df2f96711a", "59c8f2777acde814", "532459a716ae5134",
       "01ecfdd7d5aceb8e"]
NEW = ["v2:cd09ce7a08834a9e", "v2:a509725585dcb271", "v2:725ba8bc00013ae4", "v2:cfe42ca68a54dd5a",
       "v2:d547b7bb4c4d0e14", "v2:235f294e1bb20aad"]  # the same batches I0..I5, by the writer after C16


def _drawn(attempt: int, steps, hashes) -> list[dict]:
    return [{"schema_version": 1, "run_id": "c16", "rank": 0, "attempt": attempt, "step": s, "ts": float(s),
             "data": {"batches": 1, "batch_hash": h}} for s, h in zip(steps, hashes)]


def test_c16_a_resume_across_the_encoding_change_is_not_compared():
    # The old writer drew I0..I5; the resumed attempt, by the new writer, draws I3..I5 again from step 3.
    records = _drawn(0, range(6), OLD) + _drawn(1, range(3, 6), NEW[3:])
    not_checked: list[dict] = []
    mismatch, _ = dc.detect_resume_state(records, not_checked=not_checked)
    assert mismatch is None  # never a difference
    assert [(i["check"], i["step"]) for i in not_checked] == [("resume_data", 3)]
    assert "encoding" in not_checked[0]["reason"]


def test_c16_a_resume_compares_the_hashes_of_its_own_kind_when_both_are_there():
    # Attempt 0 by the old writer; attempt 1 by the new one ran 3..5 and was cut short; attempt 2
    # resumes at 4 with I4, which attempt 1 drew there in its kind: compared, a match.
    records = _drawn(0, range(6), OLD) + _drawn(1, range(3, 6), NEW[3:]) + _drawn(2, [4, 5], NEW[4:])
    not_checked: list[dict] = []
    mismatch, _ = dc.detect_resume_state(records, not_checked=not_checked)
    assert mismatch is None and [i["step"] for i in not_checked] == [3]
    # And in its kind a different batch is still a difference.
    records = _drawn(0, range(6), OLD) + _drawn(1, range(3, 6), NEW[3:]) + _drawn(2, [4, 5], [NEW[0], NEW[5]])
    mismatch, _ = dc.detect_resume_state(records, not_checked=[])
    assert mismatch is not None and mismatch["changes"][-1]["step"] == 4


@pytest.mark.parametrize("hashes", [OLD, NEW])
def test_c16_hashes_of_one_kind_are_compared_as_before(hashes):
    same = _drawn(0, range(6), hashes) + _drawn(1, range(3, 6), hashes[3:])
    not_checked: list[dict] = []
    assert dc.detect_resume_state(same, not_checked=not_checked)[0] is None and not_checked == []
    other = _drawn(0, range(6), hashes) + _drawn(1, range(3, 6), [hashes[0], *hashes[4:]])
    assert dc.detect_resume_state(other, not_checked=[])[0]["changes"][0]["finding"] == "repeated"


def test_c16_hash_kinds():
    assert dc.hash_kind("5330bfb822ed0aaf") == "" and dc.hash_kind("v2:cd09ce7a08834a9e") == "v2"
    assert dc.hash_kind("v3:x:y") == "v3"


FIXTURE = Path(__file__).parent / "fixtures" / "c16-batch-hash-versions.json"


@pytest.mark.skipif(not FIXTURE.is_file(), reason="lane V2's shared fixture is not in this tree yet")
def test_c16_the_shared_fixture():
    """Lane V2's fixture, the one both readers are held to: each case's detector gives its verdict."""
    fixture = json.loads(FIXTURE.read_text())
    for case in fixture["cases"]:
        records = case["records"]
        detector = case["detector"]
        if detector == "resume_data_mismatch":
            found = dc.detect_resume_state(records, not_checked=[])[0]
        elif detector == "duplicate_batches":
            found = dc.detect_duplicate_batches(ev.resolve_attempts(records))
        elif detector == "data_stalled":
            found = dc.detect_data_stalled(ev.resolve_attempts(records))
        else:
            found = dc.detect_epoch_order_repeated(ev.resolve_attempts(records))
        if case["incident"] is None:
            assert found is None, case["name"]
        else:
            assert found is not None, case["name"]
            if detector == "resume_data_mismatch":
                found = {**found, **found["changes"][0]}
            assert {k: found.get(k) for k in case["incident"]} == case["incident"], case["name"]
