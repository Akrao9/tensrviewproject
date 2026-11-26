"""Wave AO, lane AO-5: findings on `evidence.py` at the wave AL candidate.

Each section is one finding: the test that showed it first, then the pins beside the fix. The
records are written here, by hand, as `JsonlSink` writes them; none comes from a golden fixture.

Needs nothing beyond the SDK and pytest, except where a test says so: the Rerun export tests
replace Rerun with a recorder and need NumPy, and the tests that compare the two engines need
pyarrow and a `tensorquill-server` binary (`TENSORQUILL_SERVER_BIN`); each skips on its own.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from unittest import mock

import pytest

from tensorquill import evidence


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _step(attempt: int, step: int, **more) -> dict:
    return {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": attempt, "step": step, **more}


def _point(attempt: int, step: int, seq: int, **scalars) -> dict:
    return {"schema_version": 2, "record": "scalars", "run_id": "r", "rank": 0, "attempt": attempt, "step": step,
            "seq": seq, "scalars": scalars}


def _rerun_recorder(monkeypatch) -> list[tuple[int | None, str, object]]:
    """Rerun replaced by a recorder of `(step, entity, value)` for each `rr.log`; the entity path
    escaping is Rerun's own rule for the characters these tests use (a `/` in a part is `\\/`)."""
    pytest.importorskip("numpy")
    logged: list[tuple[int | None, str, object]] = []
    now: list[int | None] = [None]
    rr = mock.MagicMock(name="rerun")
    rr.escape_entity_path_part.side_effect = lambda part: part.replace("\\", "\\\\").replace("/", "\\/")
    rr.set_time.side_effect = lambda _timeline, sequence: now.__setitem__(0, sequence)
    rr.Scalars.side_effect = lambda value: ("scalars", value)
    rr.log.side_effect = lambda entity, value, **_: logged.append((now[0], entity, value))
    blueprint = mock.MagicMock(name="rerun.blueprint")
    rr.blueprint = blueprint
    monkeypatch.setitem(sys.modules, "rerun", rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", blueprint)
    return logged


# ---- BUG-20: a superseded attempt's late scalar ------------------------------------------------
#
# Attempt 100 trains steps 0 to 6. Attempt 200 resumes at step 5 and logs `val` at step 2 first;
# attempt 100, still running, then flushes its own late `val` at step 2. Both points survive the
# rollback (2 < 5). The attempt that first appears later in the whole stream is 200 (contract,
# AL-4′ decision 2), so its point stands; in the scalar-only records alone 200 appears first.

LATE = [*[_step(100, s, scalars={"loss": 10.0}) for s in range(7)],
        _point(200, 2, 0, val=2.0),
        _step(200, 5, scalars={"loss": 1.0}),
        _point(100, 2, 0, val=999.0)]


def test_bug20_a_superseded_attempts_late_scalar_does_not_replace_the_newer_one(tmp_path):
    path = _write(tmp_path / "r.rank0.jsonl", LATE)
    assert evidence.scalar_series(path)["val"] == [[2, 2.0]]
    assert evidence.scalar_series(LATE)["val"] == [[2, 2.0]]


def test_bug20_the_second_audits_order_gives_the_newer_attempts_value(tmp_path):
    """Attempt 10 trains step 0; attempt 20 logs `loss` at step 1; attempt 10 logs its late `loss` at
    step 1; attempt 20 trains step 2. The later attempt's 20.0 stands."""
    records = [_step(10, 0), _point(20, 1, 0, loss=20.0), _point(10, 1, 0, loss=10.0), _step(20, 2)]
    assert evidence.scalar_series(_write(tmp_path / "r.rank0.jsonl", records))["loss"] == [[1, 20.0]]


def test_bug20_the_order_is_the_whole_streams_before_any_record_is_resolved_away():
    """Attempt 100 trains steps 10 to 12 and is rolled back whole by attempt 200, which resumes at
    step 5: once resolved, 200's records come first. 100's late point at step 2 still survives
    (2 < 5) beside 200's, and 200 is still the later attempt."""
    records = [*[_step(100, s) for s in (10, 11, 12)], _point(200, 2, 0, val=2.0), _step(200, 5),
               _point(100, 2, 0, val=999.0)]
    assert evidence.scalar_series(records)["val"] == [[2, 2.0]]


def test_bug20_merged_scalar_points_takes_the_streams_order():
    training, scalar_only = evidence.set_apart_scalars(LATE)
    order = evidence.attempts_first_seen(LATE)
    assert order == {("r", 0): [100, 200]}
    assert evidence.merged_scalar_points(training, scalar_only, order=order)["val"] == {2: 2.0}
    assert [r["attempt"] for r in evidence.scalar_points_in_order(scalar_only, order=order)] == [100, 200]


def test_bug20_the_rerun_export_logs_the_newer_attempts_point(tmp_path, monkeypatch):
    logged = _rerun_recorder(monkeypatch)
    built = evidence.build_evidence(LATE, detectors="python")
    evidence.export_rerun(built, tmp_path / "r.rrd", records=evidence.load_records(LATE, scalars=True))
    assert [(step, value[1]) for step, entity, value in logged if entity == "/logged/val"] == [(2, 2.0)]


def test_bug20_pin_the_old_calls_still_work_and_order_by_first_appearance_among_them():
    """A caller that gives no order (`mcp_server`, `divergence` until the join) is read as before."""
    training, scalar_only = evidence.set_apart_scalars(LATE)
    assert [r["attempt"] for r in evidence.scalar_points_in_order(scalar_only)] == [200, 100]
    assert evidence.merged_scalar_points(training, scalar_only)["val"] == {2: 999.0}


def test_bug20_pin_within_one_attempt_the_highest_seq_stands_and_a_point_beats_training():
    records = [_step(1, 0, scalars={"loss": 5.0}), _point(1, 0, 1, loss=3.0), _point(1, 0, 0, loss=4.0),
               _step(1, 1, scalars={"loss": 2.0})]
    assert evidence.scalar_series(records)["loss"] == [[0, 3.0], [1, 2.0]]


def test_bug20_pin_all_attempts_keeps_the_rolled_back_point_and_the_later_attempt_still_stands():
    records = [*[_step(1, s) for s in range(4)], _point(1, 3, 0, val=1.0), _step(2, 2), _point(2, 3, 0, val=2.0)]
    assert evidence.scalar_series(records)["val"] == [[3, 2.0]]
    assert evidence.scalar_series(records, all_attempts=True)["val"] == [[3, 2.0]]
    # Without attempt 2's point, attempt 1's at 3 is rolled back by attempt 2's first training step, 2.
    assert "val" not in evidence.scalar_series(records[:6])
    assert evidence.scalar_series(records[:6], all_attempts=True)["val"] == [[3, 1.0]]


FLIPPED = [*[_step(100, s) for s in (10, 11, 12)], _point(200, 2, 0, val=2.0), _step(200, 5),
           _point(100, 2, 0, val=999.0)]


def test_bug20_the_command_lines_rerun_export_takes_the_order_from_before_the_rollback(tmp_path, monkeypatch):
    """`python -m tensorquill <file> --rerun`: the records it exports are resolved, which here puts
    attempt 200 first; the order is taken from every record it read."""
    logged = _rerun_recorder(monkeypatch)
    path = _write(tmp_path / "r.rank0.jsonl", FLIPPED)
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(path), "--detectors", "python", "--rerun",
                                      str(tmp_path / "r.rrd"), "--output", str(tmp_path / "e.json")])
    evidence.main()
    assert [(step, value[1]) for step, entity, value in logged if entity == "/logged/val"] == [(2, 2.0)]


def test_bug20_the_export_takes_an_order_given_for_records_already_resolved(tmp_path, monkeypatch):
    logged = _rerun_recorder(monkeypatch)
    built = evidence.build_evidence(LATE, detectors="python")
    evidence.export_rerun(built, tmp_path / "r.rrd", records=evidence.resolve_attempts(FLIPPED),
                          attempt_order=evidence.attempts_first_seen(FLIPPED))
    assert [(step, value[1]) for step, entity, value in logged if entity == "/logged/val"] == [(2, 2.0)]


# The two callers of the merge in other lanes' files. Each set its scalar-only records apart before
# calling it and gave no order; at the join (lane AO-S) each passes the stream's order
# (`attempts_first_seen`).

def test_bug20_at_the_join_the_mcp_series_gives_the_newer_attempts_value(tmp_path):
    from tensorquill.mcp_server import Source, get_series

    path = _write(tmp_path / "r.rank0.jsonl", [_step(10, 0), _point(20, 1, 0, loss=20.0), _point(10, 1, 0, loss=10.0),
                                                _step(20, 2)])
    assert get_series(Source(path), {"kind": "scalar", "name": "loss"})["values"] == {"value": [20.0]}


def test_bug20_at_the_join_first_divergence_reads_the_newer_attempts_point():
    from tensorquill.divergence import first_divergence

    def run(name: str, late: bool) -> list[dict]:
        rows = [{**_step(10, s, scalars={"loss": 1.0, "val": 2.0}), "run_id": name} for s in range(30)]
        rows += [{**_point(20, 31, 0, val=2.0), "run_id": name}]
        rows += [{**_step(20, s, scalars={"loss": 1.0, "val": 2.0}), "run_id": name} for s in range(35, 60)]
        if late:  # attempt 10's late point at 31, below attempt 20's first training step (35)
            rows.append({**_point(10, 31, 0, val=999.0), "run_id": name})
        return rows

    assert first_divergence(run("a", late=True), run("b", late=False))["first_exact_difference"] is None


# ---- BUG-21: a rank's several files, resolved in file order ------------------------------------
#
# The newer attempt (200, its first `ts` later) is in the first file given and the older (100,
# rolled back at step 5) in the second. 100's scalar-only point at step 0, below the cutoff, stays,
# and so does 200's at step 0; the later attempt's must stand whatever the files' order.

def _files() -> tuple[list[dict], list[dict]]:
    new = [{**_point(200, 0, 0, val=2.0), "ts": 200.0},
           *[_step(200, s, ts=200.0 + s, scalars={"loss": 1.0, "val": 2.0}) for s in range(5, 36)]]
    old = [*[_step(100, s, ts=100.0 + s, scalars={"loss": 1.0, "val": 2.0}) for s in range(5)],
           {**_point(100, 0, 0, val=999.0), "ts": 106.0},
           _step(100, 6, ts=107.0, scalars={"loss": 9.0, "val": 9.0})]
    return new, old


def test_bug21_resolving_a_ranks_files_never_depends_on_their_order():
    new, old = _files()
    one, other = evidence.resolve_attempts_across([new, old]), evidence.resolve_attempts_across([old, new])
    assert one == other
    assert [r["attempt"] for r in one] == sorted((r["attempt"] for r in one), key=[100, 200].index)


def test_bug21_first_divergence_reads_the_newer_attempts_point_across_files():
    from tensorquill.divergence import first_divergence

    new, old = _files()
    ref = [{**_point(1, 0, 0, val=2.0), "run_id": "b"},
           *[{**_step(1, s, scalars={"loss": 1.0, "val": 2.0}), "run_id": "b"} for s in range(36)]]
    for files in ([new, old], [old, new]):
        found = first_divergence(evidence.resolve_attempts_across(files), ref, resolved=True)
        assert found["answer"] == "no_divergence", found


def test_bug21_pin_the_records_left_out_are_the_same_and_one_attempts_records_keep_their_order():
    new, old = _files()
    kept = evidence.resolve_attempts_across([new, old])
    assert len(kept) == len(new) + 5 + 1  # attempt 100's steps 0 to 4 and its point at 0; step 6 rolled back
    assert [r["step"] for r in kept if r["attempt"] == 100] == [0, 1, 2, 3, 4, 0]
    assert [r["step"] for r in kept if r["attempt"] == 200] == [0, *range(5, 36)]


# ---- BUG-29, the evidence half: one run, scalar-only records included --------------------------
#
# `scalar_series` and MCP `get_series` refuse a rank's records of two run IDs, its scalar-only
# records among them (`check_one_run`); `build_evidence` and the command line set the scalar-only
# records apart first and so took a file whose training records are of run r1 and a scalar-only
# record of run r2, and the command line's --rerun then exported r2's point in r1's recording.

MIXED = [{**_step(1, 0), "run_id": "r1"}, {**_step(1, 1), "run_id": "r1"}, {**_point(1, 0, 0, loss=2.0), "run_id": "r2"}]


def test_bug29_build_evidence_refuses_a_file_of_two_runs_whose_second_is_scalar_only(tmp_path):
    path = _write(tmp_path / "mix.rank0.jsonl", MIXED)
    with pytest.raises(ValueError, match="more than one run"):
        evidence.scalar_series(path)  # as the series reader already refuses it
    with pytest.raises(ValueError, match="more than one run"):
        evidence.build_evidence(path, detectors="python")
    with pytest.raises(ValueError, match="more than one run"):
        evidence.build_evidence(MIXED, detectors="python")
    with pytest.raises(ValueError, match="more than one run"):
        evidence.build_evidence(path, rank=0, detectors="python")


def test_bug29_the_command_line_refuses_it_too(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path / "mix.rank0.jsonl", MIXED)
    for extra in ([], ["--rerun", str(tmp_path / "r.rrd")]):
        monkeypatch.setattr(sys, "argv", ["tensorquill", str(path), "--detectors", "python", "--output",
                                          str(tmp_path / "e.json"), *extra])
        with pytest.raises(SystemExit) as stopped:
            evidence.main()
        assert stopped.value.code == 2 and "more than one run" in capsys.readouterr().err
    assert not (tmp_path / "e.json").exists()


def test_bug29_pin_one_runs_scalar_only_records_and_another_ranks_are_taken(tmp_path):
    one_run = [_step(1, 0), _step(1, 1), _point(1, 0, 0, loss=2.0)]
    assert evidence.build_evidence(_write(tmp_path / "one.jsonl", one_run), detectors="python")["run_state"]["state"]
    # Another rank's scalar-only record of another run is that rank's, not this one's.
    other_rank = [*one_run, {**_point(1, 0, 0, loss=2.0), "run_id": "r2", "rank": 1}]
    assert evidence.build_evidence(other_rank, rank=0, detectors="python")["rank"] == 0


def test_bug29_the_command_lines_rerun_export_is_the_judged_ranks(tmp_path, monkeypatch):
    logged = _rerun_recorder(monkeypatch)
    records = [_step(1, 0, scalars={"loss": 1.0}), _step(1, 1), _point(1, 0, 0, loss=2.0),
               {**_point(1, 0, 0, loss=9.0), "rank": 1}]
    path = _write(tmp_path / "r.rank0.jsonl", records)
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(path), "--detectors", "python", "--rerun",
                                      str(tmp_path / "r.rrd"), "--output", str(tmp_path / "e.json")])
    evidence.main()
    assert [(step, value[1]) for step, entity, value in logged if entity == "/logged/loss"] == [(0, 2.0)]


# ---- BUG-09: a clipping norm that holds no value ------------------------------------------------
#
# Python chose the run's gradient series by whether any record has the key `clip_grad_norm`, so
# one explicit null switched `grad_spike`, `update_spike` and `gradient_trend` off `grad_norm` for
# the whole run. Rust switches on a value its float64 column carries: a number, or the sink's
# "nan", "inf" or "-inf". A null, or a value the column cannot carry (a bool, other text), is none.

SERVER_BIN = os.environ.get("TENSORQUILL_SERVER_BIN")


def _rust(records: list[dict], **options) -> dict:
    pytest.importorskip("pyarrow")
    if not SERVER_BIN or not Path(SERVER_BIN).exists():
        pytest.skip("needs a tensorquill-server binary (TENSORQUILL_SERVER_BIN)")
    return evidence.build_evidence(records, detectors="rust", server_binary=SERVER_BIN, **options)


def _spiked(**at_12) -> list[dict]:
    rows = [{"run_id": "r", "rank": 0, "step": i, "grad_norm": 100.0 if i == 22 else 1.0} for i in range(36)]
    rows[12].update(at_12)
    return rows


def _kinds(built: dict) -> list[tuple[str, int]]:
    return [(i["kind"], i["step"]) for i in built["incidents"]]


@pytest.mark.parametrize("value", [None, True, "refused"], ids=["null", "bool", "text"])
def test_bug09_a_clipping_norm_with_no_value_leaves_the_run_on_grad_norm(value):
    built = evidence.build_evidence(_spiked(clip_grad_norm=value), detectors="python")
    assert _kinds(built) == [("grad_spike", 22)]
    assert built["gradient_trend"]["series"] == "grad_norm"


def test_bug09_a_grad_norm_the_column_cannot_carry_is_no_gradient_input():
    built = evidence.build_evidence([{"run_id": "r", "rank": 0, "step": 0, "grad_norm": True}], detectors="python")
    assert built["kind_checks"]["grad_spike"]["value"] == "not_configured"
    assert evidence.input_counts([{"step": 0, "grad_norm": True, "clip_grad_norm": "x"}])["grad_total"] == 0


@pytest.mark.parametrize("value", ["nan", "inf", 3.0], ids=["nan", "inf", "number"])
def test_bug09_pin_a_carried_clipping_norm_still_switches_the_whole_run(value):
    """A value the column carries, finite or not, says the run clips: every record's total is then
    the pre-clipping norm, and the records without one have none (the spike at 22 is not seen)."""
    built = evidence.build_evidence(_spiked(clip_grad_norm=value), detectors="python")
    assert ("grad_spike", 22) not in _kinds(built)
    assert evidence.input_counts([{"step": 0, "clip_grad_norm": value}])["grad_total"] == 1


@pytest.mark.parametrize("at_12", [{}, {"clip_grad_norm": None}, {"clip_grad_norm": True}, {"clip_grad_norm": "nan"},
                                   {"clip_grad_norm": 3.0}, {"grad_norm": True}],
                         ids=["control", "null", "bool", "nan", "number", "grad_norm_bool"])
def test_bug09_both_engines_give_the_same_gradient_verdict(at_12):
    rows = _spiked(**at_12)
    python, rust = evidence.build_evidence(rows, detectors="python"), _rust(rows)
    assert _kinds(python) == _kinds(rust)
    assert python["run_state"]["state"] == rust["run_state"]["state"]
    assert python["kind_checks"]["grad_spike"]["value"] == rust["kind_checks"]["grad_spike"]["value"]
    assert python.get("gradient_trend") == rust.get("gradient_trend")


def _random_run(seed: int) -> list[dict]:
    """60 steps of gradient and update sizes, with a spike, a skipped step and, in every third seed,
    a null clipping norm somewhere; in some seeds a clipping run."""
    draw = random.Random(seed)
    clips = seed % 5 == 4
    spike, null_at = draw.randrange(20, 55), draw.randrange(0, 60) if seed % 3 == 0 else None
    rows = []
    for step in range(60):
        total = (25.0 if step == spike else 1.0) * (1 + draw.uniform(-0.05, 0.05))
        row = {"run_id": "r", "rank": 0, "step": step, "grad_norms": {"w": total * 0.8, "b": total * 0.2},
               "update_rms": {"w": 0.01 * total * (1 + draw.uniform(-0.02, 0.02)), "b": 0.001}, "lr": [0.01]}
        if clips:
            row.update(grad_norm=None, clip_grad_norm=total)
        else:
            row["grad_norm"] = total
        if step == null_at:
            row["clip_grad_norm"] = None
        if step == 40 and seed % 4 == 1:
            row["step_skipped"] = True
        rows.append(row)
    return rows


GRAD_KINDS = ("grad_spike", "update_spike")


@pytest.mark.parametrize("seed", range(30))
def test_bug09_thirty_seeds_both_engines_agree_on_the_gradient_kinds(seed):
    rows = _random_run(seed)
    python, rust = evidence.build_evidence(rows, detectors="python"), _rust(rows)
    assert ([x for x in _kinds(python) if x[0] in GRAD_KINDS] == [x for x in _kinds(rust) if x[0] in GRAD_KINDS])
    for kind in GRAD_KINDS:
        assert python["kind_checks"][kind]["value"] == rust["kind_checks"][kind]["value"]
    assert python.get("gradient_trend") == pytest.approx(rust.get("gradient_trend"))


# ---- BUG-11: the replica comparison, read as replica.rs reads an observation -------------------
#
# REPLICA_OBSERVATION_CONTRACT.md section 1: `v` is 1; `fp` is `[norm, sum]` or `[norm, sum,
# projection]`; `sig` a decimal string. Rust reads a record of another `v` as naming no group, an
# entry with fewer than two fingerprints (or none) as unread, and compares `sig` as JSON values.

LAUNCH = "a" * 32


def _replica(rank: int, *, sig=None, fp=None, ranks=(0, 1), **replica) -> dict:
    param = {"shape": [2], "dtype": "float32", "fp": [1.0, 1.0] if fp is None else fp,
             "sig": f"s{rank}" if sig is None else sig}
    body = {"v": 1, "launch": LAUNCH, "group": list(ranks), "signature": "fold64-v1", "params": {"w": param}, **replica}
    return {"run_id": "r", "rank": rank, "step": 49, "replica": body}


def _steps(records: list[dict]) -> list[dict]:
    return evidence.compare_replicas(records)["steps"]


def _rust_replica(records: list[dict]) -> dict:
    pytest.importorskip("pyarrow")
    if not SERVER_BIN or not Path(SERVER_BIN).exists():
        pytest.skip("needs a tensorquill-server binary (TENSORQUILL_SERVER_BIN)")
    from tensorquill.server_detect import server_run_report

    return server_run_report(records, binary=SERVER_BIN)["replica"]


@pytest.mark.parametrize("version", ["absent", 2, 1.0, True, "1"])
def test_bug11_an_observation_of_another_version_names_no_group(version):
    records = [_replica(r, fp=[1.0 + 8 * r, 1.0]) for r in (0, 1)]
    for record in records:
        if version == "absent":
            del record["replica"]["v"]
        else:
            record["replica"]["v"] = version
    found = evidence.compare_replicas(records)
    assert found["replica_drift"] == []
    assert found["steps"] == [{"launch": LAUNCH, "group": None, "step": 49, "coverage": "not_established", "why": "group"}]


@pytest.mark.parametrize("fp", [[9.0], [], None, "1.0", {"0": 1.0}], ids=["one", "empty", "null", "text", "object"])
def test_bug11_an_entry_with_fewer_than_two_fingerprints_is_unread(fp):
    records = [_replica(r, sig="s", fp=[1.0 + 8 * r][: len(fp)] if isinstance(fp, list) else [1.0, 1.0]) for r in (0, 1)]
    for record in records:
        if not isinstance(fp, list):
            record["replica"]["params"]["w"]["fp"] = fp
    step = _steps(records)[0]
    assert step["fingerprints"] == "not_compared" and step["fingerprints_why"] == "no_parameters"
    assert step["signatures"] == "not_compared" and step["not_compared"] == {"w": "unread"}


@pytest.mark.parametrize("pair", [(123, "123"), (True, 1), (1, 1.0)], ids=["int_text", "bool_int", "int_float"])
def test_bug11_signatures_are_compared_as_json_values(pair):
    records = [_replica(r, sig=pair[r]) for r in (0, 1)]
    step = _steps(records)[0]
    assert step["signatures"] == "differ" and step["signature_parameters"] == ["w"]


def test_bug11_attribution_by_signatures_compares_json_values_too():
    records = [_replica(r, sig=123 if r == 0 else "123", ranks=(0, 1, 2)) for r in (0, 1, 2)]
    attributed = evidence.compare_replicas(records)["attribution"]
    assert attributed == [{"step": 49, "launch": LAUNCH, "group": [0, 1, 2], "parameter": "w", "by": "signatures",
                           "verdict": "attributed", "outliers": [0]}]


def test_bug11_pin_the_contracts_record_is_compared_as_before():
    agree = [_replica(r, sig="-4611686018427387904", fp=[1.25, -0.5, 0.125], projection="name-v1") for r in (0, 1)]
    assert _steps(agree)[0]["fingerprints"] == "agree" and _steps(agree)[0]["signatures"] == "agree"
    drift = [_replica(r, sig="7", fp=[1.25 + r, -0.5]) for r in (0, 1)]
    found = evidence.compare_replicas(drift)
    assert found["steps"][0]["fingerprints"] == "drift" and len(found["replica_drift"]) == 1


REPLICA_CASES = {
    "no_v": [{**_replica(r, fp=[1.0 + 8 * r, 1.0]), "replica": {k: v for k, v in _replica(r, fp=[1.0 + 8 * r, 1.0])["replica"].items() if k != "v"}} for r in (0, 1)],
    "short_fp": [_replica(r, sig="s", fp=[1.0 + 8 * r]) for r in (0, 1)],
    "empty_fp": [_replica(r, sig="s", fp=[]) for r in (0, 1)],
    "sig_int_text": [_replica(r, sig=123 if r == 0 else "123") for r in (0, 1)],
    "sig_bool_int": [_replica(r, sig=True if r == 0 else 1) for r in (0, 1)],
    "three_ranks_sig": [_replica(r, sig=123 if r == 0 else "123", ranks=(0, 1, 2)) for r in (0, 1, 2)],
    "group_not_ints": [_replica(r, ranks=(0, "1")) for r in (0, 1)],
    "launch_not_text": [_replica(r, launch=5) for r in (0, 1)],
    "the_contracts": [_replica(r, sig="-4611686018427387904", fp=[1.25, -0.5, 0.125], projection="name-v1") for r in (0, 1)],
}


@pytest.mark.parametrize("name", sorted(REPLICA_CASES))
def test_bug11_both_engines_give_the_same_replica_comparison(name):
    records = REPLICA_CASES[name]
    python, rust = evidence.compare_replicas(records), _rust_replica(records)
    for key in ("steps", "replica_drift", "attribution"):
        assert python.get(key) == rust.get(key), key


# ---- BUG-13: a change of `moe_top_k` during a run -------------------------------------------------
#
# Rust judges each record with the experts per token in force when it arrives (its own value, or
# the last one recorded before it; 1 before any); Python took the run's last value for every step.

TOP_K_CHANGE = [{"run_id": "r", "rank": 0, "step": 0, "moe": {"router": [45, 45, 2, 2, 2, 2, 1, 1]}, "moe_top_k": {"router": 1}},
                {"run_id": "r", "rank": 0, "step": 1, "moe": {"router": [12, 13, 12, 13, 12, 13, 12, 13]}, "moe_top_k": {"router": 4}}]


def _overloaded(built: dict) -> list[tuple[int, list[int], str | None]]:
    return [(s["step"], s["overloaded"], (s.get("overload_rule") or {}).get("rule")) for s in built["samples"]]


def test_bug13_each_step_is_judged_with_the_experts_per_token_then_in_force():
    built = evidence.build_evidence(TOP_K_CHANGE, detectors="python")
    assert _overloaded(built) == [(0, [1, 2], "share"), (1, [], "diversity_drop")]
    assert built["top_k"] == {"router": 4}  # the value in force at the end, as before


def _changing_run() -> list[dict]:
    """Top-1 for 30 steps, then top-4 with a collapse onto two experts from step 50."""
    draw = random.Random(13)
    rows = []
    for step in range(80):
        k = 1 if step < 30 else 4
        counts = [draw.randrange(90, 110) for _ in range(8)]
        if step >= 50:
            counts = [400, 400, *[draw.randrange(0, 5) for _ in range(6)]]
        row = {"run_id": "r", "rank": 0, "step": step, "moe": {"router": counts}}
        if step % 10 == 0:  # recorded on some steps only: the last value stands between them
            row["moe_top_k"] = {"router": k}
        rows.append(row)
    return rows


@pytest.mark.parametrize("top_k", [None, 2, {"router": 3}], ids=["recorded", "one_for_all", "per_router"])
def test_bug13_both_engines_give_the_same_samples_when_the_top_k_changes(top_k):
    for rows in (TOP_K_CHANGE, _changing_run()):
        python, rust = evidence.build_evidence(rows, detectors="python", top_k=top_k), _rust(rows, top_k=top_k)
        assert _overloaded(python) == _overloaded(rust)
        assert [(s["step"], s["dead"]) for s in python["samples"]] == [(s["step"], s["dead"]) for s in rust["samples"]]
        assert _kinds(python) == _kinds(rust)
        assert python["top_k"] == rust["top_k"]


def test_bug13_pin_an_override_stands_over_every_recorded_value():
    built = evidence.build_evidence(TOP_K_CHANGE, detectors="python", top_k=1)
    assert [rule for _, _, rule in _overloaded(built)] == ["share", "share"] and built["top_k"] == {"router": 1}


def test_bug13_pin_a_router_is_judged_top_1_before_its_first_recorded_value():
    rows = [{**TOP_K_CHANGE[0], "moe_top_k": {}}, TOP_K_CHANGE[1]]
    assert _overloaded(evidence.build_evidence(rows, detectors="python"))[0] == (0, [1, 2], "share")


# ---- BUG-12: a malformed field outside the identity fields ---------------------------------------
#
# WAVE_AF_CONTRACT.md section 1: only the identity fields refuse a record; any other field that
# cannot be read is unreadable to the detector that wants it, and nothing raises. Rust's `detect`
# reads such a field as absent (its column is null; the value travels in `extra_json`).

def _base(step: int) -> dict:
    record = {"schema_version": 1, "run_id": "r", "rank": 0, "attempt": 1, "step": step, "ts": 100.0 + step,
              "grad_norm": 1.0, "grad_norms": {"w": 0.9, "b": 0.1}, "lr": [0.01], "update_rms": {"w": 0.01, "b": 0.001},
              "moe": {"router": [10, 12, 9, 11]}, "moe_top_k": {"router": 1}, "scalars": {"loss": 2.0 - step / 100},
              "layers": {"block.0": {"mean": 0.0, "std": 1.0, "abs_max": 3.0, "zero_frac": 0.1}},
              "data_wait": 0.01, "step_time": 0.1, "grad_scale": 1024.0}
    if step % 5 == 0:
        record.update(weight_norms={"w": 1.0 + step / 100, "b": 0.5}, weights_unchanged=[])
    return record


FIELDS = ("ts", "grad_norm", "clip_grad_norm", "grad_norms", "lr", "update_rms", "weight_norms", "weights_unchanged",
          "weight_dtypes", "optimizer_ownership", "moe", "moe_top_k", "moe_stats", "moe_refused", "scalars", "layers",
          "probe", "data", "checkpoint", "system", "data_wait", "step_time", "grad_scale", "step_skipped",
          "step_skipped_part", "scaler_applied", "clip_calls", "replica", "replica_spread", "replica_spread_scope",
          "update_not_read", "grads_not_read", "layers_not_read", "batch_hash")
BAD = ([1, 2], 0.001, True, "x", {"w": "x"}, {"w": [1]}, [{"w": 1}], None, -1)


@pytest.mark.parametrize("field", FIELDS)
def test_bug12_no_malformed_field_outside_the_identity_fields_raises(field):
    for bad in BAD:
        records = [{**_base(step), field: bad} for step in range(12)]
        built = evidence.build_evidence(records, detectors="python")
        assert built["run_state"]["state"] in evidence.RUN_STATES, (field, bad)


@pytest.mark.parametrize("field", FIELDS)
def test_bug12_the_rerun_export_never_raises_on_one_either(field, tmp_path, monkeypatch):
    _rerun_recorder(monkeypatch)
    for bad in BAD:
        records = [{**_base(step), field: bad} for step in range(12)]
        evidence.export_rerun(evidence.build_evidence(records, detectors="python"), tmp_path / "r.rrd", records=records)


AUDIT_CASES = {
    "weight_norms_list": [{"run_id": "r", "rank": 0, "step": 0, "weight_norms": [1, 2]},
                          {"run_id": "r", "rank": 0, "step": 50, "weight_norms": [1, 2]}],
    "lr_number": [{"run_id": "r", "rank": 0, "step": 0, "lr": 0.001}],
    "weights_unchanged_bool": [{"run_id": "r", "rank": 0, "step": 0, "weights_unchanged": True}],
}


@pytest.mark.parametrize("name", sorted(AUDIT_CASES))
def test_bug12_the_audits_three_inputs_are_read_by_both_engines(name):
    records = AUDIT_CASES[name]
    python = evidence.build_evidence(records, detectors="python")
    rust = _rust(records)
    assert _kinds(python) == _kinds(rust)
    assert python["run_state"]["state"] == rust["run_state"]["state"]


def test_bug12_weight_trends_reads_the_last_check_that_holds_a_norm():
    records = [{"run_id": "r", "rank": 0, "step": 0, "weight_norms": {"w": 1.0}},
               {"run_id": "r", "rank": 0, "step": 50, "weight_norms": {"w": 2.0}},
               {"run_id": "r", "rank": 0, "step": 100, "weight_norms": {"w": "refused"}}]
    built = evidence.build_evidence(records, detectors="python")
    assert built["weight_jumps"]["checks"] == 2
    assert built["weight_trends"]["latest_step"] == 50
    assert built["weight_trends"]["kinds"]["matrix"]["median_ratio"] == 2.0


def test_bug12_pin_a_weights_unchanged_list_of_names_is_still_read():
    records = []
    for step in range(8):
        records.append({"run_id": "r", "rank": 0, "step": step, "grad_norm": 1.0, "grad_norms": {"w": 1.0},
                        "weight_norms": {"w": 1.0}, "weights_unchanged": ["w"] if step else []})
    assert ("stalled_weights", 3) in _kinds(evidence.build_evidence(records, detectors="python"))


# ---- BUG-28, the export half: one entity per logged name ------------------------------------------
#
# Capture takes any non-empty string as a scalar's name. The export split it on "/" and dropped the
# empty parts, so "loss" and "/loss" (and "a/b" and "a//b") were one `/logged/...` entity, with two
# points at one step and one series in the blueprint.

def _logged(logged: list) -> dict[str, list[tuple[int, float]]]:
    found: dict[str, list[tuple[int, float]]] = {}
    for step, entity, value in logged:
        if entity.startswith("/logged/"):
            found.setdefault(entity, []).append((step, value[1]))
    return found


def test_bug28_names_with_empty_parts_get_their_own_entities(tmp_path, monkeypatch):
    logged = _rerun_recorder(monkeypatch)
    records = [_step(1, 0, scalars={"loss": 1.0, "a/b": 3.0, "a//b": 4.0, "loss/": 5.0}), _point(1, 0, 0, **{"/loss": 9.0})]
    evidence.export_rerun(evidence.build_evidence(records[:1], detectors="python"), tmp_path / "r.rrd", records=records)
    by_entity = _logged(logged)
    assert len(by_entity) == 5 and all(len(points) == 1 for points in by_entity.values())
    assert by_entity["/logged/loss"] == [(0, 1.0)]  # a scalar-only "/loss" at step 0 does not stand over "loss"
    assert by_entity["/logged/a/b"] == [(0, 3.0)]
    assert sorted(v for points in by_entity.values() for _, v in points) == [1.0, 3.0, 4.0, 5.0, 9.0]


def test_bug28_pin_a_name_without_empty_parts_keeps_its_path(tmp_path, monkeypatch):
    logged = _rerun_recorder(monkeypatch)
    records = [_step(1, 0, scalars={"train/loss": 1.0, "eval loss": 2.0}), _point(1, 1, 0, **{"train/loss": 0.5})]
    evidence.export_rerun(evidence.build_evidence(records[:1], detectors="python"), tmp_path / "r.rrd", records=records)
    assert _logged(logged) == {"/logged/train/loss": [(0, 1.0), (1, 0.5)], "/logged/eval loss": [(0, 2.0)]}


def test_bug28_the_real_rerun_keeps_them_apart(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("rerun")
    chunk = pytest.importorskip("rerun.chunk")
    records = [_step(1, 0, scalars={"loss": 1.0}), _point(1, 0, 0, **{"/loss": 9.0, "a//b": 4.0})]
    out = evidence.export_rerun(evidence.build_evidence(records[:1], detectors="python"), tmp_path / "r.rrd", records=records)
    entities = {str(c.entity_path) for c in chunk.RrdReader(out).store().stream().filter(is_static=False).to_chunks()}
    assert {e for e in entities if e.startswith("/logged/")} == {"/logged/loss", "/logged/\\/loss", "/logged/a\\/\\/b"}
