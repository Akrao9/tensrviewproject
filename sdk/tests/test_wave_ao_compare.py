"""Wave AO, lane AO-6: the comparison readers (`compare.py`, `divergence.py`, `runs.py`) on `453f1e2`.

- BUG-22: a `.jsonl` file whose last line is not whole is read on its whole lines in the file's
  order, so the attempts keep the order they first appear in, and the verdict is the one the
  whole file gives. Before, the fallback joined the training records and then the scalar-only
  records, and an older attempt whose first line was a scalar-only record became the newest.
- BUG-23: a direct `first_divergence` on raw rows rolls back the scalar-only records of a
  superseded attempt, as `load_records` does, while rows resolved already (any order of their
  attempts) are still read as they are.
- BUG-26: `divergence_summary` names `clip_grad_norm` the total gradient norm before clipping.
- BUG-29 (the listing's half): a step at or above 2**63 is kept by the listing of a resumed
  file, and a file whose scalar-only record names another run says only what the readers do.
- BUG-21 (lane AO-5's fix in `evidence.py`): shown through `compare_runs`, an expected failure.

Needs nothing beyond the SDK and pytest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tensorquill import compare as compare_module
from tensorquill import evidence, runs
from tensorquill.compare import compare_runs, open_run
from tensorquill.divergence import divergence_summary, first_divergence
from tensorquill.evidence import load_records, resolve_attempts, resolve_attempts_across, scalar_series
from tensorquill.identity import MalformedRecordError

PARTIAL = '{"schema_version": 1, "run_id": "pine", "rank": 0, "step": 40, "scal'  # a line still being written


def write(path: Path, rows: list[dict[str, Any]], end: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + end)
    return path


def step_row(run: str, attempt: int, step: int, **fields: Any) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": run, "rank": 0, "attempt": attempt, "step": step, **fields}


def point(run: str, attempt: int, step: int, seq: int, **values: float) -> dict[str, Any]:
    return {"schema_version": 2, "record": "scalars", "run_id": run, "rank": 0, "attempt": attempt,
            "step": step, "seq": seq, "scalars": values}


def wobble(step: int) -> float:
    return 3.0 - 0.002 * step + 0.004 * (step % 3)


def trained(run: str, attempt: int, steps: range, logged: dict[str, float] | None = None) -> list[dict[str, Any]]:
    return [step_row(run, attempt, s, grad_norm=0.5 + 0.001 * s, scalars={"acc": wobble(s), **(logged or {})})
            for s in steps]


# --- BUG-22: a partial last line ---------------------------------------------------------------


def an_older_scalar_only_attempt() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run pine: attempt 11 logged `acc` at steps 36-38 (far off) and ended before any training
    step; attempt 12 then trained steps 0-39. Its first step, 0, rolls the points back. Run oak
    trained the same steps with the same values."""
    old = [point("pine", 11, s, n, acc=40.0 + s) for n, s in enumerate((36, 37, 38))]
    return [*old, *trained("pine", 12, range(40))], trained("oak", 1, range(40))


def an_older_attempt_that_also_trained() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run pine: attempt 21 logged `val` at steps 44-49 (1.0, then far off from step 46), attempt
    22 trained steps 0-39, then a late training row of attempt 21 at step 40 (both attempts
    trained). Attempt 22's step 0 rolls back all of attempt 21's rows. Run oak trained steps 0-39
    and logged `val` 1.0 at 44-49: pine has no `val` left to compare."""
    old = [point("pine", 21, s, n, val=1.0 if s < 46 else 70.0) for n, s in enumerate(range(44, 50))]
    pine = [*old, *trained("pine", 22, range(40)), step_row("pine", 21, 40, grad_norm=0.6, scalars={"acc": 3.0})]
    oak = [*trained("oak", 1, range(40)), *[point("oak", 1, s, n, val=1.0) for n, s in enumerate(range(44, 50))]]
    return pine, oak


@pytest.mark.parametrize("case", [an_older_scalar_only_attempt, an_older_attempt_that_also_trained])
def test_a_partial_last_line_leaves_the_verdict_as_the_whole_lines_give_it(tmp_path, case):
    pine, oak = case()
    other = write(tmp_path / "oak" / "oak.rank0.jsonl", oak)
    whole = compare_runs(str(write(tmp_path / "whole" / "pine.rank0.jsonl", pine)), str(other))["divergence"]
    assert whole["state"] == "no_divergence", whole["detail"]
    torn = compare_runs(str(write(tmp_path / "torn" / "pine.rank0.jsonl", pine, PARTIAL)), str(other))["divergence"]
    assert torn["state"] == "no_divergence", torn["detail"]
    assert torn["unfinished_lines_left_out"] == {"A": [0]}
    assert {k: v for k, v in torn.items() if k not in ("detail", "unfinished_lines_left_out")} == \
        {k: v for k, v in whole.items() if k != "detail"}


@pytest.mark.parametrize("case", [an_older_scalar_only_attempt, an_older_attempt_that_also_trained])
def test_a_partial_last_line_in_one_of_a_ranks_several_files_leaves_the_verdict(tmp_path, case):
    """The other branch of the fallback: a rank's records in two files of one run, the first torn."""
    pine, oak = case()
    other = write(tmp_path / "oak" / "oak.rank0.jsonl", oak)
    states = []
    for end in ("", PARTIAL):
        folder = tmp_path / f"run{len(end)}"
        write(folder / "pine-a.jsonl", pine, end)
        # A second file of the same run and rank: attempt 12's or 22's own copy of one more step.
        newest = max(row["attempt"] for row in pine if row["step"] == 0)
        write(folder / "pine-b.jsonl", [step_row("pine", newest, 40, grad_norm=0.54, scalars={"acc": wobble(40)})])
        found = compare_runs("pine", str(other), source=folder)["divergence"]
        states.append(found["state"])
    assert states == ["no_divergence", "no_divergence"]


def test_the_fallback_reads_the_lines_the_watcher_reads_and_checks_each_one(tmp_path):
    """Whole lines only, as `_Tail` reads them: a line torn in the middle and a last line with no
    newline are left out, and part 4 reads what the run's opening read. A record that is not valid
    after a torn line still refuses the file, naming its line."""
    rows = trained("pine", 3, range(5))
    path = tmp_path / "pine.rank0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[:2]) + "{torn\n"
                    + "".join(json.dumps(r) + "\n" for r in rows[2:4]) + json.dumps(rows[4]))
    got, torn = compare_module._step_records(path, 0)
    tail = evidence._Tail(path, 0)
    tail.poll()
    assert torn is True
    assert got == resolve_attempts([*tail.records, *tail.scalars]) == rows[:4]
    assert open_run(path)["last_step"] == 3
    bad = tmp_path / "bad.rank0.jsonl"
    bad.write_text(json.dumps(rows[0]) + "\n{torn\n" + json.dumps({**rows[1], "step": -1}) + "\n" + PARTIAL)
    with pytest.raises(MalformedRecordError, match="line 3"):
        compare_module._step_records(bad, 0)


def test_the_fallback_keeps_only_the_ranks_own_records_in_the_file_order(tmp_path):
    rows = [point("pine", 5, 2, 0, acc=1.0), {**step_row("pine", 5, 0), "rank": 1}, step_row("pine", 6, 0),
            point("pine", 6, 1, 0, acc=2.0)]
    path = write(tmp_path / "pine.jsonl", rows, PARTIAL)
    got, _ = compare_module._step_records(path, 0)
    assert got == [rows[2], rows[3]]  # attempt 5's point at 2 rolled back by attempt 6's step 0
    got, _ = compare_module._step_records([path], 0)
    assert got == [rows[2], rows[3]]


# --- BUG-23: a direct comparison of raw rows ---------------------------------------------------


def test_raw_rows_roll_back_stray_points_of_an_attempt_that_also_trained():
    """Attempt 31 trained 0-9 and logged `acc` far off at 33-36; attempt 32 resumed at step 10.
    The training spans meet but do not overlap: the points are still history."""
    pine = [*trained("pine", 31, range(10)), *[point("pine", 31, s, n, acc=80.0) for n, s in enumerate(range(33, 37))],
            *trained("pine", 32, range(10, 40))]
    oak = trained("oak", 1, range(40))
    assert first_divergence(load_records(pine, scalars=True), oak, resolved=True)["answer"] == "no_divergence"
    direct = first_divergence(pine, oak)
    assert direct["answer"] == "no_divergence", direct["reason"]
    assert direct["runs"]["a"]["records"] == 40


def test_raw_rows_roll_back_an_attempt_that_logged_only_scalars():
    pine, oak = an_older_scalar_only_attempt()
    assert first_divergence(load_records(pine, scalars=True), oak, resolved=True)["answer"] == "no_divergence"
    direct = first_divergence(pine, oak)
    assert direct["answer"] == "no_divergence", direct["reason"]


def test_rows_resolved_already_newest_first_are_still_read_as_they_are():
    """A rank's two files resolved in their attempts' order and given newest first: the restart
    (a baseline point at step 0, then steps 8-39) and what the original kept (0-7). Resolving
    again by first appearance would drop the restart; the guard keeps them, with or without the
    keyword."""
    original = [{**row, "ts": 10.0 + row["step"]} for row in trained("pine", 41, range(30))]
    restart = [{**point("pine", 42, 0, 0, acc=wobble(0)), "ts": 900.0},
               *[{**row, "ts": 901.0 + row["step"]} for row in trained("pine", 42, range(8, 40))]]
    rows = resolve_attempts_across([restart, original])
    assert sum(1 for r in rows if "record" not in r) == 40
    oak = trained("oak", 1, range(40))
    plain, keyword = first_divergence(oak, rows), first_divergence(oak, rows, resolved=True)
    assert plain["runs"]["b"]["records"] == keyword["runs"]["b"]["records"] == 40
    assert plain["answer"] == keyword["answer"] == "no_divergence"


def test_raw_rows_whose_first_attempt_trained_wholly_above_a_later_one_stay_as_they_are():
    """The guard's documented limit for training rows is unchanged: they look like resolved rows."""
    upper = trained("pine", 51, range(50, 60))
    lower = trained("pine", 52, range(0, 40))
    found = first_divergence(trained("oak", 1, range(60)), upper + lower)
    assert found["runs"]["b"]["records"] == 50


# --- BUG-26: the label of the norm before clipping ---------------------------------------------


def test_the_summary_names_clip_grad_norm_the_norm_before_clipping():
    pine = [step_row("pine", 1, s, clip_grad_norm=2.0 + 0.01 * (s % 2) if s < 34 else 90.0, grad_norm=1.0)
            for s in range(40)]
    oak = [step_row("oak", 1, s, clip_grad_norm=2.0 + 0.01 * (s % 2), grad_norm=1.0) for s in range(40)]
    found = first_divergence(pine, oak)
    assert found["divergence"]["series"] == "grad_norm/clipped"  # the series id is unchanged
    text = divergence_summary(found)
    assert "total gradient norm before clipping" in text and "after clipping" not in text


def test_the_summary_names_grad_norm_the_total_gradient_norm():
    pine = [step_row("pine", 1, s, grad_norm=1.0 + 0.01 * (s % 2) if s < 34 else 90.0) for s in range(40)]
    oak = [step_row("oak", 1, s, grad_norm=1.0 + 0.01 * (s % 2)) for s in range(40)]
    text = divergence_summary(first_divergence(pine, oak))
    assert "on the total gradient norm:" in text and "clipping" not in text


# --- BUG-29: the listing ------------------------------------------------------------------------


def test_a_resumed_files_last_step_at_or_above_2_63_is_the_one_compare_gives(tmp_path):
    big = 2**63 + 9
    path = write(tmp_path / "runs" / "far.rank0.jsonl",
                 [step_row("far", 1, 0), step_row("far", 1, 1), step_row("far", 2, 1), step_row("far", 2, big)])
    (row,) = runs.list_runs(tmp_path / "runs")
    assert row["last_step"] == open_run(path)["last_step"] == big


def test_a_resumed_file_whose_later_attempt_resumed_above_2_63_rolls_back_from_there(tmp_path):
    big = 2**63
    path = write(tmp_path / "runs" / "far.rank0.jsonl",
                 [step_row("far", 1, s) for s in (0, big + 1, big + 5)] + [step_row("far", 2, big + 2)])
    (row,) = runs.list_runs(tmp_path / "runs")
    assert row["last_step"] == open_run(path)["last_step"] == big + 2


def test_one_attempts_last_step_at_or_above_2_63_is_kept_as_before(tmp_path):
    big = 2**63 + 3
    path = write(tmp_path / "runs" / "far.rank0.jsonl", [step_row("far", 1, 0), step_row("far", 1, big)])
    (row,) = runs.list_runs(tmp_path / "runs")
    assert row["last_step"] == open_run(path)["last_step"] == big


def test_a_resumed_files_last_step_below_2_63_is_unchanged(tmp_path):
    write(tmp_path / "runs" / "near.rank0.jsonl",
          [step_row("near", 1, s) for s in range(10)] + [step_row("near", 2, s) for s in range(4, 7)])
    (row,) = runs.list_runs(tmp_path / "runs")
    assert row["last_step"] == 6


def _cli_refuses(path: Path, monkeypatch) -> bool:
    monkeypatch.setattr(sys, "argv", ["tensorquill", str(path), "--detectors", "python"])
    try:
        evidence.main()
    except SystemExit as stop:
        return stop.code not in (0, None)
    return False


def test_a_file_whose_scalar_only_record_names_another_run_says_only_what_the_readers_do(tmp_path, monkeypatch):
    """Training records of run elm and one scalar-only record of run ash in one file: the listing
    keeps `several_run_ids`, as `compare` and `scalar_series` read the file. What its detail says
    of a reader holds of that reader."""
    path = write(tmp_path / "out" / "elm.rank0.jsonl",
                 [*trained("elm", 1, range(40)), point("ash", 1, 3, 0, acc=2.0)])
    (row,) = runs.list_runs(tmp_path / "out")
    assert row["evidence"]["state"] == "several_run_ids"
    assert row["evidence"]["run_ids"] == ["ash", "elm"]
    with pytest.raises(ValueError, match="more than one run"):
        scalar_series(path)
    other = write(tmp_path / "oak" / "oak.rank0.jsonl", trained("oak", 1, range(40)))
    found = compare_runs(str(path), str(other))["divergence"]
    assert found["state"] == "no_answer" and found["several_run_ids"] == {"a": ["ash", "elm"]}
    detail = row["evidence"]["detail"]
    assert "python -m tensorquill compare" in detail
    if "python -m tensorquill refuses the file" in detail:
        assert _cli_refuses(path, monkeypatch), detail
    assert "python -m tensorquill refuses the file for the same reason" not in detail


def test_a_file_whose_training_records_name_two_runs_is_refused_by_both_as_the_detail_says(tmp_path, monkeypatch):
    path = write(tmp_path / "out" / "elm.rank0.jsonl", [*trained("elm", 1, range(20)), *trained("ash", 1, range(20, 40))])
    (row,) = runs.list_runs(tmp_path / "out")
    assert row["evidence"]["state"] == "several_run_ids"
    assert _cli_refuses(path, monkeypatch)
    assert "python -m tensorquill refuses" in row["evidence"]["detail"]


# --- BUG-21: across a rank's several files (lane AO-5's fix in evidence.py) ----------------------


@pytest.mark.xfail(strict=True, reason="BUG-21: lane AO-5's fix in evidence.resolve_attempts_across turns this on")
def test_a_ranks_several_files_give_the_same_comparison_whichever_sorts_first(tmp_path):
    """Run birch on rank 0 in two files. The older attempt 61 (ts 100) trained 0-36 and logged `val`
    far off at 31-36 after them, then trained step 37; the newer attempt 62 (ts 500) logged `val` at
    31-36 again, then resumed at step 37 and trained to 39. Attempt 62's points stand at 31-36. Run
    oak is the same run with one attempt."""
    newer = [*[{**point("birch", 62, s, n, val=4.0), "ts": 500.0 + n} for n, s in enumerate(range(31, 37))],
             *[{**row, "ts": 510.0 + row["step"]} for row in trained("birch", 62, range(37, 40), logged={"val": 4.0})]]
    older = [*[{**row, "ts": 100.0 + row["step"]} for row in trained("birch", 61, range(37), logged={"val": 4.0})],
             *[{**point("birch", 61, s, n, val=650.0), "ts": 140.0 + n} for n, s in enumerate(range(31, 37))],
             {**step_row("birch", 61, 37, grad_norm=9.0, scalars={"acc": 9.0}), "ts": 150.0}]
    oak = [point("oak", 1, 0, 0, val=4.0), *trained("oak", 1, range(40), logged={"val": 4.0})]
    other = write(tmp_path / "oak" / "oak.rank0.jsonl", oak)
    seen = []
    for first, second in (("a-newer.jsonl", "b-older.jsonl"), ("b-newer.jsonl", "a-older.jsonl")):
        folder = tmp_path / first
        write(folder / first, newer)
        write(folder / second, older)
        found = compare_runs("birch", str(other), source=folder)["divergence"]
        seen.append(found["state"])
    assert seen == ["no_divergence", "no_divergence"], seen
