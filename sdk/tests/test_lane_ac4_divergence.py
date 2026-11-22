"""Lane AC4: `first_divergence` given raw rows resolves each run's attempts itself.

A capture can hold several attempts of the same steps: a job that restarted from a checkpoint
writes those steps again as a new attempt, and what the earlier attempt recorded from there on is
history that training rolled back. Before this lane, `first_divergence` read every row it was
given, so called directly on raw rows it could report a departure the run never had: the
superseded attempt's rows past the replacement's last step were read as the run's (bug register:
the divergence was named at step 80 although the replacement attempt had stopped at 75).

What is held here:
- raw rows give what the resolved rows give (`load_records`, which resolves attempts);
- rows already resolved, and rows with no `attempt` key, give what they gave before: resolving
  again changes nothing, also for `compare`'s rows of a rank's several files, which are resolved
  in an order the rows alone do not show;
- with several ranks, `rank=` reads that rank's own resolved rows;
- the result says how many rows were left out, only when some were.
"""

import json

from tensorquill.divergence import divergence_summary, first_divergence
from tensorquill.evidence import load_records, resolve_attempts, resolve_attempts_across

from test_divergence import (
    identical_early_pair,
    make,
    noisy_pair,
    unpaired,
    wave,
    with_grad,
)


def spiked(steps, *, attempt, run_id="b", rank=0, start=80):
    """Run B's rows of one attempt: `w1`'s gradient norm fifty times A's from `start` on."""
    rows = make(steps, run_id=run_id, rank=rank,
                grads={"w1": lambda s: wave(s, 0.1, 1.0) * (50 if s >= start else 1), "w2": lambda s: wave(s, 0.2, 2.0)})
    return [{**row, "attempt": attempt} for row in rows]


def healthy(steps, *, attempt, run_id="b", rank=0):
    return [{**row, "attempt": attempt} for row in make(steps, run_id=run_id, rank=rank)]


def register_case():
    """Run A, and run B's raw rows: B's first attempt ran on past step 80 and departs there;
    the job restarted from the checkpoint at step 60 and the replacement attempt stopped at 75,
    agreeing with A all the way."""
    a = make()
    raw_b = spiked(range(120), attempt=1) + healthy(range(60, 76), attempt=2)
    return a, raw_b


def without_left_out(result):
    """The result as it would be without the key that counts rows left out."""
    out = json.loads(json.dumps(result, allow_nan=False))
    for side in ("a", "b"):
        out["runs"][side].pop("replaced_records", None)
    return out


def same(one, other):
    return json.dumps(one, sort_keys=True, allow_nan=False) == json.dumps(other, sort_keys=True, allow_nan=False)


# ---- the register's case -------------------------------------------------------------------


def test_raw_rows_of_a_rolled_back_attempt_are_not_read_as_the_runs():
    a, raw_b = register_case()
    resolved = first_divergence(a, load_records(raw_b))
    assert resolved["answer"] == "no_divergence", resolved["reason"]
    assert resolved["runs"]["b"]["last_step"] == 75
    found = first_divergence(a, raw_b)
    # Before lane AC4 this was "diverged" at step 80, on rows training had rolled back.
    assert found["answer"] == "no_divergence", found["reason"]
    assert found["divergence"] is None and found["diverged_series"] == []
    assert found["runs"]["b"]["last_step"] == 75 and found["runs"]["b"]["records"] == 76
    assert same(without_left_out(found), without_left_out(resolved))


def test_the_result_counts_the_rows_left_out_only_when_there_are_some():
    a, raw_b = register_case()
    found = first_divergence(a, raw_b)
    # Attempt 1's rows from step 60 on (60 of them) were replaced by the restart at step 60.
    assert found["runs"]["b"]["replaced_records"] == 60
    assert "replaced_records" not in found["runs"]["a"]
    resolved = first_divergence(a, load_records(raw_b))
    assert "replaced_records" not in resolved["runs"]["a"] and "replaced_records" not in resolved["runs"]["b"]
    assert divergence_summary(found) == divergence_summary(resolved)


def test_both_runs_are_resolved_and_the_answer_is_the_resolved_rows_answer():
    # Run A restarted too: its first attempt ran on, changed, past where its replacement stopped.
    raw_a = spiked(range(120), attempt=7, run_id="a", start=90) + healthy(range(50, 100), attempt=3, run_id="a")
    a, raw_b = register_case()
    found = first_divergence(raw_a, raw_b)
    expected = first_divergence(load_records(raw_a), load_records(raw_b))
    assert found["answer"] == expected["answer"] == "no_divergence", found["reason"]
    assert same(without_left_out(found), without_left_out(expected))
    assert found["runs"]["a"]["replaced_records"] == 70 and found["runs"]["b"]["replaced_records"] == 60


def test_a_departure_on_the_replacement_attempt_itself_is_still_found():
    # The restart's own rows depart at step 85: that is the run's, and it is named.
    a = make()
    raw_b = healthy(range(100), attempt=1) + spiked(range(60, 120), attempt=2, start=85)
    found = first_divergence(a, raw_b)
    assert found["answer"] == "diverged" and found["divergence"]["step"] == 85
    assert same(without_left_out(found), without_left_out(first_divergence(a, load_records(raw_b))))
    assert found["runs"]["b"]["replaced_records"] == 40


def test_raw_rows_given_as_a_generator_are_read_once_and_resolved():
    a, raw_b = register_case()
    found = first_divergence(iter(a), (row for row in raw_b))
    assert found["answer"] == "no_divergence" and found["runs"]["b"]["last_step"] == 75


# ---- ranks ---------------------------------------------------------------------------------


STEPLESS = [{"run_id": "b", "rank": 1, "attempt": 2, "note": "no step"}, {"run_id": "b", "rank": 0, "step": None}]


def two_ranks():
    """Raw rows of two ranks of run B: rank 1 restarted at 60 and stopped at 75, its first
    attempt departing at 80; rank 0 ran once and departs at 100. Rows with no step were among
    them before wave AF, read as not step records; such a row is refused since
    (WAVE_AF_CONTRACT.md, section 1), as `test_rows_with_no_step_are_refused` says."""
    a = make() + make(rank=1)
    raw_b = (
        spiked(range(120), attempt=1, rank=0, start=100)
        + spiked(range(120), attempt=1, rank=1)
        + healthy(range(60, 76), attempt=2, rank=1)
    )
    return a, raw_b


def test_rows_with_no_step_are_refused():
    """A row with no step, or a null one, cannot be placed in a step: the input is refused with the
    row's place and run, never read past (WAVE_AF_CONTRACT.md, section 1)."""
    import pytest

    from tensorquill.identity import MalformedRecordError

    a, raw_b = two_ranks()
    for row in STEPLESS:
        with pytest.raises(MalformedRecordError) as raised:
            first_divergence(a, raw_b + [row], rank=1)
        assert raised.value.field == "step" and raised.value.where == f"record {len(raw_b)} of the records passed, run B"


def resolved_rank(rows, rank):
    """`load_records(rows, rank=rank)` over the rows that have a step."""
    return load_records([row for row in rows if row.get("step") is not None], rank=rank)


def test_rank_reads_that_ranks_own_resolved_rows():
    a, raw_b = two_ranks()
    on_one = first_divergence(a, raw_b, rank=1)
    assert on_one["rank"] == 1 and on_one["answer"] == "no_divergence", on_one["reason"]
    assert on_one["runs"]["b"]["last_step"] == 75 and on_one["runs"]["b"]["replaced_records"] == 60
    assert same(without_left_out(on_one), without_left_out(first_divergence(a, resolved_rank(raw_b, 1), rank=1)))
    # Rank 0 had no restart: its rows are read as they are, and its own departure is named.
    on_zero = first_divergence(a, raw_b, rank=0)
    assert on_zero["answer"] == "diverged" and on_zero["divergence"]["step"] == 100
    assert "replaced_records" not in on_zero["runs"]["b"]
    assert same(on_zero, first_divergence(a, resolved_rank(raw_b, 0), rank=0))
    # A rank in neither run: the reason still names the ranks the raw rows hold.
    missing = first_divergence(raw_b, a, rank=3)
    assert missing["answer"] == "no_answer"
    assert "run A has no records of rank 3 (it has ranks [0, 1])" in missing["reason"]


# ---- what was already resolved, and rows with no attempt key, are read as before ------------


def inputs():
    """Pairs of runs from the existing divergence tests: diverged, no divergence and no answer."""
    a, raw_b = register_case()
    return {
        "identical": (make(), make(run_id="b")),
        "one_parameter": (make(), with_grad(lambda s, v: v * 50 if s >= 70 else v)),
        "noisy_inside": noisy_pair(11.0),
        "noisy_outside": noisy_pair(13.0),
        "came_back": noisy_pair(30.0, start=40, length=5),
        "identical_early": identical_early_pair(0.02),
        "unpaired": (unpaired(1), unpaired(2)),
        "two_ranks": (make() + make(rank=1), make(run_id="b") + [
            {**row, "rank": 1} for row in with_grad(lambda s, v: v * 50 if s >= 70 else v)]),
        "register_case_resolved": (a, load_records(raw_b)),
    }


def test_rows_already_resolved_or_with_no_attempt_give_the_same_result_and_raw_rows_give_the_resolved_ones():
    several_files_resolved_newest_first_are_not_resolved_again()
    a, raw_b = register_case()
    pairs = {**inputs(), "register_case_raw": (a, raw_b)}
    for name, (one, other) in pairs.items():
        for rank in (0, 1):
            expected = first_divergence(load_records(one), load_records(other), rank=rank)
            assert "replaced_records" not in expected["runs"]["b"], name
            # Resolved once, or twice as `compare` gives them: the same result, byte for byte.
            twice = first_divergence(resolve_attempts(load_records(one)), resolve_attempts(load_records(other)), rank=rank)
            assert same(twice, expected), name
            # Rows with no `attempt` key at all (raw rows without it would be one attempt: not asked here).
            if name != "register_case_raw":
                bare = [[{k: v for k, v in row.items() if k != "attempt"} for row in rows] for rows in (one, other)]
                assert same(first_divergence(*bare, rank=rank), expected), name
            # Raw rows: what the resolved rows give (fails before lane AC4 on the raw register case).
            assert same(without_left_out(first_divergence(one, other, rank=rank)), without_left_out(expected)), name


def several_files_resolved_newest_first_are_not_resolved_again():
    # `compare` reads a rank's several files and resolves them in the attempts' own order, which
    # the rows alone do not show: here the restart's file is given first. Resolving those rows
    # again by first appearance would take the original attempt for the later one and drop every
    # row of the restart. They are left as they are.
    original = healthy(range(81), attempt=100)
    restart = [{**row, "ts": 5000.0 + row["step"]} for row in spiked(range(50, 120), attempt=200, start=90)]
    rows = resolve_attempts_across([restart, original])
    assert len(rows) == 120 and {row["attempt"] for row in rows} == {100, 200}
    found = first_divergence(make(), rows)
    assert found["runs"]["b"]["records"] == 120 and found["runs"]["b"]["last_step"] == 119
    assert "replaced_records" not in found["runs"]["b"]
    assert found["answer"] == "diverged" and found["divergence"]["step"] == 90
