"""Where two paired runs part (`tensorquill.divergence`).

Two groups. The first is made-up records and needs nothing but Python: each number of the rule has
a pair just inside it and a pair just outside it, so halving a threshold, dropping the sustain to
one record, letting short stretches count, or interpolating a missing step each fail a test here.
The second runs `examples/seeded_failures.py`'s toy Llama on CPU (under a minute in all, skipped
without `transformers`): a healthy run against each seeded run, against itself, and against
another seed.
"""

import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path

import pytest

import tensorquill
from tensorquill import divergence as dv
from tensorquill.divergence import divergence_summary, first_divergence

SEEDED = "model.layers.2.self_attn.k_proj.weight"  # the middle block of the 4-block toy
STEPS = 120


# ---- made-up runs ------------------------------------------------------------------------


def wave(step, scale=1.0, phase=0.0):
    """A series that moves at every step: about 0.3 * scale between two records."""
    return scale * (2.0 + math.sin(0.7 * step + phase) + 0.5 * math.sin(2.3 * step + phase))


def make(steps=range(STEPS), *, rank=0, run_id="a", loss=None, grads=None, layers=None, weights=None, moe=None, extra=None):
    """Step records as the capture writes them. Each argument maps a step to a value, per name."""
    loss = loss or (lambda s: wave(s))
    grads = grads if grads is not None else {"w1": lambda s: wave(s, 0.1, 1.0), "w2": lambda s: wave(s, 0.2, 2.0)}
    layers = layers if layers is not None else {
        "block.0": lambda s: wave(s, 0.5, 3.0), "block.1": lambda s: wave(s, 0.5, 4.0),
    }
    records = []
    for step in steps:
        norms = {name: fn(step) for name, fn in grads.items()}
        record = {
            "schema_version": 1, "run_id": run_id, "rank": rank, "attempt": 1, "step": step,
            "ts": 1000.0 + step + random.random(),
            "layers": {name: {"mean": 0.0, "std": fn(step), "abs_max": 3 * fn(step), "zero_frac": 0.0} for name, fn in layers.items()},
            "grad_norm": math.sqrt(sum(v * v for v in norms.values())) if norms else None,
            "grad_norms": norms,
            "scalars": {"train/loss": loss(step)},
        }
        if weights and step % 50 == 49:
            record["weight_norms"] = {name: fn(step) for name, fn in weights.items()}
        if moe:
            record["moe"] = {name: fn(step) for name, fn in moe.items()}
        if extra:
            record.update(extra(step))
        records.append(record)
    return records


def jitter(seed, size):
    """Run-to-run noise: a fixed pseudo-random number in [-size, size] for each step."""
    values = [random.Random(seed * 100003 + step).uniform(-size, size) for step in range(2000)]
    return lambda step: values[step]


def strict(result):
    """The result is JSON as it stands: no NaN, nothing that is not plain data."""
    return json.loads(json.dumps(result, allow_nan=False))


def with_grad(change):
    """Run B: run A with `w1`'s gradient norm changed by `change(step, value)`."""
    return make(run_id="b", grads={"w1": lambda s: change(s, wave(s, 0.1, 1.0)), "w2": lambda s: wave(s, 0.2, 2.0)})


def test_two_identical_runs_have_no_divergence_and_the_result_says_they_were_identical():
    result = strict(first_divergence(make(), make(run_id="b")))
    assert result["answer"] == "no_divergence" and result["divergence"] is None
    assert "exactly the same value" in result["reason"]
    assert result["first_exact_difference"] is None and result["early"]["identical"] is True
    assert result["common_steps"] == {"count": STEPS, "first": 0, "last": STEPS - 1}
    assert result["runs"]["a"]["run_id"] == "a" and result["runs"]["b"]["records"] == STEPS
    assert all(entry["status"] == "agrees" for entry in result["series"].values())
    assert result["series"]["grad_norms/w1"] == {
        "kind": "grad_norms", "name": "w1", "stat": None, "common_steps": STEPS, "status": "agrees",
        "early": result["series"]["grad_norms/w1"]["early"], "threshold": result["series"]["grad_norms/w1"]["threshold"],
    }
    assert result["series"]["grad_norms/w1"]["early"]["samples"] == 30


def test_a_change_in_one_parameter_is_named_with_its_step_and_the_values_around_it():
    b = with_grad(lambda s, v: v * 50 if s >= 70 else v)
    result = strict(first_divergence(make(), b))
    assert result["answer"] == "diverged"
    found = result["divergence"]
    assert (found["step"], found["series"], found["kind"], found["name"]) == (70, "grad_norms/w1", "grad_norms", "w1")
    assert found["before"] == {"step": 69, "a": wave(69, 0.1, 1.0), "b": wave(69, 0.1, 1.0)}
    assert found["at"] == {"step": 70, "a": wave(70, 0.1, 1.0), "b": 50 * wave(70, 0.1, 1.0)}
    assert found["sustained"] == 50 and found["out_share"] == 1.0 and found["non_finite"] is False
    # The total gradient norm holds w1, so it left at the same step; the parameter is named first.
    assert found["same_step_series"] == ["grad_norm"]
    assert [d["series"] for d in result["diverged_series"]] == ["grad_norms/w1", "grad_norm"]
    assert result["series"]["grad_norms/w2"]["status"] == "agrees"
    assert result["first_exact_difference"] == {"step": 70, "series": 2}
    assert "step 70" in result["reason"] and "gradient norm of w1" in result["reason"] and "not a cause" in result["reason"]
    assert "step 70" in divergence_summary(result)


def test_the_thresholds_used_are_in_the_result():
    assert first_divergence(make(), make())["thresholds"] == {
        "early_steps": 30, "out_factor": 12.0, "sustain": 3, "stay_share": 0.5, "move_floor": 0.01,
        "agree_ratio": 0.2, "min_early_samples": 2, "min_early_samples_when_not_identical": 15, "min_common_steps": 33,
    }


# ---- each number of the rule, just inside and just outside -------------------------------


def noisy_pair(offset_factor, start=70, length=None):
    """Two runs that differ early by at most 0.001 on `w1`, then by `offset_factor` times that."""
    noise = jitter(7, 0.001)
    largest = max(abs(noise(s)) for s in range(30))
    stop = STEPS if length is None else start + length

    def change(step, value):
        if start <= step < stop:
            return value + offset_factor * largest
        return value + (noise(step) if step < 30 else 0.0)

    return make(), with_grad(change)


def test_a_difference_under_twelve_times_the_early_one_is_agreement_and_over_it_is_not():
    inside = first_divergence(*noisy_pair(11.0))
    assert inside["answer"] == "no_divergence", inside["reason"]
    assert inside["early"]["identical"] is False
    outside = first_divergence(*noisy_pair(13.0))
    assert outside["answer"] == "diverged" and outside["divergence"]["step"] == 70
    assert outside["divergence"]["times_threshold"] == pytest.approx(13 / 12)


def test_halving_or_doubling_the_factor_changes_those_answers(monkeypatch):
    monkeypatch.setattr(dv, "OUT_FACTOR", 6.0)
    assert first_divergence(*noisy_pair(11.0))["answer"] == "diverged"
    monkeypatch.setattr(dv, "OUT_FACTOR", 24.0)
    assert first_divergence(*noisy_pair(13.0))["answer"] == "no_divergence"


def test_two_records_out_at_the_end_are_too_short_and_three_are_a_departure():
    two = first_divergence(*noisy_pair(100.0, start=STEPS - 2))
    assert two["answer"] == "no_divergence"
    assert two["series"]["grad_norms/w1"]["left_at_end"] == {"step": STEPS - 2, "records": 2}
    assert "when the runs ended" in two["reason"] and "too short to call" in two["reason"]
    three = first_divergence(*noisy_pair(100.0, start=STEPS - 3))
    assert three["answer"] == "diverged" and three["divergence"]["step"] == STEPS - 3
    assert three["divergence"]["sustained"] == 3


def test_a_sustain_of_one_would_call_those_two_records_a_departure(monkeypatch):
    monkeypatch.setattr(dv, "SUSTAIN", 1)
    assert first_divergence(*noisy_pair(100.0, start=STEPS - 2))["answer"] == "diverged"


def test_a_stretch_that_comes_back_is_counted_and_is_not_a_departure():
    # Out for 5 records from step 60, then back in agreement for the 55 that follow: 5 of 60.
    back = first_divergence(*noisy_pair(100.0, start=60, length=5))
    assert back["answer"] == "no_divergence"
    assert back["series"]["grad_norms/w1"]["came_back"] == {"stretches": 1, "first_step": 60, "longest": 5}
    assert "back again" in back["reason"]
    # Out for 40 of the 60 records from step 60 on: it stayed out.
    stayed = first_divergence(*noisy_pair(100.0, start=60, length=40))
    assert stayed["answer"] == "diverged" and stayed["divergence"]["step"] == 60
    assert stayed["divergence"]["out_share"] == pytest.approx(40 / 60)
    # Just under and just over half.
    assert first_divergence(*noisy_pair(100.0, start=60, length=29))["answer"] == "no_divergence"
    assert first_divergence(*noisy_pair(100.0, start=60, length=31))["answer"] == "diverged"


def test_without_the_stay_share_a_stretch_that_comes_back_would_be_a_departure(monkeypatch):
    monkeypatch.setattr(dv, "STAY_SHARE", 0.0)
    assert first_divergence(*noisy_pair(100.0, start=60, length=5))["answer"] == "diverged"


def test_a_later_stretch_that_stays_is_found_after_one_that_came_back():
    noise = jitter(7, 0.001)

    def change(step, value):
        return value + (1.0 if 40 <= step < 44 or step >= 100 else noise(step) if step < 30 else 0.0)

    result = first_divergence(make(), with_grad(change))
    assert result["answer"] == "diverged" and result["divergence"]["step"] == 100
    assert result["series"]["grad_norms/w1"]["came_back"]["first_step"] == 40


def identical_early_pair(share_of_movement):
    """Identical for 50 steps, then `w1` differs by a share of its own step-to-step movement."""
    a = make()
    movement = first_divergence(a, a)["series"]["grad_norms/w1"]["early"]["own_movement"]
    return a, with_grad(lambda s, v: v + share_of_movement * movement if s >= 50 else v)


def test_identical_early_runs_are_judged_by_the_series_own_movement():
    inside = first_divergence(*identical_early_pair(0.007))
    assert inside["answer"] == "no_divergence", inside["reason"]
    assert inside["first_exact_difference"]["step"] == 50  # the result still says where they first differ at all
    assert "first recorded value that differs at all is at step 50" in divergence_summary(inside)
    outside = first_divergence(*identical_early_pair(0.02))
    assert outside["answer"] == "diverged" and outside["divergence"]["step"] == 50
    assert outside["divergence"]["in_own_movement"] == pytest.approx(0.02)
    assert outside["divergence"]["times_threshold"] == pytest.approx(2.0)


def test_halving_or_raising_the_floor_changes_those_answers(monkeypatch):
    monkeypatch.setattr(dv, "MOVE_FLOOR", 0.005)
    assert first_divergence(*identical_early_pair(0.007))["answer"] == "diverged"
    monkeypatch.setattr(dv, "MOVE_FLOOR", 0.03)
    assert first_divergence(*identical_early_pair(0.02))["answer"] == "no_divergence"


def test_a_series_that_never_moved_and_then_differs_has_left():
    flat = {"w1": lambda s: 0.5, "w2": lambda s: wave(s, 0.2, 2.0)}
    b = make(run_id="b", grads={"w1": lambda s: 0.5 if s < 80 else 0.5001, "w2": flat["w2"]})
    result = strict(first_divergence(make(grads=flat), b))
    assert result["answer"] == "diverged" and result["divergence"]["series"] == "grad_norms/w1"
    assert result["divergence"]["step"] == 80 and result["divergence"]["times_threshold"] is None
    assert "did not move" in result["reason"]


def test_the_early_window_is_thirty_common_steps():
    # A change from step 30 is after the early steps and is found; one from step 29 is inside them.
    after = first_divergence(make(), with_grad(lambda s, v: v + 1.0 if s >= 30 else v))
    assert after["answer"] == "diverged" and after["divergence"]["step"] == 30
    assert after["early"] == {"steps": 30, "first": 0, "last": 29, "identical": True}
    inside = first_divergence(make(), with_grad(lambda s, v: v + 1.0 if s >= 29 else v))
    assert inside["answer"] == "no_divergence"
    # One early value 1.0 apart: that is then how far apart the runs normally are, and the line is 12 times it.
    assert inside["series"]["grad_norms/w1"]["status"] == "agrees"
    assert inside["series"]["grad_norms/w1"]["threshold"] == pytest.approx(12.0)


# ---- no answer ---------------------------------------------------------------------------


def unpaired(seed):
    return make(
        run_id=f"seed{seed}",
        loss=lambda s: wave(s, 1.0, seed), grads={"w1": lambda s: wave(s, 0.1, 10 * seed), "w2": lambda s: wave(s, 0.2, 20 * seed)},
        layers={"block.0": lambda s: wave(s, 0.5, 30 * seed), "block.1": lambda s: wave(s, 0.5, 40 * seed)},
    )


def test_runs_that_do_not_agree_early_get_no_answer_and_the_reason():
    result = strict(first_divergence(unpaired(1), unpaired(2)))
    assert result["answer"] == "no_answer"
    assert result["divergence"] is None and result["diverged_series"] == []
    assert "do not agree in their first 30 common steps" in result["reason"] and "paired runs" in result["reason"]
    assert result["agreement"]["paired"] is False and result["agreement"]["agreeing"] == 0
    assert result["agreement"]["median_ratio"] > 0.2
    # The zero fractions are 0 in both runs, as in any two runs: they do not make the runs a pair.
    assert result["series"]["layers/block.0/zero_frac"]["status"] == "agrees"
    assert result["agreement"]["series"] == len(result["series"]) - 4  # two blocks' mean and zero_frac do not move


def test_the_agreement_line_is_a_fifth_of_the_series_own_movement():
    def pair(share):
        a = make()
        shifted = {}
        for series_id, entry in first_divergence(a, a)["series"].items():
            shifted[series_id] = share * entry["early"]["own_movement"]
        b = make(
            run_id="b", loss=lambda s: wave(s) + shifted["scalars/train/loss"],
            grads={"w1": lambda s: wave(s, 0.1, 1.0) + shifted["grad_norms/w1"], "w2": lambda s: wave(s, 0.2, 2.0) + shifted["grad_norms/w2"]},
            layers={"block.0": lambda s: wave(s, 0.5, 3.0) + shifted["layers/block.0/std"], "block.1": lambda s: wave(s, 0.5, 4.0) + shifted["layers/block.1/std"]},
        )
        return first_divergence(a, b)

    assert pair(0.15)["answer"] == "no_divergence"
    assert pair(0.25)["answer"] == "no_answer"


def test_one_series_that_does_not_agree_early_is_not_comparable_and_the_rest_are_judged():
    b = make(run_id="b", grads={"w1": lambda s: wave(s, 0.1, 1.0), "w2": lambda s: wave(s, 0.2, 9.0)})
    result = first_divergence(make(), b)
    assert result["answer"] == "no_divergence"
    assert result["series"]["grad_norms/w2"]["status"] == "not_comparable"
    why = {entry["series"]: entry["why"] for entry in result["not_comparable"]}
    assert "do not agree on it early" in why["grad_norms/w2"] and "do not agree on it early" in why["grad_norm"]
    assert result["series"]["grad_norms/w1"]["status"] == "agrees"


def test_short_runs_get_no_answer_with_the_count_needed():
    short = first_divergence(make(range(32)), make(range(32)))
    assert short["answer"] == "no_answer"
    assert "32 steps in common and 33 are needed" in short["reason"]
    assert first_divergence(make(range(33)), make(range(33)))["answer"] == "no_divergence"


def test_runs_with_no_common_step_get_no_answer():
    result = strict(first_divergence(make(range(0, 50)), make(range(100, 150))))
    assert result["answer"] == "no_answer" and result["common_steps"]["count"] == 0
    assert "no step in common" in result["reason"] and "0 to 49" in result["reason"] and "100 to 149" in result["reason"]


def test_empty_runs_and_a_missing_rank_get_no_answer():
    assert "run B has no records of rank 0 (it has no records)" in first_divergence(make(), [])["reason"]
    result = first_divergence(make(), make(), rank=3)
    assert result["answer"] == "no_answer" and result["rank"] == 3
    assert "run A has no records of rank 3 (it has ranks [0])" in result["reason"]


def test_runs_that_share_no_series_get_no_answer():
    a = [{"step": s, "scalars": {"train/loss": wave(s)}} for s in range(STEPS)]
    b = [{"step": s, "scalars": {"eval/loss": wave(s)}} for s in range(STEPS)]
    result = first_divergence(a, b)
    assert result["answer"] == "no_answer" and "no series was recorded in both runs" in result["reason"]
    assert [entry["why"] for entry in result["not_comparable"]] == ["recorded in run B only", "recorded in run A only"]


# ---- what is compared --------------------------------------------------------------------


def test_a_series_in_one_run_only_is_listed_and_is_not_a_divergence():
    b = make(run_id="b", grads={"w1": lambda s: wave(s, 0.1, 1.0), "w2": lambda s: wave(s, 0.2, 2.0), "w3": lambda s: 5.0 * s})
    b = [{**record, "grad_norm": a_record["grad_norm"]} for record, a_record in zip(b, make())]
    result = first_divergence(make(), b)
    assert result["answer"] == "no_divergence"
    assert {"series": "grad_norms/w3", "why": "recorded in run B only"} in result["not_comparable"]
    assert "grad_norms/w3" not in result["series"]
    assert "1 series not comparable" in divergence_summary(result)


def test_only_steps_in_both_runs_are_compared_and_nothing_is_interpolated():
    # Run B recorded nothing from step 60 to 116. Run A's loss is 100 higher on steps 61 to 116
    # and back at 117: no step both runs have shows that, so there is nothing to report.
    a = make(loss=lambda s: wave(s) + (100.0 if 61 <= s <= 116 else 0.0))
    b = make([s for s in range(STEPS) if not 60 <= s <= 116], run_id="b")
    result = first_divergence(a, b)
    assert result["common_steps"] == {"count": 63, "first": 0, "last": STEPS - 1}
    assert result["series"]["scalars/train/loss"]["common_steps"] == 63
    assert result["answer"] == "no_divergence" and result["first_exact_difference"] is None

    # Values made up for B's missing steps, on the line between its neighbours, would be set
    # against A's real ones and give a step that neither run supports.
    by_step = {record["step"]: record for record in b}
    low, high = by_step[59]["scalars"]["train/loss"], by_step[117]["scalars"]["train/loss"]
    filled = [
        by_step.get(step) or {**by_step[59], "step": step, "scalars": {"train/loss": low + (high - low) * (step - 59) / 58}}
        for step in range(STEPS)
    ]
    made_up = first_divergence(a, filled)
    assert made_up["answer"] == "diverged" and made_up["divergence"]["step"] == 60  # the first made-up value


def test_the_early_steps_are_counted_in_common_steps():
    result = first_divergence(make(), make(range(0, STEPS, 2), run_id="b"))
    assert result["common_steps"] == {"count": STEPS // 2, "first": 0, "last": STEPS - 2}
    assert result["early"]["last"] == 58  # the 30th common step, not step 29
    assert result["answer"] == "no_divergence"


def test_the_value_before_is_the_last_common_step_not_the_step_before():
    a = make(range(0, 400))
    b = make(range(0, 400, 5), run_id="b", grads={"w1": lambda s: wave(s, 0.1, 1.0) * (9.0 if s >= 202 else 1.0), "w2": lambda s: wave(s, 0.2, 2.0)})
    found = first_divergence(a, b)["divergence"]
    assert found["step"] == 205 and found["before"]["step"] == 200 and found["at"]["step"] == 205


def test_wall_clock_readings_are_left_out_and_listed():
    def clock(offset):
        return lambda step: {
            "data_wait": offset + step, "step_time": offset * 2.0, "system": {"gpu": offset},
            "scalars": {"train/loss": wave(step), "perf/step_time_ms": offset * step, "tokens_per_sec": offset, "lr": 0.001},
            "probe": {"anything": offset}, "lr": [offset],
        }

    result = first_divergence(make(extra=clock(1.0)), make(extra=clock(50.0)))
    assert result["answer"] == "no_divergence"
    assert result["left_out"]["wall_clock"] == ["data_wait", "scalars/perf/step_time_ms", "scalars/tokens_per_sec", "step_time", "system", "ts"]
    assert result["left_out"]["not_compared"] == ["lr", "probe"]
    assert "scalars/lr" in result["series"] and "scalars/perf/step_time_ms" not in result["series"]
    assert "Left out as wall-clock readings: data_wait" in divergence_summary(result)


def test_a_scalar_is_a_clock_only_by_a_whole_word_of_its_name():
    clocks = ["perf/step_time_ms", "tokens_per_sec", "train/samples_per_second", "data.wait.ms", "Throughput", "gpu-mem", "epoch_time"]
    series = ["val_mse", "loss_mse", "train/beta", "eval/score", "train/step_loss", "metadata/x", "timestep_loss", "train/loss", "lr"]
    assert all(dv._is_wall_clock(name) for name in clocks)
    assert not any(dv._is_wall_clock(name) for name in series)
    # Two runs that part on such a scalar only are told apart.
    names = lambda step, factor: {"scalars": {name: wave(step, 1.0, i) * factor for i, name in enumerate(series)}}  # noqa: E731
    result = first_divergence(make(extra=lambda s: names(s, 1.0)), make(extra=lambda s: names(s, 3.0 if s >= 70 else 1.0)))
    assert result["left_out"]["wall_clock"] == ["ts"]
    assert sorted(d["series"] for d in result["diverged_series"]) == sorted(f"scalars/{name}" for name in series)
    assert result["divergence"]["step"] == 70


def test_one_rank_is_read_and_named():
    healthy = make() + make(rank=1)
    changed = make(run_id="b") + [
        {**record, "rank": 1} for record in with_grad(lambda s, v: v * 50 if s >= 70 else v)
    ]
    on_zero = first_divergence(healthy, changed)
    assert on_zero["rank"] == 0 and on_zero["answer"] == "no_divergence"
    assert on_zero["runs"]["b"]["records"] == STEPS
    on_one = first_divergence(healthy, changed, rank=1)
    assert on_one["rank"] == 1 and on_one["answer"] == "diverged" and on_one["divergence"]["step"] == 70
    assert "rank 1" in divergence_summary(on_one)
    # A record without a rank is rank 0, as `load_records` reads it.
    bare = [{k: v for k, v in record.items() if k != "rank"} for record in make()]
    assert first_divergence(bare, make())["answer"] == "no_divergence"


def test_a_step_written_twice_keeps_the_later_record():
    a = make()
    twice = make(run_id="b") + [record for record in a if record["step"] == 40]
    twice[40] = {**twice[40], "scalars": {"train/loss": 1e9}}
    assert first_divergence(a, twice)["answer"] == "no_divergence"


def test_at_one_step_the_first_block_in_the_captures_order_leads_then_the_loss_then_gradients():
    def block(phase, factor):
        return lambda s: wave(s, 0.5, phase) * (factor if s >= 60 else 1.0)

    # block.1 changes 3 times, block.2 after it 30 times, and the loss and both gradients with them.
    layers = {"block.0": block(3.0, 1.0), "block.1": block(4.0, 3.0), "block.2": block(5.0, 30.0)}
    b = make(
        run_id="b", layers=layers, loss=lambda s: wave(s) * (40.0 if s >= 60 else 1.0),
        grads={"w1": lambda s: wave(s, 0.1, 1.0) * (50.0 if s >= 60 else 1.0), "w2": lambda s: wave(s, 0.2, 2.0)},
    )
    a = make(layers={"block.0": block(3.0, 1.0), "block.1": block(4.0, 1.0), "block.2": block(5.0, 1.0)})
    result = first_divergence(a, b)
    assert result["divergence"]["name"] == "block.1" and result["divergence"]["kind"] == "layers"
    order = [d["series"].rsplit("/", 1)[0] if d["kind"] == "layers" else d["series"] for d in result["diverged_series"]]
    assert order == [
        "layers/block.1", "layers/block.1", "layers/block.2", "layers/block.2",
        "scalars/train/loss", "grad_norms/w1", "grad_norm",
    ]
    sizes = {d["series"]: d["in_own_movement"] for d in result["diverged_series"]}
    assert sizes["layers/block.2/std"] > 5 * sizes["layers/block.1/std"]  # the later block moved more and does not lead
    assert result["divergence"]["same_step"] == 6


def test_an_earlier_step_wins_over_a_larger_change():
    b = make(
        run_id="b", loss=lambda s: wave(s) * (40.0 if s >= 90 else 1.0),
        grads={"w1": lambda s: wave(s, 0.1, 1.0), "w2": lambda s: wave(s, 0.2, 2.0) * (1.5 if s >= 61 else 1.0)},
    )
    result = first_divergence(make(), b)
    assert (result["divergence"]["step"], result["divergence"]["series"]) == (61, "grad_norms/w2")
    assert [d["step"] for d in result["diverged_series"]] == [61, 61, 90]


def test_router_shares_are_compared():
    even = lambda s: [10 + (s + e) % 3 for e in range(8)]  # noqa: E731
    collapsed = lambda s: even(s) if s < 80 else [90, 0, 0, 0, 0, 0, 0, 0]  # noqa: E731
    result = first_divergence(make(moe={"router.0": even, "router.1": even}), make(run_id="b", moe={"router.0": even, "router.1": collapsed}))
    assert result["answer"] == "diverged" and result["divergence"]["step"] == 80
    assert result["divergence"]["kind"] == "moe" and result["divergence"]["name"] == "router.1"
    assert {d["series"] for d in result["diverged_series"]} == {"moe/router.1/max_share", "moe/router.1/dead_frac"}
    assert result["divergence"]["at"]["b"] in (1.0, 7 / 8)
    assert result["series"]["moe/router.0/max_share"]["status"] == "agrees"
    assert "moe_top_k" not in result["series"]


def test_a_series_recorded_every_fifty_steps_is_judged_when_the_runs_are_identical_early():
    steps = range(400)
    moving = {"w1": lambda s: 1.0 + 0.001 * s}
    frozen = {"w1": lambda s: 1.0 + 0.001 * min(s, 100)}  # stops changing after step 100
    result = first_divergence(make(steps, weights=moving), make(steps, run_id="b", weights=frozen))
    found = result["divergence"]
    assert (found["step"], found["series"], found["before"]["step"]) == (149, "weight_norms/w1", 99)
    assert result["series"]["weight_norms/w1"]["early"] == {
        "samples": 2, "last_step": 99, "largest_difference": 0.0, "typical_difference": 0.0,
        "own_movement": pytest.approx(0.05), "ratio": 0.0,
    }


def test_that_series_is_not_compared_when_the_runs_differ_early():
    steps = range(400)
    noise = jitter(3, 1e-6)
    moving = {"w1": lambda s: 1.0 + 0.001 * s}
    frozen = {"w1": lambda s: 1.0 + 0.001 * min(s, 100)}
    noisy = make(steps, run_id="b", weights=frozen, loss=lambda s: wave(s) + noise(s))
    result = first_divergence(make(steps, weights=moving), noisy)
    assert result["early"]["identical"] is False and result["answer"] == "no_divergence"
    entry = result["series"]["weight_norms/w1"]
    assert entry["status"] == "not_comparable"
    assert "0 of its samples fall in the early steps (up to step 29; 15 are needed)" in entry["why"]
    assert result["series"]["scalars/train/loss"]["status"] == "agrees"


# ---- values that are not finite ----------------------------------------------------------


def test_a_value_not_finite_in_one_run_is_a_departure_and_is_written_as_the_records_write_it():
    b = make(run_id="b", loss=lambda s: wave(s) if s < 70 else math.nan)
    result = first_divergence(make(), b)
    found = strict(result)["divergence"]
    assert (found["step"], found["series"], found["non_finite"]) == (70, "scalars/train/loss", True)
    assert found["at"] == {"step": 70, "a": wave(70), "b": "nan"}
    assert found["difference"] is None and found["size"] is None and found["times_threshold"] is None
    assert "not finite in one run only" in result["reason"]


def test_infinity_and_the_text_form_of_a_non_finite_value_are_read_the_same():
    as_float = make(run_id="b", loss=lambda s: wave(s) if s < 70 else math.inf)
    as_text = make(run_id="b", loss=lambda s: wave(s) if s < 70 else "inf")
    first, second = strict(first_divergence(make(), as_float)), strict(first_divergence(make(), as_text))
    assert first["divergence"] == second["divergence"] and first["divergence"]["at"]["b"] == "inf"
    opposite = make(loss=lambda s: wave(s) if s < 70 else -math.inf)
    assert first_divergence(opposite, as_float)["divergence"]["step"] == 70  # inf against -inf is a difference


def test_the_same_non_finite_value_in_both_runs_is_agreement():
    broken = lambda s: wave(s) if s < 70 else math.nan  # noqa: E731
    result = strict(first_divergence(make(loss=broken), make(run_id="b", loss=broken)))
    assert result["answer"] == "no_divergence" and result["first_exact_difference"] is None


def test_a_value_not_finite_in_one_run_in_the_early_steps_makes_the_series_not_comparable():
    b = make(run_id="b", loss=lambda s: math.nan if s == 5 else wave(s))
    result = strict(first_divergence(make(), b))
    assert result["answer"] == "no_divergence"
    entry = result["series"]["scalars/train/loss"]
    assert entry["status"] == "not_comparable" and "not finite in one run only in the early steps" in entry["why"]
    assert entry["early"]["largest_difference"] is None


def test_a_non_finite_stretch_that_ends_is_counted_as_one_that_came_back():
    b = make(run_id="b", loss=lambda s: math.nan if 60 <= s < 64 else wave(s))
    result = strict(first_divergence(make(), b))
    assert result["answer"] == "no_divergence"
    assert result["series"]["scalars/train/loss"]["came_back"] == {"stretches": 1, "first_step": 60, "longest": 4}


# ---- the toy Llama of examples/seeded_failures.py ----------------------------------------


def healthy_run(sf, path, steps, seed=0):
    """`seeded_failures.py --toy` with no fault: its model, optimizer groups, batches and capture."""
    import torch
    from tensorquill.capture import WEIGHT_EVERY

    args = sf.build_parser().parse_args(["--toy", "--out", str(path.parent), "--steps", str(steps)])
    device = torch.device("cpu")
    batches = sf.make_batches(args, device, steps)
    set_seed = torch.manual_seed
    torch.manual_seed = lambda _seed: set_seed(seed)  # `build_models` seeds with 0
    try:
        model = sf.build_models(args, device)[0][0]
    finally:
        torch.manual_seed = set_seed
    name = sf.seeded_parameter_name(model, args)
    optimizer, _group = sf.make_optimizer(model, dict(model.named_parameters())[name], args.lr, device)
    with tensorquill.init(
        model, run_id=path.stem, capture="full", every=sf.FLUSH_EVERY, weights=WEIGHT_EVERY,
        sink=tensorquill.JsonlSink(path),
    ) as session:
        for step in range(steps):
            sf.train_step(model, optimizer, batches[step], session)
    return tensorquill.load_records(path)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    pytest.importorskip("transformers")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import seeded_failures as sf

    root = tmp_path_factory.mktemp("divergence")
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        try:
            sf.main(["--toy", "--out", str(root / "seeded"), "--case", "both"])
        except SystemExit as exit:
            assert not exit.code, printed.getvalue()
        runs = {
            "healthy": healthy_run(sf, root / "healthy.jsonl", 350),
            "again": healthy_run(sf, root / "again.jsonl", 350),
            "other_seed": healthy_run(sf, root / "other-seed.jsonl", 350, seed=1),
        }
    for case in sf.CASES:
        runs[case] = tensorquill.load_records(root / "seeded" / case / f"{case}.jsonl")
    return runs


def test_toy_gradient_spike_is_named_at_the_seeded_step_on_the_seeded_parameter(toy):
    result = strict(first_divergence(toy["healthy"], toy["grad_spike"]))
    found = result["divergence"]
    assert result["answer"] == "diverged" and found["step"] == 250  # seeded from step 250
    assert (found["kind"], found["name"]) == ("grad_norms", SEEDED)
    assert found["before"]["step"] == 249 and found["before"]["a"] == found["before"]["b"]
    assert found["at"]["b"] > 1000 * found["at"]["a"]
    assert found["same_step_series"] == ["grad_norm"]  # the total holds that parameter
    assert result["first_exact_difference"] == {"step": 250, "series": 2}
    assert result["early"]["identical"] is True and result["agreement"]["median_ratio"] == 0.0


def test_toy_stalled_weights_are_named_within_one_check_of_the_freeze_on_the_seeded_parameter(toy):
    result = strict(first_divergence(toy["healthy"], toy["stalled_weights"]))
    found = result["divergence"]
    assert result["answer"] == "diverged"
    assert (found["kind"], found["name"]) == ("weight_norms", SEEDED)
    assert 100 <= found["step"] <= 150 and found["step"] == 149  # frozen from step 100; weights are checked every 50
    assert found["before"]["step"] == 99 and found["before"]["a"] == found["before"]["b"]
    assert found["at"]["b"] == found["before"]["b"] and found["at"]["a"] != found["before"]["a"]  # B stopped, A moved on
    # Other series differ from just after the freeze, by rounding: that is said, and is not the answer.
    assert 100 < result["first_exact_difference"]["step"] < 149


def test_toy_same_job_twice_has_no_divergence(toy):
    result = strict(first_divergence(toy["healthy"], toy["again"]))
    assert result["answer"] == "no_divergence" and result["first_exact_difference"] is None
    assert result["common_steps"]["count"] == 350 and result["not_comparable"] == []
    assert result["left_out"]["wall_clock"] == ["ts"]


def test_toy_another_seed_gets_no_answer_and_not_a_step(toy):
    result = strict(first_divergence(toy["healthy"], toy["other_seed"]))
    assert result["answer"] == "no_answer" and result["divergence"] is None and result["diverged_series"] == []
    assert "do not agree in their first 30 common steps" in result["reason"]
    assert result["agreement"]["median_ratio"] > 0.5  # the agreement line is 0.2; paired runs were at 0
    # The seeded runs are no pair for it either.
    assert first_divergence(toy["other_seed"], toy["grad_spike"])["answer"] == "no_answer"


def test_toy_the_seeded_runs_against_each_other_part_at_the_first_fault(toy):
    result = first_divergence(toy["grad_spike"], toy["stalled_weights"])
    assert result["answer"] == "diverged" and result["divergence"]["step"] == 149
    assert result["divergence"]["name"] == SEEDED
