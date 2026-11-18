"""The two seeded failures of `examples/seeded_failures.py`, through the real capture and detectors.

Two groups. The first needs nothing but PyTorch: the faults on a tiny module, the arithmetic that says
when a freeze can first be seen and what factor a spike needs, and the two judges on made-up incidents.
The second runs the script's toy Llama on CPU (about a minute in all) and skips without `transformers`.
Each judge is shown to fail as well as pass: a missing incident, the wrong parameter, a window missed,
and runs that seeded nothing.
"""

import contextlib
import io
import statistics
import sys
from pathlib import Path

import pytest
import torch

import tensorquill

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import seeded_failures as sf  # noqa: E402

SEEDED = "model.layers.2.self_attn.k_proj.weight"  # the middle block of the 4-block toy
OTHER = "model.layers.1.self_attn.k_proj.weight"


# ---- the faults, on a tiny module --------------------------------------------------------


def tiny():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def gradients(model, x):
    model.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def test_the_hook_scales_one_gradient_and_leaves_every_other_alone():
    model = tiny()
    x = torch.randn(16, 8)
    plain = gradients(model, x)
    scale = sf.GradientScale(model[0].weight, 250.0)
    assert all(torch.equal(plain[n], g) for n, g in gradients(model, x).items())  # off: nothing changes
    scale.on = True
    scaled = gradients(model, x)
    assert torch.allclose(scaled["0.weight"], 250.0 * plain["0.weight"])
    assert all(torch.equal(plain[n], scaled[n]) for n in plain if n != "0.weight")
    scale.remove()
    assert torch.equal(gradients(model, x)["0.weight"], plain["0.weight"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_zero_learning_rate_group_freezes_one_parameter_and_only_that_one(dtype):
    model = tiny().to(dtype)
    seeded = model[0].weight
    optimizer, group = sf.make_optimizer(model, seeded, 1e-2, torch.device("cpu"))
    x = torch.randn(16, 8, dtype=dtype)
    start = {n: p.detach().clone() for n, p in model.named_parameters()}

    def step():
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()

    step()
    assert not torch.equal(seeded, start["0.weight"])  # trains before the freeze
    group["lr"] = 0.0
    frozen = seeded.detach().clone()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(5):
        step()
        assert torch.equal(seeded, frozen)  # exactly unchanged, weight decay included
        assert seeded.grad is not None and torch.isfinite(seeded.grad).all() and seeded.grad.abs().sum() > 0
    assert all(not torch.equal(p, before[n]) for n, p in model.named_parameters() if p is not seeded)
    assert any(p is seeded for g in optimizer.param_groups for p in g["params"])  # still held


# ---- when a freeze can first be seen, and what factor a spike needs ----------------------


def first_stall_by_simulation(freeze_step, every=50, consecutive=3, horizon=3000):
    """Weight checks on steps every-1, 2*every-1, ...; the weight changes with each update up to
    `freeze_step - 1` and not after. The first step with `consecutive` unchanged checks in a row."""
    last, streak = None, 0
    for step in range(horizon):
        if (step + 1) % every:
            continue
        value = min(step, freeze_step - 1)
        streak = streak + 1 if last is not None and value == last else 0
        last = value
        if streak >= consecutive:
            return step


def test_first_visible_stall_matches_a_step_by_step_simulation():
    for freeze_step in range(1, 400):
        assert sf.first_visible_stall(freeze_step) == first_stall_by_simulation(freeze_step), freeze_step
    assert sf.first_visible_stall(100) == 249
    assert sf.first_visible_stall(101) == 299
    assert sf.first_visible_stall(1) == 199  # the first check has nothing to compare with
    assert sf.first_visible_stall(100, check_every=10) == first_stall_by_simulation(100, every=10)


@pytest.mark.parametrize("share", [0.0017, 0.01, 0.05, 0.3])
def test_the_chosen_factor_takes_the_total_to_about_twice_the_threshold(share):
    factor = sf.factor_for(share)
    reached = sf.total_ratio(factor, share)
    assert 2 * sf.SPIKE_FACTOR <= reached <= 1.12 * 2 * sf.SPIKE_FACTOR


def test_x100_on_a_small_parameter_does_not_move_the_total_and_an_unreachable_share_has_no_factor():
    assert sf.total_ratio(100, 0.0018) < 1.05  # the toy's k_proj: well under the detector's 5
    assert sf.total_ratio(100, 0.05) > sf.SPIKE_FACTOR - 0.2  # about where x100 starts to be enough
    assert sf.total_ratio(1, 0.3) == pytest.approx(1.0)
    assert sf.factor_for(0.0) == float("inf")


# ---- the judges, on made-up incidents ----------------------------------------------------


def spike(step, names=("a", "b"), count=2):
    parameters = [{"name": n, "ratio": 50.0 - i, "grad_norm": 1.0, "baseline": 0.02, "rising_since": step} for i, n in enumerate(names)]
    return {"incidents": [{"kind": "grad_spike", "step": step, "ratio": 9.0, "parameters": parameters,
                           "spike_count": count, "spike_steps": list(range(step, step + count))}]}


@pytest.mark.parametrize("step, passed", [(249, False), (250, True), (300, True), (301, False)])
def test_grad_spike_judge_window(step, passed):
    verdict = sf.judge_grad_spike(spike(step), [], "a", 250, 100.0)
    assert verdict.passed is passed
    assert verdict.summary.startswith("grad_spike")


def test_grad_spike_judge_needs_the_seeded_name_and_says_its_rank():
    assert sf.judge_grad_spike(spike(250), [], "c", 250, 100.0).passed is False
    assert "does not list c" in sf.judge_grad_spike(spike(250), [], "c", 250, 100.0).summary
    second = sf.judge_grad_spike(spike(250), [], "b", 250, 100.0)
    assert second.passed and "rank 2 of 2" in second.summary and "x100" in second.summary


def test_grad_spike_judge_without_an_incident_fails_and_says_so():
    verdict = sf.judge_grad_spike({"incidents": [{"kind": "stalled_weights"}]}, [], "a", 250, 100.0)
    assert not verdict.passed and verdict.summary.startswith("no grad_spike incident")


def stall(step, names, stalled=None, optimizer=None):
    return {"incidents": [{
        "kind": "stalled_weights", "step": step, "parameters": list(names),
        "stalled": len(names) if stalled is None else stalled, "with_gradients": 30, "consecutive": 3,
        "optimizer": optimizer or {"state": "known", "not_held": [], "not_held_count": 0, "held_count": len(names)},
    }]}


@pytest.mark.parametrize("step, passed", [(248, False), (249, True), (299, True), (300, False)])
def test_stalled_weights_judge_window(step, passed):
    assert sf.judge_stalled_weights(stall(step, ["w"]), [], "w", 100).passed is passed


def test_stalled_weights_judge_accepts_extras_but_needs_the_seeded_name():
    ok = sf.judge_stalled_weights(stall(249, ["n1", "w", "n2"]), [], "w", 100)
    assert ok.passed and "2 other" in ok.summary
    assert any("n1" in note for note in ok.notes)  # the extras are printed
    missing = sf.judge_stalled_weights(stall(249, ["n1", "n2"]), [], "w", 100)
    assert not missing.passed and "does not list w" in missing.summary


def test_stalled_weights_judge_names_why_a_stall_is_missing():
    early = sf.judge_stalled_weights(stall(199, ["n1"]), [], "w", 100)
    assert not early.passed and early.summary == "stalled_weights from step 199 does not list w"
    cut = sf.judge_stalled_weights(stall(249, ["n"] * 20, stalled=29), [], "w", 100)
    assert not cut.passed and "9 stalled parameter(s) and 0 onset(s) are not listed" in cut.summary
    none = sf.judge_stalled_weights({"incidents": []}, [], "w", 100)
    assert not none.passed and none.summary.startswith("no stalled_weights incident")


def with_onsets(first_step, *onsets, omitted=0):
    """A `stalled_weights` incident that began at `first_step`, with (step, names, trained_before) onsets."""
    evidence = stall(first_step, onsets[0][1])
    evidence["incidents"][0].update(
        onsets=[{"step": step, "parameters": list(names), "stalled": len(names), "trained_before": trained}
                for step, names, trained in onsets],
        stalled_total=sum(len(names) for _, names, _ in onsets), onsets_omitted=omitted,
    )
    return evidence


def test_stalled_weights_judge_reads_the_seeded_parameters_own_onset():
    # The incident began at step 199 with weights that never moved; the seeded one stalled at 249.
    verdict = sf.judge_stalled_weights(with_onsets(199, (199, ["n1", "n2"], 0), (249, ["w"], 1)), [], "w", 100)
    assert verdict.passed, verdict.summary
    assert "lists w at step 249" in verdict.summary and "after it changed at earlier checks" in verdict.summary
    assert "with 2 other parameter(s)" in verdict.summary
    assert any("step 199: 2 parameter(s), 0 of them changed at an earlier check" in note for note in verdict.notes)


@pytest.mark.parametrize("step, passed", [(199, False), (249, True), (299, True), (349, False)])
def test_stalled_weights_judge_window_is_on_the_onset_not_on_the_incident(step, passed):
    evidence = with_onsets(149, (149, ["n1"], 0), (step, ["w"], 1)) if step != 149 else with_onsets(149, (149, ["w"], 0))
    verdict = sf.judge_stalled_weights(evidence, [], "w", 100)
    assert verdict.passed is passed
    if not passed:
        assert f"lists w at step {step}, outside 249 to 299" in verdict.summary


def test_stalled_weights_judge_says_when_onsets_were_left_out():
    verdict = sf.judge_stalled_weights(with_onsets(199, (199, ["n1"], 0), omitted=3), [], "w", 100)
    assert not verdict.passed and "0 stalled parameter(s) and 3 onset(s) are not listed" in verdict.summary


def test_grad_spike_judge_reads_an_incident_raised_by_one_parameter():
    evidence = spike(250, names=("a",), count=0)
    evidence["incidents"][0].update(trigger="parameter", sustained=3, ratio=1.02)
    verdict = sf.judge_grad_spike(evidence, [], "a", 250, 100.0)
    assert verdict.passed and "the parameter's own norm, 3 steps in a row; total ratio 1.0" in verdict.summary
    evidence["incidents"][0]["ratio"] = None  # no history for the total yet
    assert "total ratio unknown" in sf.judge_grad_spike(evidence, [], "a", 250, 100.0).summary


def test_unchanged_streak_start_reads_the_capture_not_the_incident():
    def record(step, unchanged, grad=1.0):
        return {"step": step, "weights_unchanged": unchanged, "grad_norms": {"w": grad}}

    records = [{"step": 49, "grad_norms": {"w": 1.0}}, record(99, []), record(149, ["w"]), record(199, ["w"]), record(249, ["w"])]
    assert sf.unchanged_streak_start(records, "w") == 249
    assert sf.unchanged_streak_start(records[:4], "w") is None
    records[3] = record(199, ["w"], grad=0.0)  # a zero gradient restarts the streak
    assert sf.unchanged_streak_start(records, "w") is None


# ---- the toy Llama, through the script ---------------------------------------------------


def run_main(out, *argv):
    """`seeded_failures.main` on the toy Llama: its exit status and everything it printed."""
    buffer = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer):
        try:
            sf.main(["--toy", "--out", str(out), *argv])
        except SystemExit as exit:
            code = exit.code or 0
    return code, buffer.getvalue()


def evidence_of(out, case):
    records = tensorquill.load_records(Path(out) / case / f"{case}.jsonl")
    return records, tensorquill.build_evidence(records)


def incident(evidence, kind):
    return next(i for i in evidence["incidents"] if i["kind"] == kind)


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("seeded") / "out"
    code, text = run_main(out, "--case", "both")
    return {"out": out, "code": code, "text": text}


def test_both_cases_pass_and_the_script_exits_zero(both):
    assert both["code"] == 0, both["text"]
    assert "PASS: grad_spike" in both["text"] and "PASS: stalled_weights" in both["text"]
    assert "FAIL" not in both["text"] and "All cases passed" in both["text"]


def test_the_default_factor_is_raised_until_the_total_can_move(both):
    # x100 on the toy's k_proj holds 0.2% of the total norm: the script says so and picks the factor itself.
    assert "x100 on it moves the total x1.0" in both["text"] and "using x" in both["text"]
    records, evidence = evidence_of(both["out"], "grad_spike")
    spike_incident = incident(evidence, "grad_spike")
    assert spike_incident["step"] == 250
    assert spike_incident["ratio"] >= sf.SPIKE_FACTOR
    assert [p["name"] for p in spike_incident["parameters"]] == [SEEDED]  # no other gradient moved
    assert spike_incident["parameters"][0]["ratio"] > 1000
    assert spike_incident["parameters"][0]["rising_since"] == 250
    assert sf.judge_grad_spike(evidence, records, SEEDED, 250, 100.0).passed


def test_the_stall_is_found_at_the_first_step_three_checks_can_show_it(both):
    records, evidence = evidence_of(both["out"], "stalled_weights")
    stalled = incident(evidence, "stalled_weights")
    assert stalled["step"] == sf.first_visible_stall(100) == 249
    assert stalled["parameters"] == [SEEDED]
    assert stalled["optimizer"]["state"] == "known" and SEEDED not in stalled["optimizer"]["not_held"]
    assert sf.unchanged_streak_start(records, SEEDED) == 249
    norms = {r["step"]: r["weight_norms"] for r in records if "weight_norms" in r}
    assert sorted(norms) == [49, 99, 149, 199, 249, 299]
    assert norms[49][SEEDED] != norms[99][SEEDED]  # it trained until the freeze
    assert len({norms[s][SEEDED] for s in (99, 149, 199, 249, 299)}) == 1  # then not at all
    assert norms[149][OTHER] != norms[199][OTHER]  # and the others kept moving
    grads = [r["grad_norms"][SEEDED] for r in records if r["step"] >= 100]
    assert all(g > 0 and g < float("inf") for g in grads)  # the gradient stays finite and non-zero


def test_the_judges_fail_for_a_different_parameter(both):
    for case, judge in (("grad_spike", lambda e, r: sf.judge_grad_spike(e, r, OTHER, 250, 100.0)),
                        ("stalled_weights", lambda e, r: sf.judge_stalled_weights(e, r, OTHER, 100))):
        records, evidence = evidence_of(both["out"], case)
        verdict = judge(evidence, records)
        assert not verdict.passed and f"does not list {OTHER}" in verdict.summary


def test_a_run_that_seeds_no_spike_fails_and_exits_one(tmp_path_factory):
    pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("noseed") / "out"
    code, text = run_main(out, "--case", "grad_spike", "--factor", "1", "--seed-step", "100", "--steps", "200")
    assert code == 1
    assert "FAIL: no grad_spike incident" in text and "PASS" not in text


def test_a_run_that_freezes_nothing_fails(tmp_path_factory):
    pytest.importorskip("transformers")
    args = sf.build_parser().parse_args(["--toy", "--out", "unused", "--freeze-step", "1000", "--steps", "200"])
    out = tmp_path_factory.mktemp("nofreeze")
    verdict = sf.run_case("stalled_weights", args, torch.device("cpu"), sf.make_batches(args, torch.device("cpu"), 200), out / "case")
    assert not verdict.passed and verdict.summary.startswith("no stalled_weights incident")


def test_a_freeze_off_the_default_boundary_is_found_where_the_arithmetic_says(tmp_path_factory):
    pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("late") / "out"
    code, text = run_main(out, "--case", "stalled_weights", "--freeze-step", "101")
    assert code == 0, text
    records, evidence = evidence_of(out, "stalled_weights")
    assert incident(evidence, "stalled_weights")["step"] == sf.first_visible_stall(101) == 299
    assert sf.unchanged_streak_start(records, SEEDED) == 299  # one step later and it waits a whole interval


def test_too_few_steps_is_refused_before_any_training(tmp_path):
    code, _ = run_main(tmp_path / "out", "--case", "stalled_weights", "--steps", "200")
    assert code == 2 and not (tmp_path / "out").exists()


@pytest.fixture(scope="module")
def bfloat16_stall(tmp_path_factory):
    pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("bf16") / "out"
    code, text = run_main(out, "--case", "stalled_weights", "--dtype", "bfloat16")
    records, evidence = evidence_of(out, "stalled_weights")
    return {"code": code, "text": text, "records": records, "evidence": evidence}


def test_with_bfloat16_weights_the_capture_sees_the_freeze_whatever_the_incident_says(bfloat16_stall):
    # At lr 1e-5 bf16 weights near 1.0 never change, so RMSNorm weights stall without any seeding.
    records = bfloat16_stall["records"]
    assert sf.unchanged_streak_start(records, SEEDED) == 249
    stalled = incident(bfloat16_stall["evidence"], "stalled_weights")
    assert stalled["stalled"] >= 2 and any("norm" in name for name in stalled["parameters"])  # the background
    assert stalled["optimizer"]["state"] == "known"
    assert stalled["weight_dtypes"] == {"bfloat16": len(records[0]["grad_norms"])}


def test_with_bfloat16_weights_the_incident_names_the_seeded_parameter(bfloat16_stall):
    # Until Nov 16, 2026 this was a strict xfail: one incident, from the first step anything stalled
    # (199, the RMSNorm weights), named what stalled then and nothing that stalled later.
    assert bfloat16_stall["code"] == 0, bfloat16_stall["text"]
    stalled = incident(bfloat16_stall["evidence"], "stalled_weights")
    assert stalled["step"] == 199 and SEEDED not in stalled["parameters"]  # the incident still begins with the background
    first, own = stalled["onsets"][0], next(o for o in stalled["onsets"] if SEEDED in o["parameters"])
    assert first["step"] == 199 and first["trained_before"] == 0 and all("norm" in n for n in first["parameters"])
    assert (own["step"], own["parameters"], own["stalled"], own["trained_before"]) == (249, [SEEDED], 1, 1)
    assert own["optimizer"] == {"state": "known", "not_held": [], "not_held_count": 0, "held_count": 1}
    assert stalled["stalled_total"] == sum(o["stalled"] for o in stalled["onsets"]) and stalled["onsets_omitted"] == 0
    summary = tensorquill.incident_summary(bfloat16_stall["evidence"])
    assert f"step 249: {SEEDED} (after changing at earlier checks)" in summary
    assert "none of them has changed since the first weight check" in summary and "the weights are bfloat16" in summary


@pytest.fixture(scope="module")
def small_spike(tmp_path_factory):
    pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("x100") / "out"
    code, text = run_main(out, "--case", "grad_spike", "--factor", "100")
    records, evidence = evidence_of(out, "grad_spike")
    return {"code": code, "text": text, "records": records, "evidence": evidence}


def test_x100_on_one_small_parameter_is_reported_although_the_total_does_not_move(small_spike):
    # The case as first specified. Until Nov 16, 2026 nothing was reported: only the total was judged.
    assert small_spike["code"] == 0, small_spike["text"]
    spike_incident = incident(small_spike["evidence"], "grad_spike")
    assert (spike_incident["step"], spike_incident["trigger"], spike_incident["sustained"]) == (250, "parameter", 3)
    assert [p["name"] for p in spike_incident["parameters"]] == [SEEDED]
    assert 50 < spike_incident["parameters"][0]["ratio"] < 200 and spike_incident["parameters"][0]["rising_since"] == 250
    assert spike_incident["ratio"] < 1.5 and spike_incident["spike_count"] == 0  # the total stayed where it was
    assert "Gradient spike in one parameter from step 250" in tensorquill.incident_summary(small_spike["evidence"])


def test_x100_is_not_reported_before_it_has_lasted_three_steps(small_spike):
    records = small_spike["records"]
    for last, found in ((250, False), (251, False), (252, True)):
        prefix = [r for r in records if r["step"] <= last]
        kinds = [i["kind"] for i in tensorquill.build_evidence(prefix)["incidents"]]
        assert ("grad_spike" in kinds) is found, last


# ---- the update size: the same seeded gradient, read from the optimizer's state ----------------


@pytest.fixture(scope="module")
def update_case(tmp_path_factory):
    pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("update") / "out"
    code, text = run_main(out, "--case", "update_spike")
    records = tensorquill.load_records(out / "update_spike" / "update_spike.jsonl")
    return {"code": code, "text": text, "records": records}


def test_the_seeded_gradient_is_found_as_a_jump_in_the_update_size_at_its_step_on_its_parameter(update_case):
    assert update_case["code"] == 0, update_case["text"]
    assert f"PASS: update_spike at step 250, seeded from step 250, names {SEEDED} first" in update_case["text"]
    records = update_case["records"]
    # Every step carries every parameter's update size, read from AdamW's state.
    assert len(records) == 350 and all(SEEDED in r["update_rms"] and len(r["update_rms"]) == len(r["grad_norms"]) for r in records)
    evidence = tensorquill.build_evidence(records)  # the detector's own defaults
    found = incident(evidence, "update_spike")
    assert (found["step"], found["parameters"][0]["name"], found["spike_steps"][0]) == (250, SEEDED, 250)
    assert evidence["kind_checks"]["update_spike"]["enough_history"] is True and "thresholds" not in evidence
    # Cut before the baseline is full, the run is too short to judge, and says so.
    short = tensorquill.build_evidence(records[:80])
    assert not [i for i in short["incidents"] if i["kind"] == "update_spike"]
    assert short["kind_checks"]["update_spike"]["enough_history"] is False


def test_the_update_judge_fails_for_another_step_or_another_parameter(update_case):
    evidence = tensorquill.build_evidence(update_case["records"])
    assert sf.judge_update_spike(evidence, SEEDED, 250, 1e4).passed
    assert not sf.judge_update_spike(evidence, SEEDED, 249, 1e4).passed
    assert not sf.judge_update_spike(evidence, "model.layers.0.self_attn.k_proj.weight", 250, 1e4).passed
    none = tensorquill.build_evidence(update_case["records"][:240])
    verdict = sf.judge_update_spike(none, SEEDED, 250, 1e4)
    assert not verdict.passed and "no update_spike incident" in verdict.summary


def test_a_seed_step_inside_the_baseline_is_refused_before_any_training(tmp_path):
    code, _ = run_main(tmp_path / "out", "--case", "update_spike", "--seed-step", "50")
    assert code == 2 and not (tmp_path / "out").exists()
