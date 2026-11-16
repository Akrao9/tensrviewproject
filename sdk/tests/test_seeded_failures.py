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
    assert not early.passed and "before the freeze could show" in early.summary
    cut = sf.judge_stalled_weights(stall(249, ["n"] * 20, stalled=29), [], "w", 100)
    assert not cut.passed and "9 stalled parameters are not listed" in cut.summary
    none = sf.judge_stalled_weights({"incidents": []}, [], "w", 100)
    assert not none.passed and none.summary.startswith("no stalled_weights incident")


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


@pytest.mark.xfail(
    strict=True,
    reason="detect_stalled_weights keeps ONE incident, from the first step anything is stalled (199, the RMSNorm "
           "weights), and lists 20 of them at most; a parameter that stalls later is not named",
)
def test_with_bfloat16_weights_the_incident_names_the_seeded_parameter(bfloat16_stall):
    assert bfloat16_stall["code"] == 0, bfloat16_stall["text"]
