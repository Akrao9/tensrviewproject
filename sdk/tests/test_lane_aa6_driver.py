"""Lane AA6 of wave AA: the replica driver's two bits-only plants, its count of signatures that
differ on healthy steps, and a stand-in reader shaped as the real one.

The author decided on Nov 20, 2026 that a difference in signatures alone stays evidence and raises
nothing, and that the eight-GPU run measures how often signatures differ on healthy steps and
whether a byte difference below the fingerprint tolerance is seen, signed zero included. So the
driver plants two more divergences (`ulp`: one element moved by the smallest step of its dtype;
`negzero`: a 0.0 made -0.0) and asserts, from the cross-rank result: coverage complete,
fingerprints `agree`, signatures `differ` on exactly the planted parameter, and no `replica_drift`.
It also counts, and writes into `verdict.json`, the healthy steps whose signatures differed.

The synthetic records and helpers are lane Z4's (`test_lane_z4_driver.py`); the torchrun tests at
the end need the server binary for server mode (`TENSORQUILL_SERVER_BIN`).
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import replica_proof as proof  # noqa: E402
from tensorquill.distributed import fold64_finish, fold64_sum  # noqa: E402
from tensorquill.evidence import compare_replicas  # noqa: E402
from test_lane_z4_driver import (  # noqa: E402
    LAUNCH, SDK, _case, _free_port, _incidents, _mutate, _passing, _plan, _record, _records, _server_binary,
    _stand_in_compare_replicas,
)
from test_replica_attribution_contract import attribution  # noqa: E402
from test_replica_contract import name_v1_seed, name_v1_weight, reference  # noqa: E402


# --- The plants -------------------------------------------------------------------------------------


@pytest.mark.parametrize("world", [2, 3, 8])
def test_the_two_bits_only_plants_are_in_the_defaults_with_the_three_kept(world):
    plan = _plan(world)
    kinds = {p.kind: p for p in plan.plants}
    assert set(kinds) == {"add", "neg", "swap", "ulp", "negzero"}
    assert {p.step for p in plan.plants if p.kind in ("add", "neg", "swap")} == {1, 3, 5}
    assert (kinds["ulp"].step, kinds["negzero"].step) == (2, 4)
    assert kinds["ulp"].quiet and kinds["negzero"].quiet and not kinds["add"].quiet
    # The controls stay: two ranks, every plant on rank 1; three or more, one rank per planted step.
    if world == 2:
        assert {p.rank for p in plan.plants} == {1}
    for step in {p.step for p in plan.plants}:
        assert len({p.rank for p in plan.plants_at(step)}) == 1
    assert [s for s in plan.scheduled if not plan.plants_at(s)] == [0, 6, 7]
    assert max(p.step for p in plan.plants) < 6  # six steps still run the defaults, as before
    one = _plan(world, "--plant", "1:6:1.weight:negzero")
    assert one.plants == [proof.Plant(1, 6, "1.weight", "negzero")] and one.quiet_at(6) == ["1.weight"]


def test_a_bits_only_plant_beside_a_moving_one_on_one_step_is_refused():
    with pytest.raises(proof.UsageError, match="on one step"):
        _plan(3, "--plant", "1:2:0.bias:ulp", "--plant", "2:2:0.weight:add")
    # Two bits-only plants on one step are allowed, as are the two kinds on separate steps.
    assert len(_plan(3, "--plant", "1:2:0.bias:ulp", "--plant", "1:2:0.weight:negzero").plants) == 2
    with pytest.raises(SystemExit):  # argparse refuses the choice
        proof.validate(["--cpu", "--healthy-mismatch", "maybe"], {"WORLD_SIZE": "2", "RANK": "0"}, None)
    assert _plan(2).healthy_mismatch == "report" and _plan(2, "--healthy-mismatch", "fail").healthy_mismatch == "fail"


def _model_and_base():
    model = proof.build_model(torch.device("cpu"))
    base = proof.base_values(model)
    proof.reset_and_plant(model, base, ())  # as the driver does before training
    return model, base


def _bits(t: torch.Tensor) -> list[int]:
    return t.detach().reshape(-1).view(torch.int32).tolist()


def _observe(model) -> dict[str, tuple[list[float], int]]:
    """Each parameter's fingerprints as the contract defines them (norm, sum, the `name-v1`
    projection, from the reference reading) and its `fold64-v1` signature by the writer's own
    functions."""
    out = {}
    for name, param in model.named_parameters():
        flat = param.detach().double().reshape(-1).tolist()
        seed = name_v1_seed(name)
        fp = [math.sqrt(sum(v * v for v in flat)), sum(flat), sum(v * name_v1_weight(i, seed) for i, v in enumerate(flat))]
        out[name] = (fp, fold64_finish(int(fold64_sum(param.detach())), "float32", list(param.shape)))
    return out


def test_the_base_holds_an_exact_positive_zero_and_the_three_kinds_still_move_what_they_should():
    model, base = _model_and_base()
    for name, value in base.items():
        flat = value.reshape(-1)
        assert flat[proof.ZERO_INDEX].item() == 0.0 and not torch.signbit(flat[proof.ZERO_INDEX]).item(), name
        assert (flat[:proof.ZERO_INDEX] > 0).all() and len(set(flat[:proof.ZERO_INDEX].tolist())) == flat.numel() - 1
    # The three kinds touch elements 0 to 2, away from the zero, and still do what they always did.
    clean = _observe(model)
    for plant, column, moves in ((proof.Plant(1, 1, "0.weight", "add"), 0, True),
                                 (proof.Plant(1, 1, "1.weight", "neg"), 1, True),
                                 (proof.Plant(1, 1, "1.weight", "swap"), 1, False)):
        proof.reset_and_plant(model, base, [plant])
        after = _observe(model)[plant.param][0]
        assert (abs(after[column] - clean[plant.param][0][column]) > 1e-3) == moves, plant
    proof.reset_and_plant(model, base, [])
    assert _observe(model) == clean


def test_negzero_turns_the_zero_negative_leaves_every_fingerprint_equal_and_changes_the_signature():
    model, base = _model_and_base()
    clean = _observe(model)
    clean_bits = {n: _bits(p) for n, p in model.named_parameters()}
    for name in proof.PARAM_NAMES:
        proof.reset_and_plant(model, base, [proof.Plant(1, 4, name, "negzero")])
        param = dict(model.named_parameters())[name]
        flat = param.detach().reshape(-1)
        assert flat[proof.ZERO_INDEX].item() == 0.0 and torch.signbit(flat[proof.ZERO_INDEX]).item()
        changed = [i for i, (a, b) in enumerate(zip(_bits(param), clean_bits[name])) if a != b]
        assert changed == [flat.numel() - 1]
        fp, sig = _observe(model)[name]
        assert fp == clean[name][0]  # exactly, not approximately
        assert sig != clean[name][1]
        proof.reset_and_plant(model, base, [])
        assert not torch.signbit(dict(model.named_parameters())[name].detach().reshape(-1)[proof.ZERO_INDEX]).item()


def test_ulp_moves_one_element_by_one_step_with_every_fingerprint_inside_the_tolerance():
    model, base = _model_and_base()
    clean = _observe(model)
    clean_bits = {n: _bits(p) for n, p in model.named_parameters()}
    for name in proof.PARAM_NAMES:
        proof.reset_and_plant(model, base, [proof.Plant(1, 2, name, "ulp")])
        param = dict(model.named_parameters())[name]
        diffs = [(i, a - b) for i, (a, b) in enumerate(zip(_bits(param), clean_bits[name])) if a != b]
        assert diffs == [(0, 1)], diffs  # element 0, one up in its bit pattern
        fp, sig = _observe(model)[name]
        scale = (fp[0] + clean[name][0][0]) / 2
        spreads = [abs(a - b) / scale for a, b in zip(fp, clean[name][0])]
        assert all(s <= proof.TOLERANCE for s in spreads), spreads
        assert spreads[1] > 0  # the sum did see it, far below the tolerance
        assert sig != clean[name][1]
    proof.reset_and_plant(model, base, [])
    assert _observe(model) == clean


# --- The verdict from the cross-rank result ---------------------------------------------------------


@pytest.mark.parametrize("world", [2, 3, 8])
def test_a_result_reading_the_bits_only_plants_as_the_contract_says_passes(world):
    plan = _plan(world)
    result = reference(_records(plan), 1e-6)
    assert proof.check_result(result, plan, LAUNCH) == []
    by_step = {e["step"]: e for e in result["steps"]}
    for step, param in ((2, "0.bias"), (4, "0.weight")):
        entry = by_step[step]
        assert (entry["coverage"], entry["fingerprints"], entry["signatures"], entry["signature_parameters"]) == (
            "complete", "agree", "differ", [param])
    assert {i["step"] for i in result["replica_drift"]} == {1, 3, 5}


_quiet_names = ["it raised replica_drift", "it read as drift", "its fingerprints were not compared",
                "its signatures agree", "its signatures were not compared", "another parameter differs too",
                "the wrong parameter differs"]
# A bits-only step left insufficient is caught by the check as it was before this lane
# (test_lane_z4_driver.py, "a step left insufficient"), so it is not repeated here.


def _quiet_mutations(result: dict, step: int) -> dict[str, tuple[dict, str]]:
    planted = next(e for e in result["steps"] if e["step"] == step).get("signature_parameters") or ["0.bias"]
    incident = {"step": step, "parameters": planted, "source": "replica", "launch": LAUNCH, "group": [0, 1, 2]}
    return {
        "it raised replica_drift": (_incidents(result, lambda xs: xs + [incident]), "raised for a bits-only plant"),
        "it read as drift": (_mutate(result, step, fingerprints="drift", drift_parameters=planted),
                             "fingerprints agree is due"),
        "its fingerprints were not compared": (_mutate(result, step, fingerprints="not_compared",
                                                       fingerprints_why="device"), "fingerprints agree is due"),
        "its signatures agree": (_mutate(result, step, signatures="agree", signature_parameters=None), "was not seen"),
        "its signatures were not compared": (_mutate(result, step, signatures="not_compared",
                                                     signatures_why="method", signature_parameters=None),
                                             "was not seen"),
        "another parameter differs too": (_mutate(result, step, signature_parameters=sorted(planted + ["1.weight"])),
                                          "not exactly the planted"),
        "the wrong parameter differs": (_mutate(result, step, signature_parameters=["1.weight"]),
                                        "not exactly the planted"),
    }


@pytest.mark.parametrize("step", [2, 4])
@pytest.mark.parametrize("name", _quiet_names)
def test_a_bits_only_plant_that_raises_or_is_not_seen_fails_with_its_reason(step, name):
    plan = _plan(3)
    mutations = _quiet_mutations(reference(_records(plan), 1e-6), step)
    assert set(mutations) == set(_quiet_names), "the parametrized names and the mutations drifted apart"
    mutated, says = mutations[name]
    failures = proof.check_result(mutated, plan, LAUNCH)
    assert any(says in f and f"step {step}" in f for f in failures), (name, failures)


def test_the_fixture_of_a_signature_only_difference_reads_as_a_bits_only_plant():
    def plan_for(kind):
        plan = _plan(2, "--no-plants")
        plan.steps, plan.replica_every = 100, 100
        plan.plants = [proof.Plant(1, 99, "w", kind)]
        return plan

    signature_only = _case("signature_only_disagreement")["expected"]
    assert proof.check_result(signature_only, plan_for("ulp"), LAUNCH) == []
    assert proof.check_result(signature_only, plan_for("negzero"), LAUNCH) == []
    assert proof.check_result(signature_only, plan_for("add"), LAUNCH)  # a moving plant must drift
    assert any("fingerprints agree is due" in f
               for f in proof.check_result(_case("drift_between_two_ranks")["expected"], plan_for("ulp"), LAUNCH))


def test_check_kinds_reads_the_bits_only_plants_from_every_rank():
    plan = _plan(3)
    records = _records(plan)
    failures, lines = proof.check_kinds(records, plan, LAUNCH)
    assert failures == []
    assert any("step 2 0.bias (ulp" in line and "spreads" in line for line in lines)
    assert any("step 4 0.weight" in line and "2 distinct signature(s)" in line for line in lines)

    def broken(step, change):
        out = copy.deepcopy(records)
        for record in out:
            if record["step"] == step and record["rank"] == plan.plants_at(step)[0].rank:
                change(record["replica"]["params"][plan.plants_at(step)[0].param])
        return proof.check_kinds(out, plan, LAUNCH)[0]

    assert any("bits were not seen" in f for f in broken(2, lambda p: p.update(sig=str(1000 + len("0.bias")))))
    assert any("bits were not seen" in f for f in broken(4, lambda p: p.update(sig=str(1000 + len("0.weight")))))
    assert any("must leave every fingerprint equal" in f for f in broken(4, lambda p: p["fp"].__setitem__(1, 1.5 + 1e-12)))
    assert any("within the tolerance" in f for f in broken(2, lambda p: p["fp"].__setitem__(1, 0.25 + 1e-3)))


# --- The count of healthy steps -----------------------------------------------------------------------


def test_the_healthy_count_is_zero_when_no_clean_step_differs():
    plan = _plan(3)
    healthy = proof.healthy_signatures(reference(_records(plan), 1e-6), plan, LAUNCH)
    assert healthy == {"steps_without_plant": 3, "compared": 3, "signatures_differ": 0, "by_parameter": {},
                       "not_compared": {}, "not_judged": [], "on_mismatch": "report"}


def _clean_differs(result: dict, step: int, params: list[str]) -> dict:
    return _mutate(result, step, signatures="differ", signature_parameters=params)


def test_a_clean_step_whose_signatures_differ_is_counted_and_reported_not_failed():
    """Before lane AA6 this was a failure of the run whatever was asked; now it is counted, and
    fails only with `--healthy-mismatch fail`."""
    plan = _plan(3)
    result = reference(_records(plan), 1e-6)
    result = _clean_differs(_clean_differs(result, 6, ["0.bias", "1.weight"]), 7, ["1.weight"])
    assert proof.check_result(result, plan, LAUNCH) == []
    healthy = proof.healthy_signatures(result, plan, LAUNCH)
    assert (healthy["compared"], healthy["signatures_differ"]) == (3, 2)
    assert healthy["by_parameter"] == {"0.bias": 1, "1.weight": 2}
    strict = _plan(3, "--healthy-mismatch", "fail")
    failures = proof.check_result(result, strict, LAUNCH)
    assert [f.split(":")[0] for f in failures] == ["step 6", "step 7"]
    assert all("--healthy-mismatch is fail" in f for f in failures)


def test_steps_not_compared_or_never_judged_are_counted_apart():
    plan = _plan(3)
    result = reference(_records(plan), 1e-6)
    result = _mutate(result, 0, signatures="not_compared", signatures_why="method")
    result = _mutate(result, 7, coverage="insufficient", missing=[2], fingerprints=None, signatures=None)
    healthy = proof.healthy_signatures(result, plan, LAUNCH)
    assert (healthy["compared"], healthy["not_compared"], healthy["not_judged"]) == (1, {"0": "method"}, [7])
    never = _plan(3, "--hold-rank", "2", "--release", "never")
    healthy = proof.healthy_signatures(reference(_records(never), 1e-6), never, LAUNCH)
    assert (healthy["compared"], healthy["not_judged"]) == (1, [6, 7])  # step 0 only, before the hold
    nothing = proof.healthy_signatures({"steps": [], "replica_drift": []}, plan, LAUNCH)
    assert (nothing["compared"], nothing["not_judged"]) == (0, [0, 6, 7])


def test_the_driver_prints_the_count_and_carries_it_in_its_verdict():
    plan, log, reports, records = _passing()
    verdict = proof.evaluate(plan, log, {}, reports, LAUNCH, records, reader_found=True)
    assert verdict.code == proof.EXIT_PASS, verdict.lines
    assert verdict.healthy["compared"] == 3 and verdict.healthy["signatures_differ"] == 0
    [line] = [x for x in verdict.lines if "healthy steps:" in x]
    assert "differed on 0 of 3 compared steps" in line and "reported, not failed" in line
    # A differing clean step: still PASS by default, with the count; FAIL when asked.
    differ = proof.VerdictLog()
    differ.add(100.5, _clean_differs(log.as_result(), 6, ["0.bias"]))
    verdict = proof.evaluate(plan, differ, {}, reports, LAUNCH, records, reader_found=True)
    assert verdict.code == proof.EXIT_PASS and verdict.healthy["by_parameter"] == {"0.bias": 1}
    plan.healthy_mismatch = "fail"
    assert proof.evaluate(plan, differ, {}, reports, LAUNCH, records, reader_found=True).code == proof.EXIT_FAIL


# --- The stand-in reader ------------------------------------------------------------------------------


def _legacy_record() -> dict:
    return {"schema_version": 1, "run_id": "sim", "rank": 0, "attempt": 1795000000000, "step": 50,
            "scalars": {"train/loss": 1.0}, "replica_spread": {"0.weight": [0.5, 0.0, 0.0], "0.bias": [0.0, 0.0, 0.0]}}


@pytest.mark.parametrize("which", ["three ranks", "two ranks", "a rank missing", "with a legacy record"])
def test_the_stand_in_reader_gives_what_the_real_compare_replicas_gives(which):
    """Before lane AA6 the stand-in added `v` and `tolerance`, which the real reader does not give."""
    plan = _plan(2 if which == "two ranks" else 3)
    records = _records(plan, without=((2, 4),) if which == "a rank missing" else ())
    if which == "with a legacy record":
        records = records + [_legacy_record()]
    real = compare_replicas(records, tolerance=1e-6)
    # D1 (contract 4a): `attribution` is held to the second reading of section 4a. Since lane AB4 the
    # stand-in carries the key too (wave AB's seam: lane AB1 wrote this test for a stand-in without
    # it), so the two results are compared whole.
    assert real.get("attribution", []) == attribution(records, 1e-6)
    stand_in = _stand_in_compare_replicas(records, tolerance=1e-6)
    assert set(stand_in) == set(real) and "v" not in real and "tolerance" not in real
    assert ("legacy" in real) == (which == "with a legacy record")
    assert stand_in == real


# --- Under torchrun -----------------------------------------------------------------------------------


def _run(world: int, *argv: str, extra_path: Path | None = None) -> tuple[subprocess.CompletedProcess, dict]:
    out = Path(argv[argv.index("--out") + 1])
    path = [str(extra_path)] if extra_path else []
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path + [str(SDK / "src")])}
    env.pop("TENSORQUILL_ENDPOINT", None)
    done = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={world}", "--master-addr", "127.0.0.1",
         "--master-port", str(_free_port()), str(SDK / "examples" / "replica_check.py"), "--cpu", *argv],
        capture_output=True, text=True, timeout=240, env=env, cwd=SDK,
    )
    found = [json.loads(p.read_text()) for p in out.glob("*/verdict.json")]
    assert len(found) == 1, done.stdout + done.stderr
    return done, found[0]


def _assert_bits_only_steps_read(stdout: str) -> None:
    assert "step 2: complete fingerprints agree signatures differ on ['0.bias']" in stdout, stdout
    assert "step 4: complete fingerprints agree signatures differ on ['0.weight']" in stdout, stdout
    raised = [line for line in stdout.splitlines() if "replica_drift: {" in line]
    # The file reader writes `step` first and the server writes it last: match either spelling.
    assert len(raised) == 3 and not any(re.search(r"'step': [24][,}]", x) for x in raised), raised


@pytest.mark.parametrize("world", [2, 3])
def test_files_mode_sees_both_bits_only_plants_and_counts_the_healthy_steps(tmp_path, world):
    done, verdict = _run(world, "--out", str(tmp_path), "--steps", "8", "--step-seconds", "0.1")
    assert (verdict["code"], verdict["failures"]) == (proof.EXIT_PASS, []), done.stdout + done.stderr
    assert verdict["healthy_signatures"] == {"steps_without_plant": 3, "compared": 3, "signatures_differ": 0,
                                             "by_parameter": {}, "not_compared": {}, "not_judged": [],
                                             "on_mismatch": "report"}
    _assert_bits_only_steps_read(done.stdout)
    assert "healthy steps: signatures differed on 0 of 3 compared steps" in done.stdout


_WRONG_READER = '''
from tensorquill.evidence import compare_replicas


def raises_on_bits(records, *, tolerance=1e-6, attempt_order=None):
    """The real reader, then an incident for the step of the `ulp` plant, as a reader that raised
    on a signature-only difference would."""
    out = compare_replicas(records, tolerance=tolerance)
    for entry in out["steps"]:
        if entry["step"] == 2 and entry.get("coverage") == "complete":
            out["replica_drift"].append({"step": 2, "parameters": ["0.bias"], "source": "replica",
                                         "launch": entry["launch"], "group": entry["group"]})
    return out


def blind_to_bits(records, *, tolerance=1e-6, attempt_order=None):
    """The real reader, with the signatures of the `negzero` step read as agree."""
    out = compare_replicas(records, tolerance=tolerance)
    for entry in out["steps"]:
        if entry["step"] == 4 and entry.get("signatures") == "differ":
            entry["signatures"] = "agree"
            entry.pop("signature_parameters", None)
    return out
'''


@pytest.mark.parametrize("reader,says", [("raises_on_bits", "step 2: replica_drift raised for a bits-only plant"),
                                         ("blind_to_bits", "step 4: negzero on 0.weight")])
def test_a_reader_that_raises_on_a_bits_only_plant_or_misses_it_fails_the_run(tmp_path, reader, says):
    (tmp_path / "wrong_reader.py").write_text(_WRONG_READER)
    done, verdict = _run(2, "--out", str(tmp_path / "out"), "--steps", "8", "--step-seconds", "0.1",
                         "--reader", f"wrong_reader:{reader}", extra_path=tmp_path)
    assert verdict["code"] == proof.EXIT_FAIL and done.returncode == proof.EXIT_FAIL, done.stdout + done.stderr
    assert any(says in f for f in verdict["failures"]), verdict["failures"]
    if reader == "blind_to_bits":
        assert any("was not seen" in f for f in verdict["failures"]), verdict["failures"]


@pytest.mark.skipif(_server_binary() is None, reason="tensorquill-server is not built")
@pytest.mark.parametrize("world", [2, 3])
def test_server_mode_sees_both_bits_only_plants_and_counts_the_healthy_steps(tmp_path, world):
    store = tmp_path / "store"
    server = subprocess.Popen(
        [str(_server_binary()), "--listen", "127.0.0.1:0", "--store", str(store), "--watchdog-seconds", "0.5",
         "--flush-seconds", "0.3"], stdout=subprocess.PIPE, text=True,
        env={k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"},
    )
    try:
        port = int(server.stdout.readline().rsplit(":", 1)[1])
        done, verdict = _run(world, "--out", str(tmp_path / "out"), "--mode", "server", "--endpoint",
                             f"grpc://127.0.0.1:{port}", "--store", str(store), "--status-interval", "0.5",
                             "--steps", "24", "--step-seconds", "0.3")
        assert (verdict["code"], verdict["failures"]) == (proof.EXIT_PASS, []), done.stdout + done.stderr
        healthy = verdict["healthy_signatures"]
        assert (healthy["steps_without_plant"], healthy["compared"], healthy["signatures_differ"]) == (19, 19, 0)
        _assert_bits_only_steps_read(done.stdout)
    finally:
        server.terminate()
        server.wait(timeout=30)
