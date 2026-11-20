"""Lane Z4 of the SDK-22 wave: the replica driver (`examples/replica_check.py` and its helper
`examples/replica_proof.py`), tested on synthetic records and status files.

The real writer and both readers are not in this lane's worktree, so the driver's own logic is
tested here: its argument checks on two, three and eight ranks; each assertion against the
contract's result objects (the fixtures' `expected`, and the reference reading of
`test_replica_contract.py` over synthetic records), each with a mutation that it must catch; and
contract section 7's two proofs (a delayed rank, a rank that never reports) on two and three
simulated ranks, in both modes: the file reader's looks and a stand-in server that acknowledges
lines and writes `watchdog.json`.

The tests at the end run the driver under torchrun. Two run here (they train and end NOT PROVEN,
the reader being absent); the rest need the trial merge (the writer's `replica` key and the
reader in `tensorquill.evidence`) and are skipped until it resolves.
"""

from __future__ import annotations

import copy
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import replica_proof as proof  # noqa: E402
from tensorquill.flight import offset_path, write_offset  # noqa: E402
from tensorquill.sink import JsonlSink  # noqa: E402
from test_replica_contract import CASES, reference  # noqa: E402

LAUNCH = "2f6c1a9e4b7d4c1e9a3f6b8d0e1c2a47"
OTHER_LAUNCH = "b09d7e6f5a4c4b3a8f2e1d0c9b8a7f65"
SDK = Path(__file__).resolve().parents[1]


def _env(world: int, rank: int = 0, **extra: str) -> dict[str, str]:
    return {"WORLD_SIZE": str(world), "RANK": str(rank), "LOCAL_RANK": str(rank), "LOCAL_WORLD_SIZE": str(world),
            **extra}


def _plan(world: int, *argv: str, rank: int = 0, cuda: int | None = None) -> proof.Plan:
    return proof.validate(["--cpu", "--out", "o", *argv] if cuda is None else ["--out", "o", *argv],
                          _env(world, rank), cuda)


def _case(name: str) -> dict:
    for file in ("incident_cases.json", "no_incident_cases.json"):
        cases = json.loads((CASES / file).read_text())["cases"]
        if name in cases:
            return cases[name]
    raise KeyError(name)


# --- Arguments: any world size ------------------------------------------------------------------


@pytest.mark.parametrize("world", [2, 3, 8])
def test_the_driver_validates_on_two_three_and_eight_ranks(world):
    plan = _plan(world)
    assert plan.group == list(range(world))
    assert plan.scheduled == list(range(8))
    assert {p.step for p in plan.plants} == {1, 3, 5}
    held = _plan(world, "--hold-rank", str(world - 1), "--steps", "12")
    assert (held.hold_rank, held.hold_from, held.release) == (world - 1, 2, "after-verdict")
    never = _plan(world, "--hold-rank", "1", "--release", "never")
    assert never.never_judged(2) and not never.never_judged(1)
    gpus = _plan(world, cuda=world)
    assert not gpus.cpu


def test_the_old_refusal_of_any_world_but_two_is_gone():
    # XC2: the driver refused WORLD_SIZE != 2 at argument validation.
    for world in (3, 4, 8, 16):
        assert _plan(world).world == world


@pytest.mark.parametrize("env,argv,says", [
    ({"WORLD_SIZE": "1", "RANK": "0"}, [], "group of one"),
    ({}, [], "torchrun"),
    ({**_env(8), "LOCAL_WORLD_SIZE": "4"}, [], "one node"),
    (_env(3), ["--plant", "3:1:0.weight:add"], "not one of the 3 ranks"),
    (_env(3), ["--plant", "1:9:0.weight:add"], "not a scheduled step"),
    (_env(3), ["--plant", "1:2:0.weight:add", "--replica-every", "2"], "not a scheduled step"),
    (_env(3), ["--plant", "1:1:layers.0:add"], "parameters are"),
    (_env(3), ["--plant", "1:1:0.weight:scale"], "kinds are"),
    (_env(3), ["--plant", "1:1:0.weight"], "RANK:STEP:PARAM:KIND"),
    (_env(3), ["--hold-rank", "3"], "not one of the 3 ranks"),
    (_env(3), ["--hold-rank", "1", "--steps", "6"], "no training after the release"),
    (_env(3), ["--hold-rank", "1", "--step-seconds", "0"], "--step-seconds above 0"),
    (_env(3), ["--mode", "server", "--endpoint", "grpc://x:1"], "--store, --status-interval"),
    (_env(3), ["--no-plants", "--plant", "1:1:0.weight:add"], "opposites"),
])
def test_bad_arguments_are_refused_before_any_process_group(env, argv, says):
    with pytest.raises(proof.UsageError, match=says):
        proof.validate(["--cpu", "--out", "o", *argv], env, None)


def test_cuda_needs_a_gpu_per_rank():
    with pytest.raises(proof.UsageError, match="8 visible CUDA GPUs; 2"):
        proof.validate(["--out", "o"], _env(8), 2)
    with pytest.raises(proof.UsageError, match="--cpu"):
        proof.validate(["--out", "o"], _env(2), None)


def test_the_default_plants_keep_the_two_controls_d1_will_need():
    # Two ranks: every plant is on rank 1, two ranks that disagree.
    assert {p.rank for p in _plan(2).plants} == {1}
    # Three or more: each planted step has one rank changed, one outlier among the rest.
    for world in (3, 8):
        plan = _plan(world)
        for step in {p.step for p in plan.plants}:
            assert len({p.rank for p in plan.plants_at(step)}) == 1


def test_the_three_kinds_move_the_fingerprints_they_should():
    """The model and the plants themselves, fingerprinted as the contract defines it (norm, sum,
    and the `name-v1` projection, from the reference reading, not from the writer): an added value
    moves the norm, a sign flip keeps it and moves the sum, a swap moves only the projection."""
    import torch

    from test_replica_contract import name_v1_seed, name_v1_weight

    model = proof.build_model(torch.device("cpu"))
    assert tuple(name for name, _ in model.named_parameters()) == proof.PARAM_NAMES
    base = proof.base_values(model)

    def fingerprint(*plants):
        proof.reset_and_plant(model, base, plants)
        out = {}
        for name, param in model.named_parameters():
            flat = param.detach().double().view(-1).tolist()
            seed = name_v1_seed(name)
            out[name] = [math.sqrt(sum(v * v for v in flat)), sum(flat),
                         sum(v * name_v1_weight(i, seed) for i, v in enumerate(flat))]
        return out

    clean = fingerprint()
    added = fingerprint(proof.Plant(1, 1, "0.weight", "add"))
    negated = fingerprint(proof.Plant(1, 1, "1.weight", "neg"))
    swapped = fingerprint(proof.Plant(1, 1, "1.weight", "swap"))
    assert added["0.weight"][0] != clean["0.weight"][0] and added["1.weight"] == clean["1.weight"]
    assert negated["1.weight"][0] == pytest.approx(clean["1.weight"][0]) and negated["1.weight"][1] != clean["1.weight"][1]
    assert swapped["1.weight"][:2] == pytest.approx(clean["1.weight"][:2])
    assert swapped["1.weight"][2] != pytest.approx(clean["1.weight"][2])
    assert fingerprint() == clean  # the reset undoes every plant


# --- Synthetic records --------------------------------------------------------------------------

_BASE = {"0.weight": [3.0, 1.5, 0.25], "0.bias": [0.5, 0.25, -0.125], "1.weight": [2.0, 1.0, 0.75]}
_SHAPES = {"0.weight": [8, 4], "0.bias": [8], "1.weight": [1, 8]}


def _record(plan: proof.Plan, rank: int, step: int, launch: str = LAUNCH, run_id: str = "sim") -> dict:
    params = {}
    for name, fp in _BASE.items():
        fp = list(fp)
        sig = str(1000 + len(name))
        for plant in plan.plants_at(step, rank):
            if plant.param != name:
                continue
            sig = "999"
            if plant.kind == "add":
                fp = [fp[0] + 0.1, fp[1] + 0.25, fp[2] + 0.3]
            elif plant.kind == "neg":
                fp = [fp[0], fp[1] - 2.0, fp[2] + 0.5]
            else:
                fp = [fp[0], fp[1], fp[2] + 0.2]
        params[name] = {"shape": _SHAPES[name], "dtype": "float32", "fp": fp, "sig": sig}
    return {"schema_version": 1, "run_id": run_id, "rank": rank, "attempt": 1795000000000 + rank * 7, "step": step,
            "replica": {"v": 1, "launch": launch, "group": plan.group, "device": "cpu", "projection": "name-v1",
                        "signature": "fold64-v1", "params": params}}


def _records(plan: proof.Plan, *, without: tuple = ()) -> list[dict]:
    return [_record(plan, rank, step) for rank in plan.group for step in plan.scheduled
            if (rank, step) not in without and not (rank == plan.hold_rank and plan.never_judged(step))]


@pytest.mark.parametrize("world", [2, 3, 8])
def test_a_result_with_every_planted_step_right_passes(world):
    plan = _plan(world)
    result = reference(_records(plan), 1e-6)
    assert proof.check_result(result, plan, LAUNCH) == []
    assert {i["step"] for i in result["replica_drift"]} == {1, 3, 5}


def test_a_run_with_no_plants_passes_only_when_nothing_is_raised():
    plan = _plan(3, "--no-plants")
    assert proof.check_result(reference(_records(plan), 1e-6), plan, LAUNCH) == []
    drifting = reference(_records(_plan(3)), 1e-6)
    assert proof.check_result(drifting, plan, LAUNCH)


def _mutate(result: dict, step: int, **change) -> dict:
    out = copy.deepcopy(result)
    for entry in out["steps"]:
        if entry["step"] == step:
            entry.update(change)
            for key, value in change.items():
                if value is None:
                    del entry[key]
    return out


def _drop_step(result: dict, step: int) -> dict:
    out = copy.deepcopy(result)
    out["steps"] = [e for e in out["steps"] if e["step"] != step]
    return out


def _incidents(result: dict, change) -> dict:
    out = copy.deepcopy(result)
    out["replica_drift"] = change(out["replica_drift"])
    return out


def _mutations(result: dict) -> dict[str, dict]:
    first = lambda xs: xs[0]  # noqa: E731
    return {
        "a clean step that is not listed": _drop_step(result, 0),
        "a planted step that is not listed": _drop_step(result, 3),
        "a clean step read as drift": _mutate(result, 2, fingerprints="drift", drift_parameters=["0.bias"]),
        "a clean step not compared": _mutate(result, 2, fingerprints="not_compared", fingerprints_why="device"),
        "a clean step whose signatures differ": _mutate(result, 2, signatures="differ"),
        "a planted step read as agree": _mutate(result, 1, fingerprints="agree", drift_parameters=None),
        "drift on another parameter as well": _mutate(result, 1, drift_parameters=["0.bias", "0.weight"]),
        "drift on the wrong parameter": _mutate(result, 1, drift_parameters=["1.weight"]),
        "signatures agree on a planted step": _mutate(result, 3, signatures="agree"),
        "a step left insufficient": _mutate(result, 4, coverage="insufficient", missing=[1], fingerprints=None,
                                            signatures=None),
        "a step in another group": _mutate(result, 4, group=[0, 1]),
        "an entry of another launch": {**result, "steps": result["steps"] + [{**result["steps"][0],
                                                                              "launch": OTHER_LAUNCH}]},
        "an entry that names no launch": {**result, "steps": result["steps"] + [
            {"launch": None, "group": [0, 1, 2], "step": 0, "coverage": "not_established", "why": "launch"}]},
        "rows set aside as a later attempt": {**result, "steps": result["steps"] + [
            {"launch": LAUNCH, "group": [0, 1, 2], "step": 0, "coverage": "not_established", "why": "attempt",
             "ranks": [1]}]},
        "an entry for a step not scheduled": {**result, "steps": result["steps"] + [
            {**result["steps"][0], "step": 40}]},
        "a planted step's incident missing": _incidents(result, lambda xs: [i for i in xs if i["step"] != 3]),
        "every incident missing": _incidents(result, lambda xs: []),
        "an incident on a clean step": _incidents(result, lambda xs: xs + [{**first(xs), "step": 2}]),
        "an incident twice": _incidents(result, lambda xs: xs + [first(xs)]),
        "an incident on the wrong parameter": _incidents(result, lambda xs: [{**first(xs), "parameters": ["0.bias"]},
                                                                             *xs[1:]]),
        "an incident of another launch": _incidents(result, lambda xs: [{**first(xs), "launch": OTHER_LAUNCH},
                                                                        *xs[1:]]),
        "an incident of another group": _incidents(result, lambda xs: [{**first(xs), "group": [0, 1]}, *xs[1:]]),
        "an incident from a gathered spread": _incidents(result, lambda xs: xs + [
            {"step": 1, "parameters": ["0.weight"], "source": "replica_spread"}]),
    }


@pytest.mark.parametrize("name", list(_mutations(reference(_records(_plan(3)), 1e-6))))
def test_each_assertion_catches_its_mutation(name):
    """Each mutation changes one thing in a result that passes; the check must name it. A check
    dropped from `check_result` turns one of these red."""
    plan = _plan(3)
    result = _mutations(reference(_records(plan), 1e-6))[name]
    assert proof.check_result(result, plan, LAUNCH), name


def test_a_launch_rank_0_does_not_know_is_a_failure():
    plan = _plan(3)
    result = reference(_records(plan), 1e-6)
    assert "no launch" in proof.check_result(result, plan, None)[0]
    assert proof.check_result(result, plan, OTHER_LAUNCH)  # every entry is of another start


def test_the_fixtures_expected_objects_read_as_the_contract_says():
    """The fixtures' `expected` objects, through a plan that says what was planted and held."""
    def plan_for(world, step, *plants, hold=None):
        plan = _plan(world, "--no-plants")
        plan.steps, plan.replica_every = step + 1, step + 1
        plan.plants = [proof.Plant(rank, step, param, "add") for rank, param in plants]
        if hold is not None:
            plan.hold_rank, plan.hold_from, plan.release = hold, step, "never"
        return plan

    assert proof.check_result(_case("drift_on_one_of_three")["expected"], plan_for(3, 99, (1, "w")), LAUNCH) == []
    assert proof.check_result(_case("drift_between_two_ranks")["expected"], plan_for(2, 99, (1, "b")), LAUNCH) == []
    assert proof.check_result(_case("healthy_peers_with_distinct_attempts")["expected"], plan_for(3, 99), LAUNCH) == []
    # Rank 2 never reported: insufficient naming it, and drift between the other two raises nothing.
    assert proof.check_result(_case("delayed_rank_before_it_arrives")["expected"],
                              plan_for(3, 99, hold=2), LAUNCH) == []
    assert proof.check_result(_case("drift_with_a_rank_missing")["expected"],
                              plan_for(3, 99, (1, "w"), hold=2), LAUNCH) == []
    never = plan_for(3, 99, hold=2)
    never.steps, never.replica_every = 200, 100
    assert proof.check_result(_case("rank_that_never_reports")["expected"], never, LAUNCH) == []
    # And the same objects fail a plan they do not fit.
    assert proof.check_result(_case("drift_on_one_of_three")["expected"], plan_for(3, 99), LAUNCH)
    assert proof.check_result(_case("delayed_rank_before_it_arrives")["expected"], plan_for(3, 99), LAUNCH)
    assert proof.check_result(_case("delayed_rank_before_it_arrives")["expected"], plan_for(3, 99, hold=1), LAUNCH)
    # A gathered spread raising beside the observations is not what the driver plants.
    assert proof.check_result(_case("explicit_spread_and_observations_on_one_step")["expected"],
                              plan_for(2, 99, (1, "w")), LAUNCH)


def test_a_rank_that_never_reports_must_raise_nothing_for_its_steps():
    plan = _plan(3, "--hold-rank", "2", "--release", "never")  # plants at 3 and 5 fall in the held range
    result = reference(_records(plan), 1e-6)
    assert proof.check_result(result, plan, LAUNCH) == []
    raised = _incidents(result, lambda xs: xs + [{"step": 3, "parameters": ["1.weight"], "source": "replica",
                                                  "launch": LAUNCH, "group": [0, 1, 2]}])
    assert proof.check_result(raised, plan, LAUNCH)
    judged = reference(_records(_plan(3)), 1e-6)  # every step complete, as if the rank had reported
    assert proof.check_result(judged, plan, LAUNCH)
    wrong_rank = _mutate(result, 4, missing=[1])
    assert proof.check_result(wrong_rank, plan, LAUNCH)


def test_check_kinds_reads_every_rank_and_names_no_rank():
    plan = _plan(3)
    records = _records(plan)
    failures, lines = proof.check_kinds(records, plan, LAUNCH)
    assert failures == []
    assert all("D1 held" in line for line in lines)
    assert not any("outlier" in line or "drifted" in line for line in lines)
    # A swap that also moved the sum, or an add that kept the norm, is not what was planted.
    for step, column in ((5, 1), (1, 0)):
        broken = copy.deepcopy(records)
        for record in broken:
            if record["step"] == step and record["rank"] == 2:
                param = plan.plants_at(step)[0].param
                record["replica"]["params"][param]["fp"][column] = _BASE[param][column] + (0.5 if column else 0.0)
        assert proof.check_kinds(broken, plan, LAUNCH)[0], step
    # One rank's record missing: no cross-rank reading.
    assert proof.check_kinds(_records(plan, without=((1, 3),)), plan, LAUNCH)[0]


# --- The verdict log ------------------------------------------------------------------------------


def test_the_log_keeps_steps_a_later_status_no_longer_lists():
    plan = _plan(2, "--no-plants")
    log = proof.VerdictLog()
    early = reference(_records(plan), 1e-6)
    log.add(10.0, early)
    log.add(11.0, {"steps": [e for e in early["steps"] if e["step"] >= 6], "replica_drift": []})
    assert proof.check_result(log.as_result(), plan, LAUNCH) == []
    assert proof.check_result({"steps": [e for e in early["steps"] if e["step"] >= 6], "replica_drift": []},
                              plan, LAUNCH)
    assert log.first_judged(LAUNCH, plan.group, 0) == 10.0


def test_the_log_keeps_the_newest_entry_and_the_first_time_of_each_state():
    plan = _plan(3, "--no-plants")
    log = proof.VerdictLog()
    insufficient = reference(_records(plan, without=((2, 4),)), 1e-6)
    log.add(1.0, insufficient)
    log.add(2.0, insufficient)
    log.add(3.0, reference(_records(plan), 1e-6))
    assert log.first(LAUNCH, plan.group, 4, ("insufficient", (2,))) == 1.0
    assert log.first_judged(LAUNCH, plan.group, 4) == 3.0
    assert log.entries[(LAUNCH, (0, 1, 2), 4, None)]["coverage"] == "complete"
    log.add(4.0, None)  # a status with no `replica` object is a look that saw nothing
    assert log.looks == 4


# --- The clock --------------------------------------------------------------------------------------


def _report(rank: int, steps, t: float, *, end_step: int = 100) -> dict:
    return {"rank": rank, "written": [{"step": s, "end": end_step * (s + 1), "t": t + s, "replica": True}
                                      for s in steps], "offsets": []}


def test_the_clock_runs_from_the_last_line_written_to_the_first_look_that_judges():
    plan = _plan(2, "--no-plants", "--steps", "2")
    log = proof.VerdictLog()
    reports = {0: _report(0, [0, 1], 100.0), 1: _report(1, [0, 1], 100.2)}
    result = reference(_records(plan), 1e-6)
    log.add(101.0, {"steps": [e for e in result["steps"] if e["step"] == 0], "replica_drift": []})
    log.add(101.5, result)
    failures, lines = proof.check_latency(log, plan, LAUNCH, reports)
    assert failures == []
    assert "+0.800 s" in lines[0] and "+0.300 s" in lines[1]  # 101.0 - 100.2, 101.5 - 101.2
    assert "accepted at" in lines[0] and "verdict seen at" in lines[0]
    late = proof.VerdictLog()
    late.add(100.2 + plan.interval + plan.margin + 0.01, result)
    assert "above" in proof.check_latency(late, plan, LAUNCH, reports)[0][0]
    missing = {0: reports[0], 1: _report(1, [0], 100.2)}
    assert "never seen written" in proof.check_latency(log, plan, LAUNCH, missing)[0][0]


def test_in_server_mode_the_clock_starts_at_the_acknowledgement():
    plan = _plan(2, "--no-plants", "--mode", "server", "--endpoint", "grpc://h:1", "--store", "s",
                 "--status-interval", "0.5")
    plan.steps = 1
    report = {"rank": 0, "written": [{"step": 0, "end": 120, "t": 10.0, "replica": True}],
              "offsets": [(10.1, 60), (10.4, 120), (10.6, 200)]}
    assert proof.arrival(report, 0, "server") == 10.4
    assert proof.arrival(report, 0, "files") == 10.0
    assert proof.arrival({**report, "offsets": [(10.1, 60)]}, 0, "server") is None
    log = proof.VerdictLog()
    log.add(11.5, reference(_records(plan), 1e-6))
    failures, lines = proof.check_latency(log, plan, LAUNCH, {0: report, 1: {**report, "rank": 1}})
    assert failures == [] and "+1.100 s" in lines[0] and "status interval 0.5" in lines[0]
    log = proof.VerdictLog()
    log.add(12.0, reference(_records(plan), 1e-6))  # 1.6 s after the ack: above 0.5 + 1.0
    assert proof.check_latency(log, plan, LAUNCH, {0: report, 1: {**report, "rank": 1}})[0]


def test_a_step_never_judged_has_no_clock():
    plan = _plan(2, "--no-plants", "--hold-rank", "1", "--release", "never")
    reports = {0: _report(0, plan.scheduled, 0.0), 1: _report(1, [0, 1], 0.0)}
    log = proof.VerdictLog()
    log.add(1.2, reference(_records(plan), 1e-6))
    failures, lines = proof.check_latency(log, plan, LAUNCH, reports)
    assert failures == []
    assert sum("never judged" in line for line in lines) == len(plan.scheduled) - 2


# --- The two proofs' history, branch by branch ------------------------------------------------------


def _delayed_history(plan: proof.Plan) -> proof.VerdictLog:
    """Looks at t=1 (step 2 on, insufficient without the held rank) and t=3 (every step judged)."""
    log = proof.VerdictLog()
    log.add(1.0, reference(_records(plan, without=tuple((plan.hold_rank, s) for s in plan.scheduled
                                                        if plan.held(s))), 1e-6))
    log.add(3.0, reference(_records(plan), 1e-6))
    return log


def test_the_delayed_proof_holds_branch_by_branch():
    plan = _plan(3, "--hold-rank", "2", "--steps", "12")
    log = _delayed_history(plan)
    held = {"released_at": 2.0, "released_at_close": False, "dropped": 0}
    assert proof.check_history(log, plan, LAUNCH, held, {2: 6}) == []
    cases = {
        "released only at close": ({**held, "released_at_close": True}, {2: 6}, "at its close"),
        "never released": ({**held, "released_at": None}, {2: 6}, "never released"),
        "judged before the release": ({**held, "released_at": 3.5}, {2: 6}, "before the release"),
        "seen insufficient after the release": ({**held, "released_at": 0.5}, {2: 6}, "after the release"),
        "no training past the step": (held, {2: 2}, "no training going on"),
        "no look while insufficient": (held, {}, "no training going on"),
    }
    for name, (report, seen, says) in cases.items():
        failures = proof.check_history(log, plan, LAUNCH, report, seen)
        assert any(says in f for f in failures), (name, failures)
    assert "left no report" in proof.check_history(log, plan, LAUNCH, None, {2: 6})[0]
    judged_only = proof.VerdictLog()
    judged_only.add(3.0, reference(_records(plan), 1e-6))
    assert any("never seen insufficient" in f for f in proof.check_history(judged_only, plan, LAUNCH, held, {2: 6}))


def test_the_never_proof_holds_branch_by_branch():
    plan = _plan(3, "--hold-rank", "2", "--release", "never")
    log = proof.VerdictLog()
    log.add(1.0, reference(_records(plan), 1e-6))
    held = {"released_at": None, "released_at_close": False, "dropped": 6}
    assert proof.check_history(log, plan, LAUNCH, held, {2: 7}) == []
    assert any("released" in f for f in proof.check_history(log, plan, LAUNCH, {**held, "released_at": 1.0}, {2: 7}))
    assert any("held nothing" in f for f in proof.check_history(log, plan, LAUNCH, {**held, "dropped": 0}, {2: 7}))
    assert proof.check_history(log, _plan(3), LAUNCH, None, {}) == []  # nothing held, nothing to prove


# --- Nothing waits ---------------------------------------------------------------------------------


def test_no_step_waits_while_a_rank_is_held():
    plan = _plan(3, "--hold-rank", "2", "--steps", "10", "--step-seconds", "0.1", "--wait-slack", "0.2")
    steady = {r: {"steps_done": 10, "durations": [0.11] * 10} for r in range(3)}
    assert proof.check_no_wait(steady, plan) == []
    waited = copy.deepcopy(steady)
    waited[1]["durations"][4] = 0.9  # a step on another rank that waited for the held rank
    assert "rank 1 step 4" in proof.check_no_wait(waited, plan)[0]
    before = copy.deepcopy(steady)
    before[1]["durations"][0] = 5.0  # the first step, before the hold, is not judged
    assert proof.check_no_wait(before, plan) == []
    short = copy.deepcopy(steady)
    short[2]["steps_done"] = 7
    assert "7 of 10" in proof.check_no_wait(short, plan)[0]
    assert "no report" in proof.check_no_wait({0: steady[0], 1: steady[1]}, plan)[0]


# --- The sink ----------------------------------------------------------------------------------------


def _lines(path: Path) -> list[int]:
    return [json.loads(line)["step"] for line in path.read_text().splitlines()]


def test_the_sink_logs_each_line_where_and_when_it_reached_the_file(tmp_path):
    plan = _plan(2)
    sink = proof.ProofSink(JsonlSink(tmp_path / "r.jsonl"))
    sink.write([_record(plan, 0, 0), _record(plan, 0, 1)])
    sink.close()
    assert [w["step"] for w in sink.written] == [0, 1]
    assert sink.written[-1]["end"] == (tmp_path / "r.jsonl").stat().st_size
    assert sink.written[0]["end"] < sink.written[1]["end"]
    assert sink.launch == LAUNCH and sink.report()["dropped"] == 0


def test_a_held_rank_is_released_by_its_timer_with_no_write_to_wait_for(tmp_path):
    plan = _plan(2)
    release = tmp_path / "release"
    progress = proof.Progress()
    sink = proof.ProofSink(JsonlSink(tmp_path / "r.jsonl"), hold_from=2, release_file=release, progress=progress,
                           poll=0.005)
    sink.write([_record(plan, 1, s) for s in range(5)])
    assert _lines(tmp_path / "r.jsonl") == [0, 1] and sink.holding
    progress.step = 7
    release.touch()
    deadline = time.time() + 5
    while sink.holding and time.time() < deadline:
        time.sleep(0.01)
    assert not sink.holding, "the release waited for a write"
    assert _lines(tmp_path / "r.jsonl") == [0, 1, 2, 3, 4]
    sink.write([_record(plan, 1, 5)])
    sink.close()
    sink.stop()
    report = sink.report()
    assert report["released_at_step"] == 7 and not report["released_at_close"]
    assert report["held_steps"] == [2, 3, 4] and report["written"][2]["t"] >= report["released_at"]


def test_a_rank_that_never_reports_drops_what_it_held(tmp_path):
    plan = _plan(2)
    sink = proof.ProofSink(JsonlSink(tmp_path / "r.jsonl"), hold_from=1, never=True, release_file=tmp_path / "release")
    (tmp_path / "release").touch()  # never means never, whatever the file says
    sink.write([_record(plan, 1, s) for s in range(4)])
    sink.close()
    sink.stop()
    assert _lines(tmp_path / "r.jsonl") == [0]
    assert sink.report()["dropped"] == 3 and sink.report()["released_at"] is None


def test_a_delayed_rank_not_released_by_its_close_says_so(tmp_path):
    plan = _plan(2)
    sink = proof.ProofSink(JsonlSink(tmp_path / "r.jsonl"), hold_from=1, release_file=tmp_path / "release")
    sink.write([_record(plan, 1, s) for s in range(3)])
    sink.close()
    sink.stop()
    assert _lines(tmp_path / "r.jsonl") == [0, 1, 2] and sink.report()["released_at_close"]


def test_the_sink_logs_the_servers_acknowledgements(tmp_path):
    path = tmp_path / "r.rank0.jsonl"
    sink = proof.ProofSink(JsonlSink(path), offsets=offset_path(path), poll=0.002)
    sink.write([_record(_plan(2), 0, 0)])
    write_offset(offset_path(path), 10)
    time.sleep(0.05)
    write_offset(offset_path(path), sink.written[0]["end"], {"server": "x"})  # the bound form
    sink.close()
    sink.stop()
    assert [o for _, o in sink.offsets] == [10, sink.written[0]["end"]]
    assert proof.arrival(sink.report(), 0, "server") == sink.offsets[1][0]


# --- Section 7's proofs on simulated ranks -------------------------------------------------------------


class FakeServer:
    """Stands in for the server. Every `interval` it writes `watchdog.json` with the `replica`
    object (by the reference reading) of what it had acknowledged at its previous tick, then
    acknowledges each rank file's whole lines (the sender's offset file). A verdict is thus
    visible one interval after its last observation was acknowledged, as the contract bounds it."""

    def __init__(self, work: Path, store: Path, run_id: str, world: int, interval: float) -> None:
        self.work, self.run_id, self.world, self.interval = work, run_id, world, interval
        self.status = store / "runs" / run_id / "watchdog.json"
        self.status.parent.mkdir(parents=True)
        self.acked: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _tick(self) -> None:
        if self.acked:
            scratch = self.status.with_name("w.tmp")
            scratch.write_text(json.dumps({"schema_version": 1, "ranks": [],
                                           "replica": {"v": 1, "tolerance": 1e-6, **reference(self.acked, 1e-6)}}))
            os.replace(scratch, self.status)
        acked = []
        for rank in range(self.world):
            path = proof.rank_file(self.work, self.run_id, rank)
            try:
                data = path.read_bytes()
            except OSError:
                continue
            end = data.rfind(b"\n") + 1
            acked += [json.loads(line) for line in data[:end].splitlines()]
            if end:
                write_offset(offset_path(path), end)
        self.acked = acked

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._tick()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._tick()
        time.sleep(self.interval)
        self._tick()


def _simulate(tmp_path: Path, plan: proof.Plan, *, hold: bool = True, block_on_hold: bool = False,
              server_interval: float | None = None) -> dict:
    """The driver's flow with each rank's training replaced by writing its synthetic record each
    step (the sink is called as the writer thread calls it). `hold=False` holds nothing although the
    plan says so; `block_on_hold` makes every rank's step wait until the held rank is released,
    as an SDK that waited for another rank's observation would."""
    run_id = "sim"
    work = tmp_path / run_id
    work.mkdir()
    progress = proof.Progress()
    sinks = {}
    for rank in plan.group:
        held = hold and rank == plan.hold_rank
        path = proof.rank_file(work, run_id, rank)
        sinks[rank] = proof.ProofSink(
            JsonlSink(path), hold_from=plan.hold_from if held else None, never=plan.release == "never",
            release_file=work / "release", progress=progress,
            offsets=offset_path(path) if plan.mode == "server" else None, poll=0.002)
    reader = (lambda records: reference(records, 1e-6)) if plan.mode == "files" else None
    observer = proof.Observer(plan, work, run_id, progress, lambda: sinks[0].launch, reader)
    durations = {rank: [] for rank in plan.group}
    server = (FakeServer(work, plan.store, run_id, plan.world, server_interval or plan.interval)
              if plan.mode == "server" else None)
    if server:
        server.__enter__()
    observer.start()
    try:
        for step in range(plan.steps):
            started = time.perf_counter()
            for rank in plan.group:
                sinks[rank].write([_record(plan, rank, step, run_id=run_id)])
            time.sleep(plan.step_seconds)
            if block_on_hold and step == plan.hold_from:
                held = sinks[plan.hold_rank]
                while held.holding and time.perf_counter() - started < plan.step_seconds + plan.wait_slack + 0.2:
                    time.sleep(0.005)
            for rank in plan.group:
                durations[rank].append(time.perf_counter() - started)
            progress.step = step
        for sink in sinks.values():
            sink.close()
    finally:
        if server:
            server.__exit__()
            time.sleep(0.05)
        for sink in sinks.values():
            sink.stop()
        observer.stop()
    observer.look()
    reports = {rank: {"rank": rank, "steps_done": plan.steps, "durations": durations[rank], **sinks[rank].report()}
               for rank in plan.group}
    launch = sinks[0].launch
    records = proof.run_records(work, run_id, plan.world)
    latency, lines = proof.check_latency(observer.log, plan, launch, reports)
    return {
        "result": proof.check_result(observer.log.as_result(), plan, launch),
        "history": proof.check_history(observer.log, plan, launch, reports.get(plan.hold_rank), observer.seen_while),
        "no_wait": proof.check_no_wait(reports, plan),
        "latency": latency, "lines": lines,
        "kinds": proof.check_kinds(records, plan, launch)[0],
        "reports": reports, "observer": observer,
    }


def _sim_plan(world: int, mode: str, release: str, tmp_path: Path) -> proof.Plan:
    argv = ["--hold-rank", str(world - 1), "--release", release, "--steps", "16", "--step-seconds", "0.05",
            "--wait-slack", "0.5", "--latency-margin", "1.0"]
    if mode == "server":
        argv += ["--mode", "server", "--endpoint", "grpc://127.0.0.1:1", "--store", str(tmp_path / "store"),
                 "--status-interval", "0.1"]
    else:
        argv += ["--look-seconds", "0.05"]
    return _plan(world, *argv)


def _failures(outcome: dict) -> list[str]:
    return [f for key in ("result", "history", "no_wait", "latency", "kinds") for f in outcome[key]]


@pytest.mark.parametrize("mode", ["files", "server"])
@pytest.mark.parametrize("world", [2, 3])
def test_a_delayed_rank_reads_insufficient_while_training_goes_on_then_is_judged(tmp_path, world, mode):
    plan = _sim_plan(world, mode, "after-verdict", tmp_path)
    outcome = _simulate(tmp_path, plan)
    assert _failures(outcome) == []
    held = outcome["reports"][plan.hold_rank]
    assert held["released_at"] is not None and not held["released_at_close"]
    assert held["released_at_step"] >= plan.hold_from + plan.hold_steps
    assert outcome["observer"].seen_while[plan.hold_from] > plan.hold_from
    assert any("verdict seen at" in line for line in outcome["lines"])


@pytest.mark.parametrize("mode", ["files", "server"])
@pytest.mark.parametrize("world", [2, 3])
def test_a_rank_that_never_reports_stays_insufficient_and_training_ends(tmp_path, world, mode):
    plan = _sim_plan(world, mode, "never", tmp_path)
    outcome = _simulate(tmp_path, plan)
    assert _failures(outcome) == []
    result = outcome["observer"].log.as_result()
    held_steps = [e for e in result["steps"] if e["step"] >= plan.hold_from]
    assert held_steps and all(e["coverage"] == "insufficient" and e["missing"] == [plan.hold_rank]
                              for e in held_steps)
    assert not [i for i in result["replica_drift"] if i["step"] >= plan.hold_from]
    assert outcome["reports"][plan.hold_rank]["dropped"] == plan.steps - plan.hold_from


@pytest.mark.parametrize("mode", ["files", "server"])
def test_the_proof_fails_when_nothing_was_held(tmp_path, mode):
    """The rank that should be held reports every step: the step is never seen insufficient."""
    plan = _sim_plan(3, mode, "after-verdict", tmp_path)
    outcome = _simulate(tmp_path, plan, hold=False)
    assert any("never seen insufficient" in f for f in outcome["history"])


def test_the_proof_fails_when_training_waits_for_the_held_rank(tmp_path):
    plan = _sim_plan(3, "files", "after-verdict", tmp_path)
    outcome = _simulate(tmp_path, plan, block_on_hold=True)
    assert outcome["no_wait"], "a step that waited for the release was not caught"


def test_the_never_proof_fails_when_the_rank_is_released(tmp_path):
    plan = _sim_plan(2, "files", "never", tmp_path)
    plan.release = "after-verdict"  # the sink releases; the checks still expect a rank that never reports
    outcome = _simulate(tmp_path, plan)
    plan.release = "never"
    observer = outcome["observer"]
    held = outcome["reports"][plan.hold_rank]
    launch = LAUNCH
    assert proof.check_history(observer.log, plan, launch, held, observer.seen_while)
    assert proof.check_result(observer.log.as_result(), plan, launch)


def test_a_status_with_the_verdict_too_late_fails_the_clock(tmp_path):
    plan = _sim_plan(2, "server", "after-verdict", tmp_path)
    plan.margin = 0.0
    plan.interval = 0.01  # the stand-in server shows a verdict 0.1 s after the ack: above 0.01 s
    outcome = _simulate(tmp_path, plan, server_interval=0.1)
    assert outcome["latency"] and all("above" in f or "never" in f for f in outcome["latency"])


# --- Under torchrun -------------------------------------------------------------------------------------

READER = proof.resolve_reader(proof.DEFAULT_READER)
needs_trial_merge = pytest.mark.skipif(
    READER is None,
    reason=f"needs the trial merge: the reader {proof.DEFAULT_READER} (lane Z2) and the writer's `replica` key (lane Z1)",
)


def _server_binary() -> Path | None:
    named = os.environ.get("TENSORQUILL_SERVER_BIN")
    if named:
        return Path(named)
    target = SDK.parent / "server" / "target"
    built = [p for p in (target / "debug" / "tensorquill-server", target / "release" / "tensorquill-server") if p.exists()]
    return max(built, key=lambda p: p.stat().st_mtime) if built else None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torchrun(world: int, *argv: str, timeout: float = 240) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SDK / "src")}
    env.pop("TENSORQUILL_ENDPOINT", None)
    return subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={world}", "--master-addr", "127.0.0.1",
         "--master-port", str(_free_port()), str(SDK / "examples" / "replica_check.py"), "--cpu", *argv],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=SDK,
    )


@pytest.mark.parametrize("world", [2, 3])
def test_the_driver_trains_to_its_end_under_torchrun(tmp_path, world):
    """Runs here: every rank trains every step with a rank held; without the reader the run ends
    NOT PROVEN (exit 3), never PASS. On the trial merge the same run must PASS."""
    done = _torchrun(world, "--out", str(tmp_path), "--hold-rank", str(world - 1), "--steps", "10",
                     "--step-seconds", "0.1")
    reports = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*/done.rank*.json"))]
    assert [r["steps_done"] for r in reports] == [10] * world, done.stdout + done.stderr
    if READER is None:
        assert "NOT PROVEN" in done.stdout and "exitcode  : 3" in done.stderr, done.stdout + done.stderr
        assert reports[-1]["held_steps"] == list(range(2, 10))
    else:
        assert done.returncode == 0 and f"PASS files on {world} ranks" in done.stdout, done.stdout + done.stderr


@needs_trial_merge
@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("hold", ["none", "after-verdict", "never"])
def test_trial_merge_files_mode(tmp_path, world, hold):
    argv = ["--out", str(tmp_path), "--steps", "16", "--step-seconds", "0.2"]
    if hold != "none":
        argv += ["--hold-rank", str(world - 1), "--release", hold]
    done = _torchrun(world, *argv)
    assert done.returncode == 0 and f"PASS files on {world} ranks" in done.stdout, done.stdout + done.stderr


@needs_trial_merge
@pytest.mark.skipif(_server_binary() is None, reason="tensorquill-server is not built")
@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("hold", ["none", "after-verdict", "never"])
def test_trial_merge_server_mode(tmp_path, world, hold):
    store = tmp_path / "store"
    server = subprocess.Popen(
        [str(_server_binary()), "--listen", "127.0.0.1:0", "--store", str(store), "--watchdog-seconds", "0.5",
         "--flush-seconds", "0.3"], stdout=subprocess.PIPE, text=True,
        env={k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"},
    )
    try:
        port = int(server.stdout.readline().rsplit(":", 1)[1])
        argv = ["--out", str(tmp_path / "out"), "--mode", "server", "--endpoint", f"grpc://127.0.0.1:{port}",
                "--store", str(store), "--status-interval", "0.5", "--steps", "24", "--step-seconds", "0.3"]
        if hold != "none":
            argv += ["--hold-rank", str(world - 1), "--release", hold]
        done = _torchrun(world, *argv)
        assert done.returncode == 0 and f"PASS server on {world} ranks" in done.stdout, done.stdout + done.stderr
    finally:
        server.terminate()
        server.wait(timeout=30)
