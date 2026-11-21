"""Lane AB4 of wave AB: the replica driver checks section 4a (D1, which rank a drift is attributed
to) on each planted step, and has a second arm whose weights really change.

D1. From the result's `attribution` key (the top level of what
`tensorquill.evidence:compare_replicas` returns in files mode; the same key in the `replica` object
of `watchdog.json` in server mode), on each planted step that was judged:

- two ranks that disagree: `inconclusive`, why `two_ranks`, no rank named;
- three or more with one planted outlier: by fingerprints `attributed`, `outliers` exactly it;
- the bits-only plants (`ulp`, `negzero`): no entry by fingerprints, and by signatures the planted
  rank named where a strict majority of the records hold one signature.

When no look carried the key and an entry is due, the run fails saying "attribution not present".
The key is not built on `main` when this lane is written (lanes AB1 and AB2 build it), so the checks
are tested here with a stand-in: the real reader, then the reference function `attribution` of
`test_replica_attribution_contract.py`. The end-to-end pass with the real readers waits for the
merge with AB1 and AB2; the tests that run the real reader say which way they expect it to go.

The learning arm. `--arm learning`: SGD with `--lr` above 0, no reset, no plant, a batch of its own
on each rank. The run fails unless every rank's weights changed from their start, and its count of
healthy steps goes into `verdict.json` under `learning_signatures`, apart from the control's
`healthy_signatures`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import replica_proof as proof  # noqa: E402
from tensorquill.evidence import compare_replicas  # noqa: E402
from test_lane_z4_driver import (  # noqa: E402
    LAUNCH, OTHER_LAUNCH, SDK, _free_port, _mutate, _passing, _plan, _records, _server_binary,
    _stand_in_compare_replicas,
)
from test_replica_attribution_contract import DOCUMENT as ATTRIBUTION_CASES  # noqa: E402
from test_replica_attribution_contract import attribution  # noqa: E402
from test_replica_contract import reference  # noqa: E402

TESTS = Path(__file__).resolve().parent


def _result(plan: proof.Plan, **kwargs) -> tuple[dict, list[dict]]:
    records = _records(plan, **kwargs)
    return _stand_in_compare_replicas(records), records


def _entries(result: dict, step: int) -> list[dict]:
    return [e for e in result.get("attribution") or [] if e["step"] == step]


def _set(result: dict, step: int, param: str, by: str, **change) -> dict:
    out = copy.deepcopy(result)
    for entry in out["attribution"]:
        if (entry["step"], entry["parameter"], entry["by"]) == (step, param, by):
            for key in ("verdict", "outliers", "why"):
                entry.pop(key, None)
            entry.update(change)
    return out


# --- What is due, on the default plants -------------------------------------------------------------


@pytest.mark.parametrize("world", [2, 3, 8])
def test_a_result_with_section_4as_entries_passes_on_two_three_and_eight_ranks(world):
    plan = _plan(world)
    result, records = _result(plan)
    failures, lines = proof.check_attribution(result, plan, LAUNCH, records)
    assert failures == [], failures
    planted = {(p.step, p.param): p for p in plan.plants}
    for (step, param), plant in planted.items():
        entries = {e["by"]: e for e in _entries(result, step) if e["parameter"] == param}
        if world == 2:
            assert all(e["verdict"] == "inconclusive" and e["why"] == "two_ranks" and "outliers" not in e
                       for e in entries.values()), entries
        else:
            if not plant.quiet:
                assert entries["fingerprints"]["outliers"] == [plant.rank]
            assert entries["signatures"]["outliers"] == [plant.rank]
        assert ("fingerprints" in entries) == (not plant.quiet)  # a bits-only plant has none
    # Every planted step and parameter gets a line; a clean step none.
    assert {int(line.split()[1]) for line in lines} == {p.step for p in plan.plants}


def test_without_the_key_the_driver_says_attribution_not_present_and_fails():
    plan, log, reports, records = _passing()
    bare = proof.VerdictLog()
    bare.add(100.5, reference(records, 1e-6))  # today's reader: no `attribution`
    failures, lines = proof.check_attribution(bare.as_result(), plan, LAUNCH, records)
    assert [f for f in failures if f.startswith("attribution not present")] == failures and len(failures) == 1
    assert "tensorquill.evidence:compare_replicas" in failures[0] and "[1, 2, 3, 4, 5]" in failures[0]
    assert all("attribution not present" in line for line in lines)
    verdict = proof.evaluate(plan, bare, {}, reports, LAUNCH, records, reader_found=True)
    assert verdict.code == proof.EXIT_FAIL and verdict.failures == failures
    assert any(line.startswith("FAIL attribution not present") for line in verdict.lines)
    # The same records with the key: PASS.
    assert proof.evaluate(plan, log, {}, reports, LAUNCH, records, reader_found=True).code == proof.EXIT_PASS


def test_in_server_mode_the_missing_key_is_the_servers():
    plan = _plan(3, "--mode", "server", "--endpoint", "grpc://h:1", "--store", "s", "--status-interval", "0.5")
    records = _records(plan)
    failures, _ = proof.check_attribution(reference(records, 1e-6), plan, LAUNCH, records)
    assert len(failures) == 1 and failures[0].startswith("attribution not present: no look of the server")


def test_a_key_present_without_a_due_entry_names_the_step():
    plan = _plan(3)
    result, records = _result(plan)
    result["attribution"] = [e for e in result["attribution"] if e["step"] != 3]
    failures, _ = proof.check_attribution(result, plan, LAUNCH, records)
    assert sorted(failures) == ["step 3 1.weight by fingerprints: no attribution entry, where one is due"
                                " (planted on rank(s) [1])",
                                "step 3 1.weight by signatures: no attribution entry, where one is due"
                                " (the records' majority signature sets apart rank(s) [1])"]


@pytest.mark.parametrize("arm", [["--no-plants"], ["--arm", "learning"]])
def test_nothing_is_due_without_a_plant(arm):
    plan = _plan(3, *arm)
    records = _records(plan)
    assert proof.check_attribution(reference(records, 1e-6), plan, LAUNCH, records) == ([], [])


def test_a_rank_that_never_reports_leaves_its_planted_steps_with_nothing_due():
    plan = _plan(3, "--hold-rank", "2", "--release", "never")  # steps 2 to 7 never judged
    result, records = _result(plan)
    assert {e["step"] for e in result.get("attribution", [])} == {1}
    assert proof.check_attribution(result, plan, LAUNCH, records)[0] == []
    without = {k: v for k, v in result.items() if k != "attribution"}
    assert proof.check_attribution(without, plan, LAUNCH, records)[0]  # step 1 was judged: still due
    # An entry on a step that stayed insufficient is wrong (section 4a reads complete steps only).
    stray = copy.deepcopy(result)
    stray["attribution"].append({**stray["attribution"][0], "step": 3, "parameter": "1.weight"})
    assert any("where coverage is insufficient" in f for f in proof.check_attribution(stray, plan, LAUNCH, records)[0])


# --- Each wrong reading is caught ---------------------------------------------------------------------


def _wrong_readings(world: int) -> dict[str, tuple[dict, str]]:
    plan = _plan(world)
    result, _ = _result(plan)
    last = world - 1
    first = result["attribution"][0]
    readings = {
        "two ranks, the higher named": (_set(result, 1, "0.weight", "fingerprints", verdict="attributed",
                                             outliers=[1]), "inconclusive, why two_ranks is due"),
        "two ranks, why no_majority": (_set(result, 3, "1.weight", "fingerprints", verdict="inconclusive",
                                            why="no_majority"), "inconclusive, why two_ranks is due"),
        "the bits-only plant named by signatures between two": (
            _set(result, 4, "0.weight", "signatures", verdict="attributed", outliers=[1]), "two_ranks is due"),
    } if world == 2 else {
        "rank 0 taken for the reference": (_set(result, 1, "0.weight", "fingerprints", verdict="attributed",
                                                outliers=[r for r in range(world) if r != 0]),
                                           f"attributed, outliers [{last}] is due"),
        "the wrong rank named": (_set(result, 5, "1.weight", "fingerprints", verdict="attributed", outliers=[0]),
                                 f"attributed, outliers [{last}] is due"),
        "no majority where one holds": (_set(result, 3, "1.weight", "fingerprints", verdict="inconclusive",
                                             why="no_majority"), "attributed, outliers [1] is due"),
        "the ulp plant's rank not named": (_set(result, 2, "0.bias", "signatures", verdict="inconclusive",
                                                why="no_majority"), f"attributed, outliers [{last}] is due"),
        "the negzero plant's wrong rank": (_set(result, 4, "0.weight", "signatures", verdict="attributed",
                                                outliers=[0]), "attributed, outliers [1] is due"),
        "a bits-only plant attributed by fingerprints": (
            {**result, "attribution": result["attribution"] + [
                {**first, "step": 2, "parameter": "0.bias", "by": "fingerprints"}]}, "for a bits-only plant"),
    }
    readings.update({
        "an entry by fingerprints on a clean step": (
            {**result, "attribution": result["attribution"] + [{**first, "step": 0}]}, "which did not drift"),
        "an entry by signatures on a clean step": (
            {**result, "attribution": result["attribution"] + [{**first, "step": 6, "by": "signatures"}]},
            "whose signatures did not differ"),
        "an entry twice": ({**result, "attribution": result["attribution"] + [first]}, "2 attribution entries"),
        "an entry of another launch": ({**result, "attribution": result["attribution"] + [
            {**first, "launch": OTHER_LAUNCH}]}, "another launch or group"),
        "attributed with no outliers": (_set(result, 1, "0.weight", "fingerprints", verdict="attributed",
                                             outliers=[]), "attributed with outliers []"),
        "inconclusive naming a rank": (_set(result, 1, "0.weight", "fingerprints", verdict="inconclusive",
                                            why="two_ranks", outliers=[1]), "name no rank"),
        "a verdict 4a does not have": (_set(result, 1, "0.weight", "fingerprints", verdict="suspect"),
                                       "verdict 'suspect'"),
    })
    return readings


@pytest.mark.parametrize("world,name", [(w, n) for w in (2, 3, 8) for n in _wrong_readings(w)])
def test_each_wrong_reading_fails_with_its_reason(world, name):
    readings = _wrong_readings(world)
    plan = _plan(world)
    records = _records(plan)
    result, says = readings[name]
    failures, _ = proof.check_attribution(result, plan, LAUNCH, records)
    assert any(says in f for f in failures), (says, failures)


def test_the_signatures_majority_comes_from_the_records():
    """The bits of one rank's record are not those a majority holds: the records name that rank
    for the step, and an entry naming any other fails."""
    plan = _plan(3)
    result, records = _result(plan)
    broken = copy.deepcopy(records)
    for record in broken:
        if record["step"] == 4 and record["rank"] == 1:
            record["replica"]["params"]["0.weight"]["sig"] = str(1000 + len("0.weight"))  # as the others
        if record["step"] == 4 and record["rank"] == 0:
            record["replica"]["params"]["0.weight"]["sig"] = "777"
    failures, _ = proof.check_attribution(result, plan, LAUNCH, broken)
    assert any("the records' signatures set apart rank(s) [0], not the planted [1]" in f for f in failures), failures
    # Three signatures on three ranks: no majority is due, whatever the reader says.
    for record in broken:
        if record["step"] == 4 and record["rank"] == 2:
            record["replica"]["params"]["0.weight"]["sig"] = "555"
    failures, _ = proof.check_attribution(result, plan, LAUNCH, broken)
    assert any("inconclusive, why no_majority is due" in f for f in failures), failures


def test_planted_ranks_that_are_not_a_strict_minority_are_shown_not_asserted():
    plan = _plan(4, "--plant", "2:1:0.weight:add", "--plant", "3:1:0.weight:add")  # two of four, the same
    result, records = _result(plan)
    failures, lines = proof.check_attribution(result, plan, LAUNCH, records)
    assert failures == [], failures
    [line] = [x for x in lines if "by fingerprints" in x]
    assert "planted on 2 of 4 ranks, not a strict minority: not asserted" in line
    # The signatures still read from the records: two hold "999" and two the base, no majority.
    [line] = [x for x in lines if "by signatures" in x]
    assert "inconclusive, why no_majority" in line


# --- Against the contract's own cases -----------------------------------------------------------------


_FIXTURE_PLANTS = {  # case: (ranks, the planted ranks on `w`, kind)
    "two_ranks_differ": (2, [1], "add"),
    "three_ranks_two_to_one": (3, [2], "add"),
    "three_ranks_rank_0_is_the_outlier": (3, [0], "add"),
    "four_ranks_three_to_one": (4, [3], "add"),
    "five_ranks_three_one_one": (5, [3, 4], "add"),
    "signature_only_two_to_one": (3, [1], "ulp"),
    "signature_only_between_two_ranks": (2, [1], "ulp"),
    "signed_zero_on_one_of_three": (3, [2], "negzero"),
    "four_ranks_two_to_two": (4, [2, 3], "add"),
}


def _fixture(case: str) -> tuple[proof.Plan, dict, list[dict]]:
    world, ranks, kind = _FIXTURE_PLANTS[case]
    body = ATTRIBUTION_CASES["cases"][case]
    plan = _plan(world, "--no-plants")
    plan.steps, plan.replica_every = 100, 100
    plan.plants = [proof.Plant(rank, 99, "w", kind) for rank in ranks]
    said = reference(body["records"], ATTRIBUTION_CASES["tolerance"])
    return plan, {**said, "attribution": body["expected"]["attribution"]}, body["records"]


@pytest.mark.parametrize("case", sorted(_FIXTURE_PLANTS))
def test_the_contracts_expected_entries_pass_as_planted(case):
    plan, result, records = _fixture(case)
    assert proof.check_attribution(result, plan, LAUNCH, records)[0] == []
    # The readers test_replica_attribution_contract.py catches are caught here too.
    against_rank_0 = copy.deepcopy(result)
    for entry in against_rank_0["attribution"]:
        if entry["verdict"] == "attributed":
            entry["outliers"] = [r for r in entry["group"] if r != 0]
    names_the_higher = copy.deepcopy(result)
    for entry in names_the_higher["attribution"]:
        if entry.get("why") == "two_ranks":
            del entry["why"]
            entry.update(verdict="attributed", outliers=[entry["group"][-1]])
    plurality = copy.deepcopy(result)
    for entry in plurality["attribution"]:
        if entry.get("why") == "no_majority":
            del entry["why"]
            entry.update(verdict="attributed", outliers=[entry["group"][-1]])
    changed = [r for r in (against_rank_0, names_the_higher, plurality) if r != result]
    assert changed, case
    for wrong in changed:
        assert proof.check_attribution(wrong, plan, LAUNCH, records)[0], case


def test_the_contracts_case_fails_a_plan_it_does_not_fit():
    plan, result, records = _fixture("three_ranks_two_to_one")
    plan.plants = [proof.Plant(1, 99, "w", "add")]
    failures = proof.check_attribution(result, plan, LAUNCH, records)[0]
    assert any("attributed, outliers [1] is due" in f for f in failures), failures


# --- The verdict log keeps the key ---------------------------------------------------------------------


def test_the_log_carries_attribution_only_once_a_look_did():
    plan = _plan(3)
    result, records = _result(plan)
    log = proof.VerdictLog()
    log.add(1.0, reference(records, 1e-6))
    assert "attribution" not in log.as_result() and not log.attribution_seen
    log.add(2.0, result)
    assert log.attribution_seen and log.as_result()["attribution"] == result["attribution"]  # in 4a's order
    # A later look that no longer lists a step keeps its entries (a server drops them with the step) ...
    later = {**result, "steps": [e for e in result["steps"] if e["step"] != 1],
             "attribution": [e for e in result["attribution"] if e["step"] != 1]}
    log.add(3.0, later)
    assert log.as_result()["attribution"] == result["attribution"]
    # ... and one that lists the step replaces them, so an entry it dropped is gone.
    relisted = {**result, "attribution": [e for e in result["attribution"] if e["step"] != 5]}
    log.add(4.0, relisted)
    assert {e["step"] for e in log.as_result()["attribution"]} == {1, 2, 3, 4}
    # A look whose key went away (no entry left) still counts as having carried it before.
    log.add(5.0, reference(records, 1e-6))
    assert log.attribution_seen and log.as_result()["attribution"] == []


def test_the_order_of_the_kept_entries_is_section_4as():
    entries = [{"step": s, "launch": LAUNCH, "group": [0, 1, 2], "parameter": p, "by": b, "verdict": "inconclusive",
                "why": "no_majority"} for s in (10, 2) for p in ("b", "a") for b in ("signatures", "fingerprints")]
    log = proof.VerdictLog()
    log.add(1.0, {"steps": [], "replica_drift": [], "attribution": entries})
    order = [(e["step"], e["parameter"], e["by"]) for e in log.as_result()["attribution"]]
    assert order == sorted(order)


# --- The learning arm: arguments, batch and weights -----------------------------------------------------


def test_the_learning_arm_plants_nothing_and_takes_a_learning_rate():
    plan = _plan(3, "--arm", "learning")
    assert (plan.arm, plan.lr, plan.plants, plan.summary_key) == ("learning", proof.LEARNING_LR, [],
                                                                  "learning_signatures")
    assert _plan(3, "--arm", "learning", "--lr", "0").lr == 0.0  # allowed: the check then fails the run
    control = _plan(3)
    assert (control.arm, control.lr, control.summary_key) == ("control", 0.0, "healthy_signatures")
    assert len(control.plants) == 5  # the control arm keeps its five plants


@pytest.mark.parametrize("argv,says", [
    (["--arm", "learning", "--plant", "1:1:0.weight:add"], "plants nothing"),
    (["--lr", "0.1"], "the control arm keeps lr 0"),
    (["--arm", "learning", "--lr", "-0.1"], "0 or more"),
    (["--arm", "learning", "--lr", "nan"], "finite"),
    (["--arm", "learning", "--lr", "inf"], "finite"),
])
def test_the_arms_refuse_what_belongs_to_the_other(argv, says):
    with pytest.raises(proof.UsageError, match=says):
        _plan(3, *argv)
    with pytest.raises(SystemExit):
        _plan(3, "--arm", "sometimes")


def test_each_rank_gets_a_batch_of_its_own_the_same_every_time():
    x0, y0 = proof.learning_batch(0, 5, torch.device("cpu"))
    again, _ = proof.learning_batch(0, 5, torch.device("cpu"))
    x1, _ = proof.learning_batch(1, 5, torch.device("cpu"))
    later, _ = proof.learning_batch(0, 6, torch.device("cpu"))
    assert x0.shape == (8, 4) and y0.shape == (8, 1)
    assert torch.equal(x0, again) and not torch.equal(x0, x1) and not torch.equal(x0, later)


def _learn(model, base, lr: float, steps: int) -> dict:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    report: dict = {}
    before = {n: v.clone() for n, v in base.items()}
    for step in range(steps):
        optimizer.zero_grad()
        x, y = proof.learning_batch(0, step, torch.device("cpu"))
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
        before = proof.weight_change(model, base, before, report)
    return report


def test_weight_change_sees_training_move_every_parameter_and_lr_0_move_none():
    model = proof.build_model(torch.device("cpu"))
    base = proof.base_values(model)
    proof.reset_and_plant(model, base, ())
    moved = _learn(model, base, proof.LEARNING_LR, 100)
    assert set(moved) == set(proof.PARAM_NAMES)
    assert all(v["changed_from_start"] and v["finite"] and v["steps_changed"] == 100 and v["max_abs_change"] > 0
               for v in moved.values()), moved
    # A hundred steps at the default rate stay bounded: the target is one the model can approach.
    assert all(v["max_abs_change"] < 10 for v in moved.values()), moved
    proof.reset_and_plant(model, base, ())
    still = _learn(model, base, 0.0, 4)
    assert all(not v["changed_from_start"] and v["steps_changed"] == 0 for v in still.values()), still


def test_weight_change_marks_a_value_that_is_not_finite():
    model = proof.build_model(torch.device("cpu"))
    base = proof.base_values(model)
    proof.reset_and_plant(model, base, ())
    report: dict = {}
    with torch.no_grad():
        dict(model.named_parameters())["0.bias"][0] = math.inf
    proof.weight_change(model, base, base, report)
    assert report["0.bias"]["finite"] is False and report["0.weight"]["finite"] is True


def _weights(changed: bool = True, finite: bool = True) -> dict:
    return {name: {"changed_from_start": changed, "max_abs_change": 0.1 if changed else 0.0,
                   "steps_changed": 8 if changed else 0, "finite": finite} for name in proof.PARAM_NAMES}


def test_check_weights_fails_a_rank_whose_weights_did_not_move():
    plan = _plan(3, "--arm", "learning")
    reports = {r: {"rank": r, "weights": _weights()} for r in plan.group}
    assert proof.check_weights(reports, plan) == []
    one = copy.deepcopy(reports)
    one[2]["weights"]["1.weight"]["changed_from_start"] = False
    [failure] = proof.check_weights(one, plan)
    assert failure.startswith("rank 2: ['1.weight'] did not change from their starting values")
    missing = copy.deepcopy(reports)
    del missing[1]["weights"]
    assert proof.check_weights(missing, plan)[0].startswith("rank 1 reported no change of its weights")
    broken = copy.deepcopy(reports)
    broken[0]["weights"]["0.bias"]["finite"] = False
    assert "did not stay finite" in proof.check_weights(broken, plan)[0]
    assert proof.check_weights({r: {"rank": r} for r in plan.group}, _plan(3)) == []  # the control arm has none


def _learning_passing(world: int = 3):
    plan = _plan(world, "--arm", "learning", "--steps", "8")
    records = _records(plan)
    log = proof.VerdictLog()
    log.add(100.5, _stand_in_compare_replicas(records))
    reports = {r: {"rank": r, "steps_done": 8, "durations": [0.2] * 8, "weights": _weights(),
                   "written": [{"step": s, "end": 10 * (s + 1), "t": 100.0, "replica": True} for s in range(8)],
                   "offsets": []} for r in plan.group}
    return plan, log, reports, records


def test_the_learning_arms_count_has_its_own_key_in_the_verdict():
    plan, log, reports, records = _learning_passing()
    verdict = proof.evaluate(plan, log, {}, reports, LAUNCH, records, reader_found=True)
    assert verdict.code == proof.EXIT_PASS, verdict.lines
    document = verdict.document()
    assert set(document) == {"code", "failures", "arm", "learning_signatures"} and document["arm"] == "learning"
    learned = document["learning_signatures"]
    assert (learned["steps_without_plant"], learned["compared"], learned["signatures_differ"]) == (8, 8, 0)
    assert learned["lr"] == proof.LEARNING_LR and learned["weights"]["2"] == _weights()
    assert learned["attribution"] == []
    [line] = [x for x in verdict.lines if "learning arm" in x and "signatures differed" in x]
    assert "differed on 0 of 8 compared steps" in line and "changed from their start on 3 of 3 ranks" in line
    assert "reset" not in line.split("no reset")[1]  # the control's words are not said of this arm
    # The control arm's verdict keeps its key, and only it.
    plan, log, reports, records = _passing()
    control = proof.evaluate(plan, log, {}, reports, LAUNCH, records, reader_found=True).document()
    assert set(control) == {"code", "failures", "arm", "healthy_signatures"} and control["arm"] == "control"


def test_the_learning_arm_counts_differing_signatures_by_parameter_and_shows_their_attribution():
    plan, log, reports, records = _learning_passing()
    differs = copy.deepcopy(records)
    for record in differs:
        if record["rank"] == 2 and record["step"] in (3, 6):
            record["replica"]["params"]["0.bias"]["sig"] = "31"
    log = proof.VerdictLog()
    log.add(100.5, _stand_in_compare_replicas(differs))
    verdict = proof.evaluate(plan, log, {}, reports, LAUNCH, differs, reader_found=True)
    assert verdict.code == proof.EXIT_PASS, verdict.lines  # reported, not failed
    learned = verdict.document()["learning_signatures"]
    assert (learned["compared"], learned["signatures_differ"], learned["by_parameter"]) == (8, 2, {"0.bias": 2})
    assert [(e["step"], e["parameter"], e["outliers"]) for e in learned["attribution"]] == [(3, "0.bias", [2]),
                                                                                            (6, "0.bias", [2])]
    plan.healthy_mismatch = "fail"
    assert proof.evaluate(plan, log, {}, reports, LAUNCH, differs, reader_found=True).code == proof.EXIT_FAIL


def test_the_learning_arm_fails_when_a_rank_did_not_learn():
    plan, log, reports, records = _learning_passing()
    reports[1]["weights"] = _weights(changed=False)
    verdict = proof.evaluate(plan, log, {}, reports, LAUNCH, records, reader_found=True)
    assert verdict.code == proof.EXIT_FAIL
    assert any("rank 1:" in f and "did not change" in f for f in verdict.failures), verdict.failures
    # Also when the reader was not found: the weights are checked from the ranks' reports.
    verdict = proof.evaluate(plan, proof.VerdictLog(), {}, reports, LAUNCH, records, reader_found=False)
    assert verdict.code == proof.EXIT_FAIL


# --- Under torchrun ------------------------------------------------------------------------------------

_STAND_IN = f'''
"""compare_replicas as it will be once D1 is built: the real reader, then section 4a's entries
from the reference reading, under `attribution` when there is one."""
import sys
sys.path.insert(0, {str(TESTS)!r})
from tensorquill.evidence import compare_replicas
from test_replica_attribution_contract import attribution


def with_attribution(records, *, tolerance=1e-6, attempt_order=None):
    out = compare_replicas(records, tolerance=tolerance)
    entries = attribution(list(records), tolerance)
    if entries:
        out["attribution"] = entries
    return out
'''


def _real_reader_attributes() -> bool:
    """Whether today's `compare_replicas` gives `attribution` (once lane AB1 is merged)."""
    plan = _plan(3)
    return "attribution" in compare_replicas(_records(plan), tolerance=1e-6)


def _run(world: int, *argv: str, extra_path: Path | None = None,
         hide_numpy: Path | None = None) -> tuple[subprocess.CompletedProcess, dict]:
    out = Path(argv[argv.index("--out") + 1])
    path = [str(p) for p in (hide_numpy, extra_path) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path + [str(SDK / "src")])}
    env.pop("TENSORQUILL_ENDPOINT", None)
    done = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={world}", "--master-addr", "127.0.0.1",
         "--master-port", str(_free_port()), str(SDK / "examples" / "replica_check.py"), "--cpu", *argv],
        capture_output=True, text=True, timeout=300, env=env, cwd=SDK,
    )
    found = [json.loads(p.read_text()) for p in out.glob("*/verdict.json")]
    assert len(found) == 1, done.stdout + done.stderr
    return done, found[0]


@pytest.mark.parametrize("world", [2, 3])
def test_files_mode_names_the_planted_rank_with_the_stand_in_reader(tmp_path, world):
    (tmp_path / "d1_stand_in.py").write_text(_STAND_IN)
    done, verdict = _run(world, "--out", str(tmp_path / "out"), "--steps", "8", "--step-seconds", "0.1",
                         "--reader", "d1_stand_in:with_attribution", extra_path=tmp_path)
    assert (verdict["code"], verdict["failures"]) == (proof.EXIT_PASS, []), done.stdout + done.stderr
    assert done.returncode == 0 and verdict["arm"] == "control" and "healthy_signatures" in verdict
    if world == 2:
        expected = ["step 1 0.weight by fingerprints: inconclusive, why two_ranks",
                    "step 2 0.bias by signatures: inconclusive, why two_ranks",
                    "step 4 0.weight by signatures: inconclusive, why two_ranks"]
    else:
        expected = ["step 1 0.weight by fingerprints: attributed, outliers [2]",
                    "step 2 0.bias by signatures: attributed, outliers [2]",
                    "step 3 1.weight by fingerprints: attributed, outliers [1]",
                    "step 4 0.weight by signatures: attributed, outliers [1]",
                    "step 5 1.weight by fingerprints: attributed, outliers [2]"]
    for line in expected:
        assert f"attribution: {line}" in done.stdout, (line, done.stdout)
    assert "by fingerprints" not in "".join(x for x in done.stdout.splitlines() if "step 2 0.bias by" in x)


def test_files_mode_with_the_real_reader_fails_until_it_attributes(tmp_path):
    """Before lanes AB1 and AB2 are merged: FAIL, and only for the missing key. After: PASS."""
    done, verdict = _run(3, "--out", str(tmp_path), "--steps", "8", "--step-seconds", "0.1")
    if _real_reader_attributes():
        assert (verdict["code"], verdict["failures"]) == (proof.EXIT_PASS, []), done.stdout + done.stderr
    else:
        assert verdict["code"] == proof.EXIT_FAIL and done.returncode == proof.EXIT_FAIL, done.stdout + done.stderr
        [failure] = verdict["failures"]
        assert failure.startswith("attribution not present: no look of the reader tensorquill.evidence:compare_replicas")
        assert "FAIL attribution not present" in done.stdout


@pytest.mark.parametrize("world", [2, 3])
def test_files_mode_learning_arm_changes_the_weights_and_counts_its_own_steps(tmp_path, world):
    done, verdict = _run(world, "--out", str(tmp_path), "--arm", "learning", "--steps", "12", "--step-seconds", "0.05")
    assert (verdict["code"], verdict["failures"]) == (proof.EXIT_PASS, []), done.stdout + done.stderr
    assert verdict["arm"] == "learning" and "healthy_signatures" not in verdict
    learned = verdict["learning_signatures"]
    assert (learned["steps_without_plant"], learned["compared"], learned["not_judged"]) == (12, 12, [])
    assert learned["signatures_differ"] == 0  # on the CPU, Gloo's averaged gradients are equal bits
    assert sorted(learned["weights"]) == [str(r) for r in range(world)]
    for weights in learned["weights"].values():
        assert all(v["changed_from_start"] and v["finite"] and v["steps_changed"] == 12 for v in weights.values())
    assert f"Weights changed from their start on {world} of {world} ranks" in done.stdout


def test_the_learning_arm_with_lr_0_fails(tmp_path):
    """The fail-before of the learning arm: no learning rate, the weights do not move, the run fails."""
    done, verdict = _run(2, "--out", str(tmp_path), "--arm", "learning", "--lr", "0", "--steps", "6",
                         "--step-seconds", "0.05")
    assert verdict["code"] == proof.EXIT_FAIL and done.returncode == proof.EXIT_FAIL, done.stdout + done.stderr
    assert sorted(f.split(":")[0] for f in verdict["failures"]) == ["rank 0", "rank 1"]
    assert all("did not change from their starting values with lr 0.0" in f for f in verdict["failures"])


def test_the_learning_arm_needs_no_numpy(tmp_path):
    hidden = tmp_path / "no_numpy"
    hidden.mkdir()
    (hidden / "sitecustomize.py").write_text('import sys\nsys.modules["numpy"] = None\n')
    done, verdict = _run(2, "--out", str(tmp_path / "out"), "--arm", "learning", "--steps", "6",
                         "--step-seconds", "0.05", hide_numpy=hidden)
    assert "Numpy is not available" not in done.stdout + done.stderr, done.stdout + done.stderr
    assert (verdict["code"], verdict["failures"]) == (proof.EXIT_PASS, []), done.stdout + done.stderr


def _server(store: Path) -> tuple[subprocess.Popen, int]:
    server = subprocess.Popen(
        [str(_server_binary()), "--listen", "127.0.0.1:0", "--store", str(store), "--watchdog-seconds", "0.5",
         "--flush-seconds", "0.3"], stdout=subprocess.PIPE, text=True,
        env={k: v for k, v in os.environ.items() if k != "TENSORQUILL_TOKEN"},
    )
    return server, int(server.stdout.readline().rsplit(":", 1)[1])


@pytest.mark.skipif(_server_binary() is None, reason="tensorquill-server is not built")
@pytest.mark.parametrize("arm", ["control", "learning"])
def test_server_mode_both_arms(tmp_path, arm):
    """The learning arm plants nothing, so nothing is due and it passes today. The control arm's
    planted steps are due an entry the server does not write before lane AB2 is merged: FAIL, and
    only for the missing key; once the server writes it, PASS."""
    store = tmp_path / "store"
    server, port = _server(store)
    try:
        done, verdict = _run(2, "--out", str(tmp_path / "out"), "--mode", "server", "--endpoint",
                             f"grpc://127.0.0.1:{port}", "--store", str(store), "--status-interval", "0.5",
                             "--steps", "16", "--step-seconds", "0.3", "--arm", arm)
    finally:
        server.terminate()
        server.wait(timeout=30)
    if arm == "learning":
        assert (verdict["code"], verdict["failures"]) == (proof.EXIT_PASS, []), done.stdout + done.stderr
        assert verdict["learning_signatures"]["compared"] == 16
        return
    [status] = list(store.glob("runs/*/watchdog.json"))
    if "attribution" in (json.loads(status.read_text()).get("replica") or {}):
        assert (verdict["code"], verdict["failures"]) == (proof.EXIT_PASS, []), done.stdout + done.stderr
    else:
        [failure] = verdict["failures"]
        assert verdict["code"] == proof.EXIT_FAIL and failure.startswith("attribution not present: no look of the server")
