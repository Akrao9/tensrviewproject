"""Lane W3: detector rules held the same in both engines.

Contract 2 (signed off Nov 19, 2026): when unchanged weights are a stall.

- A weight check that does not show a tracked parameter unchanged (it is absent from a written
  `weights_unchanged`) breaks that parameter's current streak, whether or not it had a gradient
  there (E1). It does not clear the parameter for the rest of the run: a later streak still fires.
- A: in the list with no gradient entry holds the streak. B: in the list with a zero gradient
  resets it. C: on a row whose total `grad_norm` is not finite, resets apply and nothing counts.
- A row with no list is no weight check and holds every streak; an empty list resets every one.
- `trained_before` counts parameters not shown unchanged at an earlier check (changed, or not
  compared), and the summary says so rather than claiming a change.

Every case runs through `build_evidence` with the Python detectors and with the Rust ones; the
Rust half skips when no `tensorquill-server` speaking this SDK's protocol is found.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import pytest

import tensorquill
from tensorquill.evidence import incident_summary

ENGINES = ("python", "rust")


@functools.cache
def _rust_missing() -> str | None:
    """Why the Rust detectors cannot run here, or None when they can."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return "pyarrow is not installed"
    from tensorquill.server_detect import server_binary

    try:
        server_binary()
    except RuntimeError as error:
        return str(error)
    return None


def _evidence(records: list[dict[str, Any]], engine: str) -> dict[str, Any]:
    if engine == "rust":
        missing = _rust_missing()
        if missing:
            pytest.skip(missing)
    return tensorquill.build_evidence(records, detectors=engine)


def check(step: int, unchanged: list[str] | None, grads: dict[str, float], *, total: float | None = None) -> dict[str, Any]:
    """One record. `unchanged` None means no weight check on this step (no list written)."""
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "lane-w3",
        "rank": 0,
        "attempt": 0,
        "step": step,
        "layers": {},
        "grad_norms": dict(grads),
        "grad_norm": total if total is not None else (math.sqrt(sum(v * v for v in grads.values())) if grads else None),
    }
    if unchanged is not None:
        record["weights_unchanged"] = list(unchanged)
        record["weight_norms"] = {name: 1.0 for name in {"p", "q", "r", *grads}}
    return record


def stall(records: list[dict[str, Any]], engine: str) -> tuple[dict[str, Any] | None, str]:
    evidence = _evidence(records, engine)
    found = [i for i in evidence["incidents"] if i["kind"] == "stalled_weights"]
    assert len(found) <= 1
    return (found[0] if found else None), incident_summary(evidence)


def onsets(incident: dict[str, Any] | None) -> list[tuple[int, list[str], int]]:
    if incident is None:
        return []
    return [(o["step"], o["parameters"], o["trained_before"]) for o in incident["onsets"]]


def held(step: int) -> dict[str, Any]:
    """`p` unchanged with a gradient at `step`."""
    return check(step, ["p"], {"p": 0.5})


# --- contract 2, E1 ------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("listed_at_2", [[], ["r"]], ids=["empty-list", "other-listed"])
def test_e1_a_change_without_a_gradient_breaks_the_streak(engine: str, listed_at_2: list[str]) -> None:
    # The audit's case: unchanged with a gradient at 0 and 1, changed with no gradient at 2,
    # unchanged with a gradient at 3. Both engines called it three in a row at step 3.
    records = [held(0), held(1), check(2, listed_at_2, {}), held(3)]
    incident, _ = stall(records, engine)
    assert incident is None, onsets(incident)


@pytest.mark.parametrize("engine", ENGINES)
def test_e1_acceptance_first_streak_clears_and_the_later_stall_still_fires(engine: str) -> None:
    records = [held(0), held(1), check(2, [], {}), held(3), held(4), held(5)]
    incident, summary = stall(records, engine)
    assert incident is not None
    assert incident["step"] == 5
    assert onsets(incident) == [(5, ["p"], 1)]
    assert incident["stalled_steps"] == 1
    # The amendment: absence from the list may be "not compared"; the summary must not claim a change.
    assert "1 of them changed, or could not be compared, at an earlier weight check" in summary
    assert "had changed at an earlier check" not in summary


@pytest.mark.parametrize("engine", ENGINES)
def test_e1_a_stall_after_a_reported_stall_and_a_change_counts_again_but_is_not_a_new_onset(engine: str) -> None:
    records = [held(0), held(1), held(2), check(3, [], {}), held(4), held(5), held(6)]
    incident, _ = stall(records, engine)
    assert incident is not None and incident["step"] == 2
    assert onsets(incident) == [(2, ["p"], 0)]
    # Steps 2 and 6 had a parameter whose current streak was at least three; 3, 4 and 5 did not.
    assert incident["stalled_steps"] == 2


# --- the cells the author decided ----------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_listed_without_a_gradient_holds_the_streak(engine: str) -> None:
    records = [held(0), held(1), check(2, ["p"], {}), held(3)]
    incident, _ = stall(records, engine)
    assert incident is not None and incident["step"] == 3


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("grad", [0.0, math.nan], ids=["zero", "nan"])
def test_b_listed_with_a_zero_or_non_finite_gradient_resets_the_streak(engine: str, grad: float) -> None:
    # A non-finite parameter gradient makes the total non-finite too, so that row is also case C,
    # where resets still apply: B's reset is one of them.
    records = [held(0), held(1), check(2, ["p"], {"p": grad}), held(3), held(4)]
    incident, _ = stall(records, engine)
    assert incident is None, onsets(incident)
    incident, _ = stall(records + [held(5)], engine)
    assert incident is not None and incident["step"] == 5


@pytest.mark.parametrize("engine", ENGINES)
def test_c_non_finite_total_still_resets_a_parameter_shown_changing(engine: str) -> None:
    records = [held(0), held(1), check(2, ["r"], {"p": 0.5, "q": math.inf}, total=math.inf), held(3), held(4)]
    incident, _ = stall(records, engine)
    assert incident is None, onsets(incident)
    incident, _ = stall(records + [held(5)], engine)
    assert incident is not None and incident["step"] == 5


@pytest.mark.parametrize("engine", ENGINES)
def test_c_non_finite_total_adds_no_stall_evidence(engine: str) -> None:
    records = [held(0), held(1), check(2, ["p"], {"p": 0.5, "q": math.inf}, total=math.inf), held(3)]
    incident, _ = stall(records, engine)
    # Step 2 neither counts nor breaks: three in a row at 3, not at 2.
    assert incident is not None and incident["step"] == 3


@pytest.mark.parametrize("engine", ENGINES)
def test_no_list_is_no_weight_check_and_holds(engine: str) -> None:
    records = [held(0), held(1), check(2, None, {"p": 0.5}), held(3)]
    incident, _ = stall(records, engine)
    assert incident is not None and incident["step"] == 3


# --- what `trained_before` may claim -------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_trained_before_wording_claims_no_more_than_the_list_shows(engine: str) -> None:
    # `q` is unchanged with a gradient throughout and stalls at 2, never seen changing. `p` is
    # missing from the list at 2 with no gradient, then stalls at 5.
    def row(step: int, p_listed: bool, p_grad: bool) -> dict[str, Any]:
        grads = {"q": 0.5, **({"p": 0.5} if p_grad else {})}
        return check(step, ["q", *(["p"] if p_listed else [])], grads)

    records = [row(0, True, True), row(1, True, True), row(2, False, False), row(3, True, True), row(4, True, True), row(5, True, True)]
    incident, summary = stall(records, engine)
    assert onsets(incident) == [(2, ["q"], 0), (5, ["p"], 1)]
    assert "none of them was seen changing at an earlier weight check" in summary
    assert "has changed since the first weight check" not in summary
    assert "step 5: p (changed or not compared at an earlier check)" in summary
    assert "after changing at earlier checks" not in summary
