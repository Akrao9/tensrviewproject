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


def _evidence(records: list[dict[str, Any]], engine: str, **options: Any) -> dict[str, Any]:
    if engine == "rust":
        missing = _rust_missing()
        if missing:
            pytest.skip(missing)
    return tensorquill.build_evidence(records, detectors=engine, **options)


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


# --- one rule for step order within an attempt (E5, DET-10, RUST-05) -----------------------
#
# The effective run is judged in ascending step order (rows of one step keep their arrival
# order); an attempt's resume boundary is its lowest step, not its first row.


def row(step: int, attempt: int = 1, **fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1, "run_id": "lane-w3-order", "rank": 0, "attempt": attempt, "step": step,
        "layers": {}, "grad_norm": 1.0, "grad_norms": {"w": 1.0}, "lr": [0.1],
    }
    record.update(fields)
    return record


def kinds(evidence: dict[str, Any]) -> list[tuple[str, int]]:
    return sorted((i["kind"], i["step"]) for i in evidence["incidents"])


@pytest.mark.parametrize("engine", ENGINES)
def test_e5_the_first_fault_is_the_lowest_step_not_the_first_row(engine: str) -> None:
    def fine(step: int) -> dict[str, Any]:
        return row(step, scalars={"loss": float(step)})

    def bad(step: int) -> dict[str, Any]:
        return row(step, grad_norm=math.nan, grad_norms={"w": math.nan}, scalars={"loss": float(step)})

    records = [fine(3), fine(4), bad(9), fine(0), fine(1), bad(2)]
    evidence = _evidence(records, engine)
    found = [i for i in evidence["incidents"] if i["kind"] == "non_finite"]
    assert [i["step"] for i in found] == [2]
    # The baseline is the steps before it in step order (0 and 1), not the rows before it in the file.
    assert found[0]["scalars"]["loss"]["median_before"] == 0.5


def _moe_run() -> list[dict[str, Any]]:
    """Lane T6's DET-10 records: a gradient spike at 150, and routing that collapses at 100."""
    out = []
    for step in range(200):
        grad = 400.0 if step == 150 else 1.0 + 0.01 * (step % 7)
        counts = [30, 30, 30, 30] if step < 100 else [118, 1, 1, 0]
        out.append({
            "schema_version": 1, "run_id": "lane-w3-det10", "rank": 0, "attempt": 1, "step": step, "ts": float(step),
            "grad_norm": grad, "grad_norms": {"w": grad}, "moe": {"router": counts}, "moe_top_k": {"router": 1},
        })
    return out


@pytest.mark.parametrize("engine", ENGINES)
def test_det10_rows_out_of_order_within_an_attempt_give_the_in_order_incidents(engine: str) -> None:
    import random

    ordered = _moe_run()
    shuffled = ordered[:]
    random.Random(0).shuffle(shuffled)
    expected = kinds(tensorquill.build_evidence(ordered, detectors="python"))
    assert expected, "the in-order run must have incidents for this to test anything"
    assert kinds(_evidence(shuffled, engine)) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_rust05_a_resume_is_compared_at_the_attempts_lowest_step(engine: str) -> None:
    earlier = [row(step, 1) for step in range(21)]
    resumed = [row(step, 2, lr=[0.5] if step == 12 else [0.1]) for step in (15, 16, 17, 12, 13, 14)]
    assert ("resume_lr_change", 12) in kinds(_evidence(earlier + resumed, engine))


@pytest.mark.parametrize("engine", ENGINES)
def test_rust05_a_numbering_restart_is_seen_whatever_row_comes_first(engine: str) -> None:
    earlier = [row(step, 1) for step in range(10)]
    restarted = [row(step, 2) for step in (5, 6, 0, 1, 2)]
    assert ("repeated_steps", 0) in kinds(_evidence(earlier + restarted, engine))


# --- DET-09: rates seen only in rolled-back steps -------------------------------------------

MANIFEST_WITHOUT_SCHEDULER = {"components": {"model": {}, "optimizer": {"entries": 4}, "rng": {}, "data": {}}}


def _resume_run(rolled_back_lr: float, kept_lr: float = 0.1) -> list[dict[str, Any]]:
    """Attempt 1 runs steps 0 to 19 and attempt 2 resumes at 10 with a fixed rate and no
    scheduler in its checkpoint. Attempt 1's steps 10 to 19 are rolled back."""
    earlier = [row(step, 1, lr=[kept_lr if step < 10 else rolled_back_lr]) for step in range(20)]
    resumed = [row(10, 2, lr=[kept_lr], checkpoint={"load": MANIFEST_WITHOUT_SCHEDULER})]
    resumed += [row(step, 2, lr=[kept_lr]) for step in range(11, 15)]
    return earlier + resumed


def _scheduler_missing(evidence: dict[str, Any]) -> bool:
    return any(
        finding.get("component") == "scheduler"
        for incident in evidence["incidents"] if incident["kind"] == "resume_incomplete"
        for change in incident["changes"]
        for finding in change["findings"]
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_det09_a_rate_seen_only_in_rolled_back_steps_does_not_make_a_scheduler_expected(engine: str) -> None:
    assert not _scheduler_missing(_evidence(_resume_run(rolled_back_lr=0.2), engine))


@pytest.mark.parametrize("engine", ENGINES)
def test_det09_a_rate_that_varied_in_the_kept_history_still_does(engine: str) -> None:
    records = _resume_run(rolled_back_lr=0.1)
    records[3]["lr"] = [0.05]  # step 3 of attempt 1 is kept: the rate varied before the resume
    assert _scheduler_missing(_evidence(records, engine))


# --- DET-11: a top-k router's warm-up with n_experts <= 2k ----------------------------------


def _router(step: int, counts: list[int], k: int) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": "lane-w3-moe", "rank": 0, "attempt": 1, "step": step, "layers": {},
            "moe": {"router": counts}, "moe_top_k": {"router": k}}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("counts", "hot"),
    [([300, 300, 150, 150, 0, 0], [1, 2]), ([300, 300, 150, 150, 0, 0, 0, 0], [1, 2])],
    ids=["6-experts", "8-experts"],
)
def test_det11_warm_up_collapse_names_the_hot_experts_when_n_experts_is_at_most_2k(
    engine: str, counts: list[int], hot: list[int]
) -> None:
    # Top-3: every token picks experts 1 and 2 (each at the largest share a top-3 router allows,
    # 1/3) and one of 3 or 4; the rest are dead. With 6 experts the threshold 2/6 cannot be passed.
    evidence = _evidence([_router(step, counts, 3) for step in range(3)], engine)
    first = evidence["samples"][0]
    assert first["dead"] and first["overloaded"] == hot


# --- DET-16: a bool is not a number ---------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_det16_a_bool_layer_statistic_is_not_an_activation_spike(engine: str) -> None:
    def layer(step: int, abs_max: Any) -> dict[str, Any]:
        return row(step, layers={"l": {"mean": 0.0, "std": 0.01, "abs_max": abs_max, "zero_frac": 0.0}})

    baseline = [layer(step, 0.01) for step in range(30)]
    assert ("activation_spike", 30) in kinds(_evidence(baseline + [layer(s, 1.0) for s in range(30, 45)], engine))
    assert not [k for k in kinds(_evidence(baseline + [layer(s, True) for s in range(30, 45)], engine))
                if k[0] == "activation_spike"]


def test_det16_a_logged_bool_is_left_out_of_the_scalar_context() -> None:
    # A bool scalar is left out, the rest kept: what the row sent to Rust can carry too (a float
    # map). Python only here: `flight.record_to_row` still drops the whole map on a bool, which
    # lane W1 changes to dropping the one value; then this can run on both engines.
    records = [row(step, scalars={"loss": 2.0, "is_best": True}) for step in range(5)]
    records.append(row(5, grad_norm=math.nan, grad_norms={"w": math.nan}, scalars={"loss": 2.0, "is_best": True}))
    found = [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] == "non_finite"]
    assert found[0]["scalars"]["loss"]["median_before"] == 2.0
    assert "is_best" not in found[0]["scalars"]


def test_det16_system_readings_do_not_read_a_bool_as_a_number() -> None:
    # `evidence_system._number` is lane V3's file: the change is requested, not made here.
    from tensorquill import evidence_system

    assert evidence_system._number(True) is None and evidence_system._number(False) is None
    assert evidence_system._number(1) == 1.0


# --- RUST-12: steps the loss scaler skipped are no update history ---------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("skipped", "counted"), [(True, 0), (False, 4)], ids=["skipped", "stepped"])
def test_rust12_skipped_steps_do_not_count_toward_update_history(engine: str, skipped: bool, counted: int) -> None:
    records = [row(step, update_rms={"w": 0.001}, step_skipped=skipped, grad_scale=1024.0) for step in range(4)]
    check = _evidence(records, engine, update_window=2)["kind_checks"]["update_spike"]
    assert (check["records"], check["enough_history"]) == (counted, counted >= 4)


# --- contract 1, Python side: attempts in order of first appearance -------------------------
#
# The fixture's attempt numbers go down: A = 900 writes first, B = 2 resumes at step 30. In a
# local file, an attempt's place is its first appearance; numbers are identities, not order.


def _a_then_b(bad_b_step: int | None = None, bad_a_step: int | None = None) -> list[dict[str, Any]]:
    def fault(step: int, attempt: int, bad: int | None) -> dict[str, Any]:
        if step == bad:
            return row(step, attempt, grad_norm=math.nan, grad_norms={"w": math.nan})
        return row(step, attempt)

    a = [fault(step, 900, bad_a_step) for step in range(102)]  # A's steps 0 to 100, then its late 101
    b = [fault(step, 2, bad_b_step) for step in range(30, 36)]
    return a + b


def test_contract1_resolve_attempts_orders_attempts_by_first_appearance() -> None:
    from tensorquill.evidence import resolve_attempts

    kept = resolve_attempts(_a_then_b())
    assert [(r["attempt"], r["step"]) for r in kept] == [(900, s) for s in range(30)] + [(2, s) for s in range(30, 36)]


@pytest.mark.parametrize("engine", ENGINES)
def test_contract1_the_later_attempt_is_judged_whatever_its_number(engine: str) -> None:
    # A fault in B's rows is the run's; one in A's rolled-back steps (50) is not.
    found = [i["step"] for i in _evidence(_a_then_b(bad_b_step=32, bad_a_step=50), engine)["incidents"]
             if i["kind"] == "non_finite"]
    assert found == [32]


def _store(tmp_path: Any, order: list[int] | None) -> Any:
    import json

    rank = tmp_path / "runs" / "r" / "rank=0"
    for attempt, first, last in ((900, 0, 101), (2, 30, 35)):
        folder = rank / f"attempt={attempt}"
        folder.mkdir(parents=True)
        (folder / f"part-{first}-{last}.parquet").write_bytes(b"")
    if order is not None:
        (rank / "attempts.json").write_text(json.dumps({
            "schema_version": 1, "run_id": "r", "rank": 0,
            "attempts": [{"attempt": a, "by": "rows", "first_seen_ms": 0} for a in order],
        }))
    return rank


def test_contract1_newest_part_follows_the_stored_attempt_order(tmp_path: Any) -> None:
    from tensorquill.evidence import _newest_part

    newest, path, unknown = _newest_part(_store(tmp_path, [900, 2]))
    assert newest == (2, 35) and path is not None and path.name == "part-30-35.parquet" and unknown == 0


def test_contract1_a_store_without_the_order_falls_back_to_numeric(tmp_path: Any) -> None:
    from tensorquill.evidence import _newest_part

    newest, _, _ = _newest_part(_store(tmp_path, None))
    assert newest == (900, 101)


# --- contract 3, Python side: capture stopped ----------------------------------------------


def _status(rank_state: str, summary: str, **rank: Any) -> dict[str, Any]:
    return {"summary": summary, "ranks": [{"rank": 0, "state": rank_state, "attempt": 1, "last_step": 40, **rank}]}


def _judge(status: dict[str, Any]) -> dict[str, Any]:
    from tensorquill.evidence import run_state

    return run_state(incidents=[], steps_judged=True, status=status)


STOPPED = {"capture_stopped": {"reason": "sink_error", "seconds": 3.0}}


@pytest.mark.parametrize(
    ("rank_state", "summary"),
    [("capture_stopped", "capture_stopped"), ("unresponsive", "stalled"), ("lost", "stalled"),
     ("capture_stopped", "active")],
    ids=["stopped", "unresponsive", "lost", "summary-says-active"],
)
def test_contract3_a_rank_whose_capture_stopped_is_unknown_with_its_reason(rank_state: str, summary: str) -> None:
    judged = _judge(_status(rank_state, summary, **STOPPED))
    assert judged["state"] == "unknown"
    stopped = [r for r in judged["reasons"] if r["code"] == "capture_stopped"]
    assert stopped and stopped[0]["ranks"] == [0] and stopped[0]["state"] == "unknown"


def test_contract3_a_rank_state_this_reader_does_not_know_is_never_healthy() -> None:
    judged = _judge(_status("some_future_state", "active"))
    assert judged["state"] != "healthy"
    assert any("some_future_state" in r["detail"] for r in judged["reasons"])


def test_contract3_a_summary_word_this_reader_does_not_know_is_never_healthy() -> None:
    judged = _judge(_status("active", "some_future_summary"))
    assert judged["state"] != "healthy"
    assert any("some_future_summary" in r["detail"] for r in judged["reasons"])


def test_contract1_a_rank_whose_new_owner_has_no_rows_yet_reads_no_step() -> None:
    from tensorquill.evidence import detect_watchdog_incidents

    status = _status("silent", "stalled", last_step=None, attempt_without_steps=True, severity="critical")
    for incident in detect_watchdog_incidents(status):
        assert all(r.get("last_step") is None for r in incident.get("ranks", ())), incident


def test_contract3_a_capture_stopped_rank_raises_no_progress_incident() -> None:
    # Summary rule 5: one rank stopped capture, another is active, so the run reads `degraded`.
    # `main`'s reader turns that into `rank_silent` ("ranks_never_connected") with no rank in it.
    from tensorquill.evidence import detect_watchdog_incidents

    status = {"summary": "degraded", "ranks": [
        {"rank": 0, "state": "capture_stopped", "attempt": 1, "last_step": 40, "severity": None, **STOPPED},
        {"rank": 1, "state": "active", "attempt": 1, "last_step": 41, "severity": None},
    ]}
    assert [i["kind"] for i in detect_watchdog_incidents(status) if i["kind"] in ("rank_silent", "job_stalled")] == []
