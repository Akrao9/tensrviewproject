"""Lane AB3: `session.step(skipped=...)` and the record's `step_skipped_by`.

Section 1 of `STEP_AND_EVIDENCE_CONTRACT.md`, "What is recorded". Each row of its table is a case
in `fixtures/behaviour/cases.json`, run here by name against a real capture, with a real
`torch.amp.GradScaler` (on the CPU) where the case has one. The round trip through the Arrow row
and, when a server is built, through the real `tensorquill-server` is at the end.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import tensorquill
from tensorquill import capture as capture_module
from tensorquill.flight import record_to_row, row_to_record

CASES = json.loads((Path(__file__).parent / "fixtures" / "behaviour" / "cases.json").read_text())["cases"]

TABLE = (
    "an_update_due_and_applied",
    "an_update_due_and_skipped",
    "a_skip_only_the_caller_reported",
    "a_skip_the_caller_reported_with_no_scaler_outcome",
    "a_skip_only_the_scaler_reported",
    "a_skip_both_reported",
    "nothing_said_with_a_scaler",
    "nothing_said_without_a_scaler",
    "an_accumulation_microstep",
)

# The fixture's words for the scaler given to `init`, as what the step does to it.
SCALERS = {
    None: (False, None),
    "a GradScaler that applied this step": (True, "applied"),
    "a GradScaler that skipped this step": (True, "skipped"),
    # Given to `init`, but this update went to the optimizer directly: no `scaler.step`, no verdict.
    "a GradScaler whose `step` was not called for this update": (True, None),
}
LEFT_OUT = object()  # `session.step()` with no `skipped`


def _calls(calls: list[str]) -> tuple[int, object]:
    """The fixture's calls as (backward passes before the step, the `skipped` given)."""
    *microsteps, last = calls
    assert all(c == "(backward)" for c in microsteps), calls
    found = re.fullmatch(r"session\.step\((?:skipped=(True|False))?\)", last)
    assert found, last
    return max(len(microsteps), 1), LEFT_OUT if found.group(1) is None else found.group(1) == "True"


def _step_kwargs(skipped: object) -> dict:
    return {} if skipped is LEFT_OUT else {"skipped": skipped}


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(4, 1)


def _manual_run(steps: list[tuple[str | None, object]], *, scaler: bool, every: int = 2, backwards: int = 1,
                **init) -> list[dict]:
    """A run under `auto_step=False`. Each step is (what the scaler does: None for no scaler step,
    "applied" or "skipped"; the `skipped` given to `session.step`, or LEFT_OUT)."""
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    amp = torch.amp.GradScaler("cpu", init_scale=1024.0) if scaler else None
    # In memory, or (`jsonl=True`, or a server's endpoint) the default file under `tensorquill-out/`.
    sink = None if "endpoint" in init or init.pop("jsonl", False) else tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=every, auto_step=False,
                               scaler=amp, run_id="ab3", **init)
    for verdict, skipped in steps:
        opt.zero_grad()
        for _ in range(backwards):
            loss = model(torch.ones(2, 4)).sum()
            (amp.scale(loss) if amp is not None else loss).backward()
        if amp is not None and verdict is not None:
            if verdict == "skipped":
                model.weight.grad[0, 0] = float("inf")
            amp.step(opt)
            amp.update()
        else:
            opt.step()
        session.step(**_step_kwargs(skipped))
    session.close()
    return sink.records if sink is not None else []


def _check(records: list[dict], expected: dict) -> None:
    if "records" in expected:
        assert len(records) == expected["records"]
    for record, carried in zip(records, expected.get("records_carry", []), strict=False):
        assert {key: record.get(key, "<no key>") for key in carried} == carried
    if "records_carry" in expected:
        assert len(records) >= len(expected["records_carry"])
    for key in expected.get("records_lack", []):
        assert all(key not in record for record in records), key
    # The caller's verdict is kept under a key of its own until the record is written, never in it.
    assert all(capture_module._CALLER_SKIPPED not in record for record in records)


@pytest.mark.parametrize("name", TABLE)
def test_each_row_of_the_table(name):
    case = CASES[name]
    assert case["section"] == 1
    given = case["given"]
    assert given["init"]["auto_step"] is False
    scaler, verdict = SCALERS[given["init"].get("scaler")]
    backwards, skipped = _calls(given["calls"])
    records = _manual_run([(verdict, skipped)], scaler=scaler, backwards=backwards)
    _check(records, case["expected"])
    if name == "an_accumulation_microstep":
        # Four backward passes into one update: one record, step 0, whose gradients are the four's sum.
        assert [r["step"] for r in records] == [0]


def test_the_table_is_every_section_1_case_of_what_is_recorded():
    """The eight cases are the table's rows; none is left out here."""
    recorded = {name for name, case in CASES.items() if case["section"] == 1 and case.get("given", {}).get("calls")}
    assert recorded == set(TABLE)


def test_an_applied_step_with_a_scaler_and_skipped_false():
    """The first row's other half: a scaler that applied the step and `skipped=False`."""
    records = _manual_run([("applied", False)], scaler=True)
    _check(records, CASES["an_update_due_and_applied"]["expected"])
    assert records[0]["grad_scale"] == 1024.0 and "scaler_applied" not in records[0]


def test_nothing_said_with_a_scaler_that_applied_the_step():
    records = _manual_run([("applied", LEFT_OUT)], scaler=True)
    assert records[0]["step_skipped"] is False and "step_skipped_by" not in records[0]


def test_a_caller_skip_with_a_scaler_no_optimizer_went_through():
    """`init` was given a scaler, but no optimizer went through it this step: the scaler says nothing,
    so the record has the caller's verdict alone and no loss scale."""
    records = _manual_run([(None, True), (None, False), (None, LEFT_OUT)], scaler=True)
    assert [r.get("step_skipped", "-") for r in records] == [True, False, "-"]
    assert [r.get("step_skipped_by", "-") for r in records] == ["caller", "-", "-"]
    assert all("grad_scale" not in r and "scaler_applied" not in r for r in records)


def test_skips_land_on_their_own_steps_across_blocks():
    """Mixed verdicts over steps written in several blocks (`every=2`, and `every=3` so that a ring
    that filled keeps its newest step back), and an explicit flush in the middle."""
    plan = [
        ("applied", LEFT_OUT), ("skipped", LEFT_OUT), ("applied", True), ("skipped", True),
        ("applied", False), ("skipped", False), ("applied", LEFT_OUT), ("skipped", True),
    ]
    want_skipped = [False, True, True, True, False, True, False, True]
    want_by = ["-", "scaler", "caller", "both", "-", "scaler", "-", "both"]
    want_applied = ["-", "-", True, "-", "-", "-", "-", "-"]  # only on the caller's skip the scaler applied
    for every in (1, 2, 3):
        records = _manual_run(plan, scaler=True, every=every)
        assert [r["step"] for r in records] == list(range(len(plan)))
        assert [r["step_skipped"] for r in records] == want_skipped, every
        assert [r.get("step_skipped_by", "-") for r in records] == want_by, every
        assert [r.get("scaler_applied", "-") for r in records] == want_applied, every


def test_the_argument_under_several_optimizers():
    """Two optimizers over one model, stepped by hand: `skipped` is said once, for the update."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    first = torch.optim.SGD(model[0].parameters(), lr=0.1)
    second = torch.optim.Adam(model[1].parameters(), lr=0.01)
    amp = torch.amp.GradScaler("cpu", init_scale=1024.0)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=2, auto_step=False, scaler=amp)
    plan = [(None, True), (None, False), (1, LEFT_OUT), (1, True), (None, LEFT_OUT), (0, False)]
    for overflow_in, skipped in plan:  # which optimizer's gradients overflow, if any
        for opt in (first, second):
            opt.zero_grad()
        amp.scale(model(torch.ones(2, 4)).sum()).backward()
        if overflow_in is not None:
            model[overflow_in].weight.grad[0, 0] = float("inf")
        amp.step(first)
        amp.step(second)
        amp.update()
        session.step(**_step_kwargs(skipped))
    session.close()
    records = sink.records
    assert [r["step_skipped"] for r in records] == [True, False, True, True, False, True]
    assert [r.get("step_skipped_by", "-") for r in records] == ["caller", "-", "scaler", "both", "-", "scaler"]
    # Both optimizers went through the scaler and neither overflowed on step 0: it was seen applied.
    assert [r.get("scaler_applied", "-") for r in records] == [True, "-", "-", "-", "-", "-"]


def test_the_argument_with_lr_and_a_step_number():
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=2, auto_step=False)
    for i, (skipped, step) in enumerate([(True, 10), (False, 11), (True, None), (None, 20)]):
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        kwargs = {"lr": [0.1 * (i + 1)]} if i % 2 == 0 else {"lr": 0.5}
        if step is not None:
            kwargs["step"] = step
        session.step(**kwargs, **({} if skipped is None else {"skipped": skipped}))
    session.close()
    records = sink.records
    assert [r["step"] for r in records] == [10, 11, 12, 20]
    assert [r["lr"] for r in records] == [[0.1], [0.5], [pytest.approx(0.3)], [0.5]]
    assert [r.get("step_skipped", "-") for r in records] == [True, False, True, "-"]
    assert [r.get("step_skipped_by", "-") for r in records] == ["caller", "-", "caller", "-"]
    assert all("scaler_applied" not in r for r in records)  # no scaler: the outcome is unknown
    # The step number given positionally, as before the argument existed, with `skipped` beside it.
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=2, auto_step=False)
    model(torch.ones(2, 4)).sum().backward()
    session.step(5, 0.25, True)
    session.close()
    assert [(r["step"], r["lr"], r["step_skipped"], r["step_skipped_by"]) for r in sink.records] == [(5, [0.25], True, "caller")]


@pytest.mark.parametrize("wrong", [1, 0, "yes", "True", 1.0, torch.tensor(True), [True]])
def test_anything_but_true_false_or_none_is_refused_at_the_call(wrong):
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=1, auto_step=False)
    model(torch.ones(2, 4)).sum().backward()
    with pytest.raises(TypeError, match="skipped must be True, False or None"):
        session.step(skipped=wrong)
    # Nothing of the step was recorded by the refused call: no gradients read, no step ended.
    assert session._grads_recorded is False and session._steps_committed == 0
    session.step(skipped=True)  # the next call goes on as if the refused one had not been made
    session.close()
    assert [(r["step"], r["step_skipped"], r["step_skipped_by"]) for r in sink.records] == [(0, True, "caller")]


def test_none_is_the_argument_left_out():
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=1, auto_step=False)
    model(torch.ones(2, 4)).sum().backward()
    session.step(skipped=None)
    session.close()
    assert "step_skipped" not in sink.records[0] and "step_skipped_by" not in sink.records[0]


class _FailingSink(tensorquill.MemorySink):
    """Raises on every write, as a disk that is full: capture stops (`_stop`), training goes on."""

    def write(self, records):
        raise OSError("disk gone")


def _stopped_session() -> tuple[capture_module.Session, nn.Module]:
    """A manual-step session whose capture has stopped: its sink failed at the first write, and the
    next step on the training thread turned the writer's error into the stop (`_commit`)."""
    model = _model()
    session = tensorquill.init(model, sink=_FailingSink(), capture="full", every=1, auto_step=False)
    deadline = time.monotonic() + 30
    while not session._stopped and time.monotonic() < deadline:
        model(torch.ones(2, 4)).sum().backward()
        session.step()
        if session._error is None:
            time.sleep(0.01)  # give the writer thread its turn
    assert session._stopped and not session._closed
    return session, model


def test_on_a_closed_session_a_wrong_skipped_does_what_a_wrong_lr_does():
    """What a wrong `lr` does on a closed session: nothing. `step()` returns at once, raises nothing,
    records nothing and counts nothing (`_closed_call`). A wrong `skipped` does the same."""
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, every=1, auto_step=False)
    model(torch.ones(2, 4)).sum().backward()
    session.step()
    session.close()
    written, next_step = len(sink.records), session._next_step
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning either: one would raise under `python -W error`
        session.step(lr="not a number")
        assert (len(sink.records), session._next_step) == (written, next_step)
        for wrong in (1, "yes", torch.tensor(True)):
            session.step(skipped=wrong)
        session.step(lr="not a number", skipped=1)
    assert (len(sink.records), session._next_step) == (written, next_step)


def test_on_a_stopped_session_a_wrong_skipped_does_what_a_wrong_lr_does():
    """What a wrong `lr` does on a session whose capture has stopped: no raise and no warning (its one
    warning was given when it stopped), nothing recorded, and only the step count goes on, so the
    drift check's schedule stays with the other ranks (`_count_stopped`). A wrong `skipped` does the
    same, step for step."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the stop's own warning
        session, model = _stopped_session()
    first, committed, held = session._next_step, session._steps_committed, list(session._steps)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session.step(lr="not a number")
        assert session._next_step == first + 1
        session.step(skipped=1)
        assert session._next_step == first + 2
        session.step(skipped=torch.tensor(True))
        session.step(step=40, skipped="yes")
        assert session._next_step == 41
        session.step(lr="not a number", skipped=1)
        assert session._next_step == 42
    assert (session._steps_committed, session._steps) == (committed, held)  # nothing recorded
    session.close()


def test_on_a_live_session_a_wrong_lr_and_a_wrong_skipped_are_both_refused():
    """The sessions that refuse one refuse the other. A wrong `skipped` is refused before anything of
    the step is recorded (no gradients read); a wrong `lr` is refused after the gradients are read,
    as before this lane, and neither ends the step."""
    model = _model()
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=1, auto_step=False)
    model(torch.ones(2, 4)).sum().backward()
    with pytest.raises(TypeError, match="skipped must be True, False or None"):
        session.step(skipped=1)
    assert session._grads_recorded is False and session._steps_committed == 0
    with pytest.raises((TypeError, ValueError)):
        session.step(lr="not a number")
    assert session._steps_committed == 0
    with pytest.raises(TypeError, match="skipped must be True, False or None"):
        session.step(lr="not a number", skipped=1)  # both wrong: `skipped` is said first
    session.step(skipped=False)
    session.close()
    assert [(r["step"], r["step_skipped"]) for r in sink.records] == [(0, False)]


def test_an_auto_step_session_refuses_the_call_itself_whatever_skipped_is():
    """`step()` under `auto_step` raises its own RuntimeError before either argument is read."""
    model = _model()
    session = tensorquill.init(model, sink=tensorquill.MemorySink(), every=1)
    for kwargs in ({"skipped": 1}, {"lr": "not a number"}, {"skipped": True}):
        with pytest.raises(RuntimeError, match="auto_step=False"):
            session.step(**kwargs)
    session.close()


def _scaler_run(auto_step: bool, overflow_at: set[int], steps: int = 6) -> list[dict]:
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    amp = torch.amp.GradScaler("cpu", init_scale=1024.0)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=2, scaler=amp, auto_step=auto_step, run_id="s")
    for i in range(steps):
        opt.zero_grad()
        amp.scale(model(torch.ones(2, 4)).sum()).backward()
        if i in overflow_at:
            model.weight.grad[0, 0] = float("inf")
        amp.step(opt)
        amp.update()
        if not auto_step:
            session.step()
    session.close()
    return sink.records


@pytest.mark.parametrize("auto_step", [True, False])
def test_todays_scaler_path_gains_only_the_key_on_a_skipped_step(auto_step):
    """A run with a scaler and no argument: the scaler's fields are as before (test_scaler.py),
    and the only key added to any record is `step_skipped_by`, on the skipped steps alone (never
    `scaler_applied`, which is for a caller's skip)."""
    records = _scaler_run(auto_step, overflow_at={1, 2})
    assert [r["step_skipped"] for r in records] == [False, True, True, False, False, False]
    assert [r["grad_scale"] for r in records] == [1024.0, 1024.0, 512.0, 256.0, 256.0, 256.0]
    assert [r.get("step_skipped_by", "-") for r in records] == ["-", "scaler", "scaler", "-", "-", "-"]
    # Every other key is what a run with no skip writes on the same step: none added, none taken away.
    clean = _scaler_run(auto_step, overflow_at=set())
    for record, other in zip(records, clean, strict=True):
        assert set(record) - {"step_skipped_by"} == set(other)
        assert ("step_skipped_by" in record) == record["step_skipped"]


def test_step_skipped_by_and_scaler_applied_travel_in_extra_json(tmp_path, monkeypatch):
    """Neither is a column of the Arrow schema: each goes in `extra_json` and comes back unchanged.
    Read from the JSON Lines file the capture wrote, which is what the sender turns into rows."""
    monkeypatch.chdir(tmp_path)
    _manual_run([("applied", True), ("skipped", False), ("skipped", True), ("applied", False), (None, True)],
                scaler=True, jsonl=True)
    records = tensorquill.load_records(tmp_path / "tensorquill-out" / "ab3.rank0.jsonl")
    assert [r.get("step_skipped_by", "-") for r in records] == ["caller", "scaler", "both", "-", "caller"]
    assert [r.get("scaler_applied", "-") for r in records] == [True, "-", "-", "-", "-"]
    for record in records:
        row = record_to_row(record)
        extra = json.loads(row["extra_json"]) if row["extra_json"] else {}
        for key in ("step_skipped_by", "scaler_applied"):
            assert key not in row
            if key in record:
                assert extra[key] == record[key]
            else:
                assert key not in extra
        assert row["step_skipped"] == record["step_skipped"]
        assert row_to_record(row) == record


def test_step_skipped_by_is_not_a_column_of_the_schema():
    pytest.importorskip("pyarrow")
    from tensorquill.flight import flight_schema

    names = flight_schema().names
    assert "step_skipped" in names and "extra_json" in names
    assert "step_skipped_by" not in names and "scaler_applied" not in names


def _server():
    pytest.importorskip("pyarrow.flight")
    import test_flight_e2e

    if test_flight_e2e.BINARY is None:
        pytest.skip("tensorquill-server is not built (TENSORQUILL_SERVER_BIN)")
    return test_flight_e2e


def test_step_skipped_by_through_a_real_server(tmp_path, monkeypatch):
    e2e = _server()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TENSORQUILL_TOKEN", raising=False)
    store = tmp_path / "store"
    server = e2e.Server(store)
    try:
        records = _manual_run(
            [("applied", True), ("skipped", False), ("skipped", True), ("applied", False), ("skipped", LEFT_OUT),
             (None, True)],
            scaler=True, endpoint=server.endpoint,
        )
        written = tensorquill.load_records(tmp_path / "tensorquill-out" / "ab3.rank0.jsonl")
        assert [r.get("step_skipped_by", "-") for r in written] == ["caller", "scaler", "both", "-", "scaler", "caller"]
        assert [r.get("scaler_applied", "-") for r in written] == [True, "-", "-", "-", "-", "-"]
        assert records == []  # written to the file the sender sends from
        stored = e2e.stored_records(store, "ab3")
        assert stored == written
        assert tensorquill.load_records(store / "runs" / "ab3") == written
    finally:
        assert server.terminate() == 0


def test_a_caller_skip_before_the_scalers_update_has_no_scaler_outcome():
    """`scaler.step` ran but `scaler.update()` comes only after `session.step`: the scaler's verdict is
    read at `update()` (`_record_scaler`), so at the call there is none for this update. The outcome
    is unknown, and the record does not say the scaler applied it. (That verdict then lands on the
    next record, as it does today without the argument.)"""
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    amp = torch.amp.GradScaler("cpu", init_scale=1024.0)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=1, auto_step=False, scaler=amp)
    for i in range(2):
        opt.zero_grad()
        amp.scale(model(torch.ones(2, 4)).sum()).backward()
        amp.step(opt)
        session.step(skipped=i == 0)
        amp.update()
    session.close()
    first, second = sink.records
    assert first["step_skipped"] is True and first["step_skipped_by"] == "caller"
    assert "scaler_applied" not in first and "grad_scale" not in first
    assert second["step_skipped"] is False and "scaler_applied" not in second


def _order_run(plan: list[tuple[bool, bool, bool]]) -> list[dict]:
    """Each step is (an optimizer goes through `scaler.step`, the `skipped` given, `scaler.update()`
    comes after `session.step` rather than before it)."""
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    amp = torch.amp.GradScaler("cpu", init_scale=1024.0)
    sink = tensorquill.MemorySink()
    session = tensorquill.init(model, sink=sink, capture="full", every=1, auto_step=False, scaler=amp)
    for through, skipped, late in plan:
        opt.zero_grad()
        loss = model(torch.ones(2, 4)).sum()
        (amp.scale(loss) if through else loss).backward()
        if through:
            amp.step(opt)
            if not late:
                amp.update()
        else:
            opt.step()
        session.step(skipped=skipped)
        if through and late:
            amp.update()
    session.close()
    return sink.records


def test_a_late_verdict_of_an_earlier_update_never_says_the_scaler_applied_a_caller_skip():
    """With `scaler.update()` after `session.step`, each verdict reaches the record after its own
    update's (as today). It is of an update already ended, so it never gives `scaler_applied`: not
    on the next update's caller skip in the same order, and not on a next update that went to the
    optimizer directly. A next update whose own verdict came before its `session.step` has it."""
    late_then_late = _order_run([(True, False, True), (True, True, True)])
    late_then_direct = _order_run([(True, False, True), (False, True, False)])
    late_then_in_time = _order_run([(True, False, True), (True, True, False)])
    for records in (late_then_late, late_then_direct, late_then_in_time):
        assert [r.get("step_skipped_by", "-") for r in records] == ["-", "caller"]
    assert "scaler_applied" not in late_then_late[1]
    assert "scaler_applied" not in late_then_direct[1]
    assert late_then_in_time[1]["scaler_applied"] is True
    in_time = _order_run([(True, True, False), (True, True, False)])
    assert [r.get("scaler_applied", "-") for r in in_time] == [True, True]
