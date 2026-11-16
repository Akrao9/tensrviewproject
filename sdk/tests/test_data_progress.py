"""Data progress and checkpoint manifests: capture (`data.py`) and the detectors
(`data_checks.py`), each rule's positive and negative cases."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, TensorDataset

import tensorquill
from tensorquill.data import batch_hash, checkpoint_manifest, files_manifest
from tensorquill.data_checks import (
    detect_data_stalled,
    detect_duplicate_batches,
    detect_epoch_order_repeated,
    detect_resume_state,
)


def hx(tag: object) -> str:
    return hashlib.blake2b(str(tag).encode(), digest_size=8).hexdigest()


def rec(step: int, attempt: int = 1, rank: int = 0, **values) -> dict:
    return {"schema_version": 1, "run_id": "r", "rank": rank, "attempt": attempt, "step": step, **values}


def drew(tag: object, **fields) -> dict:
    return {"batches": 1, "batch_hash": hx(tag), **fields}


# --- data_stalled -------------------------------------------------------------------------------


def test_three_steps_without_a_batch_are_a_stall_dated_at_the_first():
    records = [rec(s, data=drew(s) if s < 5 or s > 7 else {"batches": 0}) for s in range(10)]
    hit = detect_data_stalled(records)
    assert hit == {"kind": "data_stalled", "step": 5, "reason": "no_batch", "through_step": 7}


def test_the_same_batch_three_steps_running_is_a_stall():
    records = [rec(s, data=drew(min(s, 4))) for s in range(8)]  # steps 4..7 draw batch 4
    hit = detect_data_stalled(records)
    assert (hit["step"], hit["reason"], hit["through_step"], hit["batch_hash"]) == (5, "same_batch", 7, hx(4))


@pytest.mark.parametrize("records", [
    [rec(s, data=drew(s) if s not in (3, 4) else {"batches": 0}) for s in range(10)],  # two steps only
    [rec(s, data=drew(s)) for s in range(10)],  # healthy
    # A step without `data` ends a stretch and tells nothing.
    [rec(s, data={"batches": 0}) if s != 2 else rec(s) for s in range(5)],
    # Batches without hashes cannot be the same batch.
    [rec(s, data={"batches": 1}) for s in range(6)],
    [rec(s) for s in range(6)],  # no data capture at all
])
def test_no_stall(records):
    assert detect_data_stalled(records) is None


# --- epoch_order_repeated -----------------------------------------------------------------------


def epochs(orders: list[str], length: int = 5, **fields) -> list[dict]:
    """One record per batch; epoch e draws the batches of order `orders[e]`."""
    return [
        rec(e * length + p, data=drew(f"{order}-{p}", epoch=e, sampler_epoch=0, shuffle=True, **fields))
        for e, order in enumerate(orders) for p in range(length)
    ]


def test_an_epoch_in_the_previous_epochs_order_is_reported_at_its_first_step():
    hit = detect_epoch_order_repeated(epochs(["a", "b", "b"]))
    assert hit == {
        "kind": "epoch_order_repeated", "step": 10, "epoch": 2, "previous_epoch": 1, "batches_compared": 3,
        "shuffle": True, "sampler_epoch": 0, "previous_sampler_epoch": 0, "through_step": 12,
    }


def test_a_two_batch_epoch_is_compared_over_both():
    hit = detect_epoch_order_repeated(epochs(["a", "a"], length=2))
    assert (hit["step"], hit["batches_compared"]) == (2, 2)


@pytest.mark.parametrize("records", [
    epochs(["a", "b", "c"]),  # a new order each epoch
    [{**r, "data": {**r["data"], "shuffle": False}} for r in epochs(["a", "a"])],  # meant to keep its order
    epochs(["a", "a"], length=1),  # one batch an epoch: the stall rule's case, not this one
    # Epochs 0 and 2 are not consecutive.
    [r for r in epochs(["a", "b", "a"]) if r["data"]["epoch"] != 1],
    # The same first two batches, then a different third: an order that only starts alike.
    epochs(["a", "b"])[:5] + [rec(5 + p, data=drew(f"a-{p}" if p < 2 else f"z-{p}", epoch=1, shuffle=True)) for p in range(5)],
    # One batch over and over is `data_stalled`.
    [rec(s, data=drew("same", epoch=s // 4, shuffle=True)) for s in range(12)],
])
def test_no_repeated_order(records):
    assert detect_epoch_order_repeated(records) is None


def test_an_unknown_shuffle_is_compared_and_said():
    records = [{**r, "data": {k: v for k, v in r["data"].items() if k != "shuffle"}} for r in epochs(["a", "a"])]
    assert detect_epoch_order_repeated(records)["shuffle"] is None


# --- duplicate_batches --------------------------------------------------------------------------


def test_two_ranks_drawing_the_same_batch_are_duplicates():
    records = []
    for s in range(20):
        records.append(rec(s, rank=0, data=drew(f"r0-{s}")))
        records.append(rec(s, rank=1, data=drew(f"r0-{s}" if 5 <= s < 9 else f"r1-{s}")))
        records.append(rec(s, rank=2, data=drew(f"r0-{s}", data_rank=0)))  # rank 0's tensor-parallel peer
    # Rank 2 declares data rank 0 and rank 0's own number stands in for its share: the same
    # share, so the two match on every step without being duplicates. Rank 1 joins them at 5-8.
    assert detect_duplicate_batches(records) == {
        "kind": "duplicate_batches", "step": 5, "ranks": [0, 1, 2], "duplicated_steps": 4, "ranks_compared": [0, 1, 2]}


@pytest.mark.parametrize("records", [
    [rec(s, rank=r, data=drew(f"r{r}-{s}")) for s in range(5) for r in (0, 1)],
    [rec(s, rank=r, data=drew(s, data_rank=0)) for s in range(5) for r in (0, 1)],  # one share, two ranks
    [rec(s, rank=0, data=drew(s)) for s in range(5)],  # one rank
    [rec(s, rank=r, data={"batches": 1}) for s in range(5) for r in (0, 1)],  # nothing hashed
])
def test_no_duplicates(records):
    assert detect_duplicate_batches(records) is None


# --- resumes ------------------------------------------------------------------------------------


def run(attempt: int, steps, tag=lambda s: s, **fields) -> list[dict]:
    return [rec(s, attempt, data=drew(tag(s)), **{k: v(s) if callable(v) else v for k, v in fields.items()}) for s in steps]


def data_changes(records):
    not_checked: list = []
    hit, _ = detect_resume_state(records, not_checked=not_checked)
    return hit, [n for n in not_checked if n["check"] == "resume_data"]


def test_a_resume_that_draws_the_batch_the_run_drew_there_is_fine():
    assert data_changes(run(1, range(20)) + run(2, range(10, 20))) == (None, [])


def test_a_resume_that_replays_an_earlier_batch_is_repeated():
    hit, _ = data_changes(run(1, range(20)) + run(2, range(10, 20), tag=lambda s: s - 7))
    assert hit["kind"] == "resume_data_mismatch" and hit["step"] == 10
    change = hit["changes"][0]
    assert {k: change[k] for k in ("basis", "finding", "matched_step", "batches_apart", "previous_attempt")} == {
        "basis": "same_step", "finding": "repeated", "matched_step": 3, "batches_apart": 7, "previous_attempt": 1}


def test_a_resume_that_jumps_ahead_is_skipped():
    hit, _ = data_changes(run(1, range(20)) + run(2, range(10, 20), tag=lambda s: s + 4))
    assert (hit["changes"][0]["finding"], hit["changes"][0]["matched_step"]) == ("skipped", 14)


def test_a_resume_with_a_batch_never_drawn_differs_at_the_same_step_and_is_unknown_past_the_end():
    hit, skipped = data_changes(run(1, range(20)) + run(2, range(10, 20), tag=lambda s: f"new-{s}"))
    assert hit["changes"][0]["finding"] == "differs" and skipped == []
    # The earlier attempt stopped at 9: a new batch at 10 is what a correct resume draws.
    hit, skipped = data_changes(run(1, range(10)) + run(2, range(10, 20), tag=lambda s: f"new-{s}"))
    assert hit is None and [s["step"] for s in skipped] == [10]
    # Past the end, a batch drawn before is still found: the epoch started over.
    hit, _ = data_changes(run(1, range(10)) + run(2, range(10, 20), tag=lambda s: s - 10))
    assert (hit["changes"][0]["basis"], hit["changes"][0]["finding"], hit["changes"][0]["matched_step"]) == ("lookup", "repeated", 0)


def test_a_later_attempt_rolls_back_what_it_replaced_before_the_next_resume_is_judged():
    # Attempt 2 resumed at 10 and stopped at 14, so attempt 1's batches from 10 on are no longer
    # the run's history: attempt 3, resumed at 20 with attempt 1's batch for 20, cannot be judged.
    hit, skipped = data_changes(run(1, range(30)) + run(2, range(10, 15), tag=lambda s: f"b-{s}") + run(3, range(20, 25)))
    assert [(c["step"], c["finding"]) for c in hit["changes"]] == [(10, "differs")]
    assert [s["step"] for s in skipped] == [20]


def test_a_restart_of_the_numbering_and_a_run_without_data_are_not_judged():
    assert data_changes(run(1, range(20)) + run(2, range(0, 5), tag=lambda s: 99)) == (None, [])
    plain = [rec(s, 1) for s in range(10)] + [rec(s, 2) for s in range(5, 10)]
    not_checked: list = []
    assert detect_resume_state(plain, not_checked=not_checked) == (None, None) and not_checked == []


FULL = {"model": {"tensors": 4}, "optimizer": {"entries": 4, "param_groups": 1, "step": 10},
        "scheduler": {"last_epoch": 10}, "scaler": {"scale": 1024.0}, "rng": {}, "data": {}}


def state(records):
    not_checked: list = []
    _, hit = detect_resume_state(records, not_checked=not_checked)
    return hit, [n for n in not_checked if n["check"] == "resume_state"]


def scaled(attempt, steps, scale, **fields):
    return run(attempt, steps, lr=lambda s: [1e-3 * min(1, (s + 1) / 5)], grad_scale=scale, **fields)


def test_a_complete_resume_finds_nothing():
    first = scaled(1, range(20), lambda s: 65536.0 if s < 5 else 1024.0)
    resumed = scaled(2, range(10, 20), 1024.0)
    resumed[0]["checkpoint"] = {"load": {"components": FULL},
                                "optimizer": {"entries_before": 4, "entries_after": 4, "step_after": 11}}
    assert state(first + resumed) == (None, [])


def test_a_resume_that_restores_too_little():
    first = scaled(1, range(20), lambda s: 65536.0 if s < 5 else 1024.0)
    resumed = scaled(2, range(10, 20), 65536.0)
    partial = {k: v for k, v in FULL.items() if k in ("model", "optimizer", "scheduler")}
    resumed[0]["checkpoint"] = {
        "load": {"components": {**partial, "scheduler": {"last_epoch": 0}}},
        "optimizer": {"entries_before": 0, "entries_after": 4, "step_after": 1},
    }
    hit, _ = state(first + resumed)
    assert hit["kind"] == "resume_incomplete" and hit["step"] == 10
    assert hit["changes"][0]["source"] == "load"
    assert [(f["component"], f["finding"]) for f in hit["changes"][0]["findings"]] == [
        ("scaler", "missing"), ("rng", "missing"), ("data", "missing"), ("scheduler", "position"),
        ("optimizer", "empty"), ("scaler", "initial_scale")]


def test_an_optimizer_that_kept_its_state_but_restarted_its_step():
    first = run(1, range(20))
    resumed = run(2, range(10, 20))
    resumed[0]["checkpoint"] = {"optimizer": {"entries_before": 4, "entries_after": 4, "step_after": 1.0}}
    hit, _ = state(first + resumed)
    assert hit["changes"][0]["findings"] == [{"component": "optimizer", "finding": "step_reset", "step_after": 1.0}]


def test_without_a_load_manifest_the_last_save_before_the_resume_is_assumed():
    first = run(1, range(20))
    first[5]["checkpoint"] = {"save": {"components": {"model": {}, "optimizer": {"entries": 0}}, "after_step": 5}}
    first[8]["checkpoint"] = {"save": {"components": {"model": {}, "optimizer": {"entries": 3}, "rng": {}, "data": {}},
                                       "after_step": 7}}
    first[15]["checkpoint"] = {"save": {"components": {}, "after_step": 15}}  # after the resume point
    # The save after step 7 held everything this run needs (no schedule, no scaler): nothing to say.
    assert state(first + run(2, range(10, 20))) == (None, [])


def test_the_save_fallback_reports_what_the_save_left_out():
    first = run(1, range(20))
    first[9]["checkpoint"] = {"save": {"components": {"model": {}, "optimizer": {"entries": 0}}, "after_step": 9}}
    hit, _ = state(first + run(2, range(10, 20)))
    change = hit["changes"][0]
    assert (change["source"], change["saved_after_step"]) == ("save", 9)
    assert [(f["component"], f["finding"]) for f in change["findings"]] == [
        ("rng", "missing"), ("data", "missing"), ("optimizer", "empty_in_checkpoint")]


def test_a_resume_with_nothing_to_judge_is_listed_only_when_the_run_records_checkpoints():
    first = run(1, range(20))
    first[3]["checkpoint"] = {"save": {"components": FULL, "after_step": 3}}
    first[3]["checkpoint"]["save"]["after_step"] = 30  # a save the resume cannot have loaded
    hit, skipped = state(first + run(2, range(10, 20)))
    assert hit is None and [s["step"] for s in skipped] == [10]
    assert state(run(1, range(20)) + run(2, range(10, 20))) == (None, [])


# --- capture ------------------------------------------------------------------------------------


def train(session, loader, model, opt, epochs=1):
    for _ in range(epochs):
        for x, y in loader:
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


def make(tmp_path, name="run", **kwargs):
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    session = tensorquill.init(model, run_id=name, every=4, sink=tensorquill.JsonlSink(tmp_path / f"{name}.jsonl"),
                               heartbeat_seconds=None, **kwargs)
    return model, opt, session


def read(tmp_path, name="run"):
    return [json.loads(line) for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()]


DATA = TensorDataset(torch.arange(24.0).reshape(24, 1).repeat(1, 4), torch.zeros(24, 1))


@pytest.mark.parametrize("workers", [0, 2])
def test_watch_loader_hashes_the_indices_of_each_batch(tmp_path, workers):
    model, opt, session = make(tmp_path)
    loader = DataLoader(DATA, batch_size=4, shuffle=True, num_workers=workers, persistent_workers=workers > 0)
    watched = session.watch_loader(loader)
    seen = []
    for epoch in range(2):
        for x, y in watched:
            seen.append(x[:, 0].long().tolist())  # the samples are their own indices
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    session.close()
    records = read(tmp_path)
    assert [r["data"]["batch_hash"] for r in records] == [batch_hash(i) for i in seen]
    assert [r["data"]["epoch"] for r in records] == [0] * 6 + [1] * 6
    assert all(r["data"]["shuffle"] is True and r["data"]["batches"] == 1 for r in records)
    assert len(watched) == 6 and watched.dataset is DATA


def test_micro_batches_and_steps_without_a_batch(tmp_path):
    model, opt, session = make(tmp_path)
    for step in range(3):
        if step != 1:
            session.data([step, step + 1])
            session.data([step + 2])
        model(torch.zeros(1, 4)).sum().backward()
        opt.step()
    session.close()
    records = read(tmp_path)
    assert [r["data"] for r in records] == [
        {"batches": 2, "batch_hash": batch_hash([0, 1], [2])}, {"batches": 0}, {"batches": 2, "batch_hash": batch_hash([2, 3], [4])}]
    assert batch_hash([0, 1], [2]) != batch_hash([0], [1, 2])


def test_a_forgotten_set_epoch_is_caught_and_a_called_one_is_not(tmp_path):
    for name, call in (("forgot", False), ("called", True)):
        model, opt, session = make(tmp_path, name)
        sampler = DistributedSampler(DATA, num_replicas=2, rank=1, shuffle=True)
        watched = session.watch_loader(DataLoader(DATA, batch_size=3, sampler=sampler))
        for epoch in range(3):
            if call:
                sampler.set_epoch(epoch)
            train(session, watched, model, opt)
        session.close()
        evidence = tensorquill.build_evidence(read(tmp_path, name))
        kinds = [i["kind"] for i in evidence["incidents"]]
        if call:
            assert "epoch_order_repeated" not in kinds
        else:
            [hit] = [i for i in evidence["incidents"] if i["kind"] == "epoch_order_repeated"]
            assert (hit["step"], hit["epoch"], hit["sampler_epoch"], hit["previous_sampler_epoch"]) == (4, 1, 0, 0)
            assert "set_epoch" in tensorquill.incident_summary(evidence)
        assert all(r["data"]["data_rank"] == 1 for r in read(tmp_path, name))


def test_an_iterable_dataset_needs_ids_and_takes_them(tmp_path):
    class Stream(IterableDataset):
        def __iter__(self):
            return iter((torch.full((4,), float(i)), torch.zeros(1), i) for i in range(8))

    model, opt, session = make(tmp_path)
    for x, y, ids in session.watch_loader(DataLoader(Stream(), batch_size=2), ids=lambda batch: batch[2]):
        model(x).sum().backward()
        opt.step()
    for x, y, ids in session.watch_loader(DataLoader(Stream(), batch_size=2)):
        model(x).sum().backward()
        opt.step()
    session.close()
    records = read(tmp_path)
    assert [r["data"].get("batch_hash") for r in records[:4]] == [batch_hash([2 * i, 2 * i + 1]) for i in range(4)]
    assert all("batch_hash" not in r["data"] and r["data"]["batches"] == 1 for r in records[4:])  # no indices: no hash


def test_a_failing_ids_callable_warns_and_training_goes_on(tmp_path):
    model, opt, session = make(tmp_path)
    loader = session.watch_loader(DataLoader(DATA, batch_size=8), ids=lambda batch: 1 / 0)
    with pytest.warns(RuntimeWarning, match="could not note a batch"):
        train(session, loader, model, opt)
    session.close()
    assert len(read(tmp_path)) == 3


def test_checkpoint_manifests_say_what_a_save_holds_and_where(tmp_path):
    model, opt, session = make(tmp_path)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    for step in range(8):
        model(torch.zeros(1, 4)).sum().backward()
        opt.step()
        sched.step()
        if step in (3, 5):  # after step 3 the ring was just flushed (every=4); after 5 it holds the step
            session.checkpoint({
                "model": model.state_dict(), "optimizer": opt.state_dict(), "lr_scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(), "rng_state": torch.get_rng_state(), "step": step + 1,
            })
    session.close()
    records = read(tmp_path)
    assert "checkpoint" not in records[3]
    save = records[4]["checkpoint"]["save"]  # recorded with the next step, saying which it follows
    assert save["after_step"] == 3 and records[5]["checkpoint"]["save"]["after_step"] == 5
    components = save["components"]
    assert list(components) == ["model", "optimizer", "scheduler", "scaler", "rng"]
    assert components["optimizer"]["entries"] == 2 and components["optimizer"]["step"] == 4
    assert components["scheduler"]["last_epoch"] == 4 and components["scaler"]["scale"] == 1024.0
    assert components["model"]["tensors"] == 2 and save["counters"] == {"step": 4}


def test_lightning_and_nested_checkpoints_are_read_by_shape():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)
    manifest = checkpoint_manifest({
        "epoch": 3, "global_step": 120, "state_dict": model.state_dict(), "optimizer_states": [opt.state_dict()],
        "lr_schedulers": [{"last_epoch": 120, "base_lrs": [0.1]}], "loops": {"fit_loop": {}},
        "MixedPrecision": {"scale": 512.0, "growth_factor": 2.0, "_growth_tracker": 0}, "hyper_parameters": {},
        "training": {"rng": {"cpu": torch.get_rng_state()}},
    })
    assert manifest["components"]["optimizer"] == {"entries": 0, "param_groups": 1, "count": 1, "keys": ["optimizer_states"]}
    assert manifest["components"]["scheduler"]["last_epoch"] == 120
    assert manifest["components"]["data"]["keys"] == ["loops"]
    assert manifest["components"]["rng"]["keys"] == ["training.rng"]
    assert manifest["counters"] == {"epoch": 3, "global_step": 120} and manifest["other"] == ["hyper_parameters"]
    assert list(files_manifest(["optimizer.pt", "scheduler.pt", "rng_state_0.pth", "model.safetensors", "trainer_state.json",
                                "config.json"])["components"]) == ["model", "optimizer", "scheduler", "rng", "data"]
    assert list(files_manifest(["pytorch_model.bin", "optimizer.bin", "random_states_0.pkl", "sampler.bin", "scaler.pt"])[
        "components"]) == ["model", "optimizer", "scaler", "rng", "data"]  # Accelerate's save_state


def test_a_resumed_session_records_whether_its_optimizer_came_back(tmp_path):
    model, opt, session = make(tmp_path, "first")
    for _ in range(3):
        model(torch.zeros(1, 4)).sum().backward()
        opt.step()
    state = copy.deepcopy({"model": model.state_dict(), "optimizer": opt.state_dict()})
    session.close()
    assert all("checkpoint" not in r for r in read(tmp_path, "first"))  # a fresh session records nothing
    for name, restore in (("restored", True), ("forgotten", False)):
        new_opt = torch.optim.Adam(model.parameters(), 1e-2)
        if restore:
            new_opt.load_state_dict(state["optimizer"])
        resumed = tensorquill.init(model, run_id=name, every=4, start_step=3, heartbeat_seconds=None,
                                   sink=tensorquill.JsonlSink(tmp_path / f"{name}.jsonl"))
        resumed.checkpoint(state, event="load")
        model(torch.zeros(1, 4)).sum().backward()
        new_opt.step()
        resumed.close()
        [record] = read(tmp_path, name)
        snapshot = record["checkpoint"]["optimizer"]
        assert "optimizer" in record["checkpoint"]["load"]["components"]
        assert (snapshot["entries_before"], snapshot["step_after"]) == ((2, 4.0) if restore else (0, 1.0))


def test_data_and_checkpoint_fields_cross_the_wire_unchanged():
    pytest.importorskip("pyarrow")
    from tensorquill.flight import record_to_row, records_to_batch, row_to_record

    record = rec(4, data=drew(4, epoch=1, shuffle=None, shard="s-3", offset=99),
                 checkpoint={"save": {"components": FULL, "after_step": 3}, "optimizer": {"entries_before": 0}})
    row = record_to_row(record)
    assert row["extra_json"] is None and json.loads(row["data_json"])["shard"] == "s-3"
    back = row_to_record(records_to_batch([record]).to_pylist()[0])
    assert (back["ts"], back["grad_norm"]) == (None, None)  # always written, as before
    assert {k: v for k, v in back.items() if k not in ("ts", "grad_norm")} == record


def test_existing_kinds_are_unchanged_on_runs_without_the_new_fields():
    # Runs written before these fields existed give the same evidence: nothing is added.
    records = run(1, range(5))
    for r in records:
        del r["data"]
    evidence = tensorquill.build_evidence(records + [rec(s, 2) for s in range(3, 6)])
    assert evidence["incidents"] == [] and "not_checked" not in evidence or all(
        n["check"] == "resume_lr" for n in evidence.get("not_checked", []))
