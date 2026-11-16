"""Detectors over the `data` and `checkpoint` fields (see `data.py`): the reference for the
Rust ones in `server/src/detect.rs`.

Each per-rank detector is one forward pass over the records in step order, holding only
what the server's stream holds, so any prefix of a run gives the incidents the whole run
gives up to there. An incident dated before the row that makes it (a stall's or an epoch's
first step) is never dated again.

- `data_stalled`: steps go on but the data does not: `DATA_STALL_STEPS` recorded steps in a
  row drew no batch, or drew the batch the step before drew.
- `epoch_order_repeated`: a pass over the data draws its first batches in the order the pass
  before drew them, while the loader is not known to keep its order (`shuffle` is not
  False): a `DistributedSampler` whose `set_epoch` is never called.
- `duplicate_batches`: two ranks that read different shares of the data drew the same batch
  at the same step. Cross-rank: `detect_duplicate_batches` reads every rank; the server
  writes it into the run's `watchdog.json`.
- `resume_data_mismatch` and `resume_incomplete`: judged at the first step of each resumed
  attempt, as `resume_lr_change` is. The first compares the batch drawn there with the run's
  history (the same step when an earlier attempt reached it, else the step where that batch
  was drawn before: repeated or skipped). The second says what the resume did not bring
  back: components the checkpoint did not hold, an optimizer that starts empty or with its
  step counter at 1, a scheduler whose `last_epoch` is far from the step, a loss scale back
  at its initial value. Both are observations: some resumes do these things on purpose.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

DATA_STALL_STEPS = 3  # recorded steps in a row with no new batch
ORDER_MIN_BATCHES = 3  # an epoch's first batches that must match the previous epoch's
ORDER_COMPARE = 64  # positions of each epoch kept for the comparison
DUPLICATE_WINDOW = 4096  # each rank's most recent hashed steps compared across ranks
SCHEDULER_SLACK = 1  # `last_epoch` may differ from the resume step by this much
CHANGES_LISTED = 5


def _data(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    data = record.get("data")
    return data if isinstance(data, Mapping) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _hash(data: Mapping[str, Any]) -> str | None:
    value = data.get("batch_hash")
    return value if isinstance(value, str) and value else None


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _by_step(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: int(r["step"]))


def detect_data_stalled(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first stretch of `DATA_STALL_STEPS` recorded steps that drew no new batch.

    A record without `data` (capture not on yet, or a hand-made record) ends a stretch and
    tells nothing. `same_batch` needs the hashes; `no_batch` only the count.
    """
    run, first, reason, previous = 0, 0, "", None
    for record in _by_step(records):
        data = _data(record)
        if data is None:
            run, previous = 0, None
            continue
        batches, digest = _int(data.get("batches")), _hash(data)
        if batches == 0:
            why = "no_batch"
        elif batches is not None and batches > 0 and digest is not None and digest == previous:
            why = "same_batch"
        else:
            why = None
        if batches is not None and batches > 0:
            previous = digest
        if why is None:
            run = 0
            continue
        if run == 0:
            first, reason = int(record["step"]), why
        run += 1
        if run == DATA_STALL_STEPS:
            incident: dict[str, Any] = {
                "kind": "data_stalled", "step": first, "reason": reason, "through_step": int(record["step"]),
            }
            if reason == "same_batch":
                incident["batch_hash"] = previous
            return incident
    return None


def detect_epoch_order_repeated(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first pass whose first batches repeat the previous pass's order.

    Steps that drew batches with an `epoch` are read; a pass is compared with the one
    numbered just before it, position by position from its start, up to `ORDER_COMPARE`
    positions. It is reported once `min(ORDER_MIN_BATCHES, length of the previous pass)`
    positions (at least two) matched with no mismatch, among them two different batches (a
    loader that draws one batch over and over is `data_stalled`). A pass recorded with
    `shuffle` False is meant to keep its order and is not compared.
    """
    epoch: int | None = None
    current: list[str | None] = []
    count = first = 0
    shuffle: bool | None = None
    sampler_epoch: int | None = None
    previous: list[str | None] | None = None
    previous_count = 0
    previous_epoch: int | None = None
    previous_sampler: int | None = None
    equal = True
    for record in _by_step(records):
        data = _data(record)
        if data is None or not ((_int(data.get("batches")) or 0) > 0):
            continue
        number = _int(data.get("epoch"))
        if number is None:
            continue
        digest = _hash(data)
        if number != epoch:
            if epoch is not None and number == epoch + 1:
                previous, previous_count, previous_epoch, previous_sampler = current, count, epoch, sampler_epoch
            else:
                previous = None
            epoch, current, count, first, equal = number, [], 0, int(record["step"]), True
            shuffle, sampler_epoch = _bool(data.get("shuffle")), _int(data.get("sampler_epoch"))
        position = count
        count += 1
        if position < ORDER_COMPARE:
            current.append(digest)
        if previous is None or not equal or shuffle is False or position >= ORDER_COMPARE:
            continue
        if digest is None or position >= len(previous) or previous[position] != digest:
            equal = False
            continue
        needed = min(ORDER_MIN_BATCHES, previous_count)
        if needed >= 2 and position + 1 >= needed and len(set(current)) > 1:
            return {
                "kind": "epoch_order_repeated",
                "step": first,
                "epoch": epoch,
                "previous_epoch": previous_epoch,
                "batches_compared": position + 1,
                "shuffle": shuffle,
                "sampler_epoch": sampler_epoch,
                "previous_sampler_epoch": previous_sampler,
                "through_step": int(record["step"]),
            }
    return None


def detect_duplicate_batches(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Steps at which ranks reading different shares of the data drew the same batch.

    `records` are several ranks' (after `resolve_attempts`). Each rank's last
    `DUPLICATE_WINDOW` steps with a `batch_hash` are compared; ranks with the same
    `data_rank` (a tensor-parallel group reading one share) are meant to match and are not a
    duplicate. The rank's own number stands in when `data_rank` is not recorded.
    """
    by_rank: dict[int, list[tuple[int, int, str]]] = {}
    for record in _by_step(records):
        data = _data(record)
        digest = _hash(data) if data is not None else None
        if data is None or digest is None or not ((_int(data.get("batches")) or 0) > 0):
            continue
        rank = int(record.get("rank", 0))
        share = _int(data.get("data_rank"))
        by_rank.setdefault(rank, []).append((int(record["step"]), rank if share is None else share, digest))
    if len(by_rank) < 2:
        return None
    at: dict[int, list[tuple[int, int, str]]] = {}
    for rank, rows in by_rank.items():
        for step, share, digest in rows[-DUPLICATE_WINDOW:]:
            at.setdefault(step, []).append((rank, share, digest))
    first: tuple[int, list[int]] | None = None
    duplicated = 0
    for step in sorted(at):
        groups: dict[str, list[tuple[int, int]]] = {}
        for rank, share, digest in at[step]:
            groups.setdefault(digest, []).append((rank, share))
        involved = sorted(
            {rank for group in groups.values() if len({share for _, share in group}) > 1 for rank, _ in group}
        )
        if involved:
            duplicated += 1
            if first is None:
                first = (step, involved)
    if first is None:
        return None
    return {
        "kind": "duplicate_batches",
        "step": first[0],
        "ranks": first[1],
        "duplicated_steps": duplicated,
        "ranks_compared": sorted(by_rank),
    }


# --- resumes -------------------------------------------------------------------------------


def _lookup(history: Mapping[int, tuple[str | None, int]], digest: str, resumed_at: int) -> int | None:
    """The step nearest `resumed_at` (the earlier one on a tie) at which the history drew `digest`."""
    before = max((s for s, (h, _) in history.items() if h == digest and s < resumed_at), default=None)
    after = min((s for s, (h, _) in history.items() if h == digest and s > resumed_at), default=None)
    if before is None:
        return after
    if after is None or resumed_at - before <= after - resumed_at:
        return before
    return after


def _compare_data(
    history: Mapping[int, tuple[str | None, int]], new: str | None, resumed_at: int
) -> dict[str, Any] | str | None:
    known = {s: v for s, v in history.items() if v[0] is not None}
    if new is None:
        return "no batch hash recorded at the first step of the resumed attempt" if known else None
    if not known:
        return "no batch hash recorded in the earlier attempts"
    expected = known.get(resumed_at)
    if expected is not None and expected[0] == new:
        return None
    matched = _lookup(known, new, resumed_at)
    if matched is None and expected is None:
        return (
            "the earlier attempts ended before the resume step and drew the resumed batch nowhere: a resume that"
            " repeats no batch looks the same as one that skips"
        )
    finding = "differs" if matched is None else ("repeated" if matched < resumed_at else "skipped")
    return {
        "basis": "same_step" if expected is not None else "lookup",
        "finding": finding,
        "resumed_hash": new,
        "expected_hash": expected[0] if expected is not None else None,
        "matched_step": matched,
        "batches_apart": None if matched is None else abs(resumed_at - matched),
        "previous_attempt": (expected if expected is not None else known[matched])[1],  # type: ignore[index]
    }


def _expected_components(lr_varies: bool, scale_seen: bool) -> list[str]:
    return ["model", "optimizer", *(["scheduler"] if lr_varies else []), *(["scaler"] if scale_seen else []), "rng", "data"]


def _state_findings(
    first: Mapping[str, Any],
    resumed_at: int,
    run_start: int,
    saved: tuple[int, Mapping[str, Any]] | None,
    lr_varies: bool,
    first_scale: float | None,
    previous_scale: float | None,
) -> tuple[list[dict[str, Any]], str | None, bool]:
    """What a resume did not bring back, where the list came from, and whether anything was checked."""
    checkpoint = first.get("checkpoint") if isinstance(first.get("checkpoint"), Mapping) else {}
    findings: list[dict[str, Any]] = []
    checked = False
    load = checkpoint.get("load")
    manifest, source = (load, "load") if isinstance(load, Mapping) else (
        (saved[1], "save") if saved is not None else (None, None)
    )
    if manifest is not None:
        checked = True
        components = manifest.get("components") if isinstance(manifest.get("components"), Mapping) else {}
        for name in _expected_components(lr_varies, first_scale is not None):
            if name not in components:
                findings.append({"component": name, "finding": "missing"})
        optimizer = components.get("optimizer")
        if isinstance(optimizer, Mapping) and _int(optimizer.get("entries")) == 0:
            findings.append({"component": "optimizer", "finding": "empty_in_checkpoint"})
        scheduler = components.get("scheduler")
        last_epoch = _int(scheduler.get("last_epoch")) if isinstance(scheduler, Mapping) else None
        if last_epoch is not None and abs(last_epoch - resumed_at) > SCHEDULER_SLACK:
            findings.append({"component": "scheduler", "finding": "position", "last_epoch": last_epoch})
    snapshot = checkpoint.get("optimizer")
    if isinstance(snapshot, Mapping):
        checked = True
        before, after = _int(snapshot.get("entries_before")), _int(snapshot.get("entries_after"))
        step_after = _number(snapshot.get("step_after"))
        if before == 0 and after is not None and after > 0:
            findings.append({"component": "optimizer", "finding": "empty", "entries_after": after})
        elif step_after is not None and step_after <= 1 and resumed_at - run_start > 1:
            findings.append({"component": "optimizer", "finding": "step_reset", "step_after": step_after})
    scale = _number(first.get("grad_scale"))
    if scale is not None and first_scale is not None and previous_scale is not None:
        checked = True
        if scale == first_scale and previous_scale != first_scale:
            findings.append({"component": "scaler", "finding": "initial_scale", "scale": scale,
                             "expected_scale": previous_scale})
    return findings, source, checked


def detect_resume_state(
    records: list[dict[str, Any]], *, not_checked: list[dict[str, Any]] | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """`resume_data_mismatch` and `resume_incomplete` over every attempt
    (`load_records(..., all_attempts=True)`), or None for either.

    At the first step S of each later attempt that resumed from a checkpoint (it starts after
    the run's first step; a restart of the numbering is `repeated_steps`), the run's history
    as it stood before that attempt (every earlier attempt with each rollback applied) is
    the reference. A resume that cannot be compared is listed in `not_checked`, only for a run
    that records the field at all, so a run without data or checkpoint capture adds nothing.
    """
    by_rank: dict[Any, dict[int, dict[int, dict[str, Any]]]] = {}
    for record in records:
        attempts = by_rank.setdefault(record.get("rank", 0), {})
        attempts.setdefault(record.get("attempt", 0), {})[int(record["step"])] = record
    data_changes: list[dict[str, Any]] = []
    state_changes: list[dict[str, Any]] = []
    data_skipped: list[dict[str, Any]] = []
    state_skipped: list[dict[str, Any]] = []
    for rank, attempts in sorted(by_rank.items(), key=lambda kv: str(kv[0])):
        order = sorted(attempts)
        run_start = min(attempts[order[0]])
        # Whether the rows read so far (every row of the earlier attempts) had the field.
        seen_data = seen_checkpoint = False
        first_scale: float | None = None
        for step in sorted(attempts[order[0]]):
            first_scale = _number(attempts[order[0]][step].get("grad_scale"))
            if first_scale is not None:
                break
        hashes: dict[int, tuple[str | None, int]] = {}  # step -> (batch hash, attempt)
        scales: dict[int, float] = {}
        lrs: set[tuple[float, ...]] = set()
        saves: list[tuple[int, Mapping[str, Any]]] = []  # (after_step, manifest), in the order written
        for attempt in order:
            steps = attempts[attempt]
            resumed_at = min(steps)
            if attempt != order[0] and resumed_at > run_start:
                first = steps[resumed_at]
                data = _data(first)
                has_data = seen_data or data is not None
                has_checkpoint = seen_checkpoint or isinstance(first.get("checkpoint"), Mapping)
                result = _compare_data(hashes, _hash(data) if data is not None else None, resumed_at)
                if isinstance(result, str):
                    if has_data:
                        data_skipped.append({"check": "resume_data", "step": resumed_at, "rank": rank, "reason": result})
                elif result is not None:
                    data_changes.append({"step": resumed_at, "rank": rank, "attempt": attempt, **result})
                earlier = [s for s in scales if s <= resumed_at]
                previous_scale = scales[max(earlier)] if earlier else None
                saved = max((s for s in saves if s[0] < resumed_at), key=lambda s: s[0], default=None)
                if saved is not None:
                    # The last one written among those after the same step.
                    saved = [s for s in saves if s[0] == saved[0]][-1]
                findings, source, checked = _state_findings(
                    first, resumed_at, run_start, saved, len(lrs) > 1, first_scale, previous_scale
                )
                if findings:
                    change: dict[str, Any] = {"step": resumed_at, "rank": rank, "attempt": attempt, "source": source,
                                              "findings": findings}
                    if source == "save":
                        change["saved_after_step"] = saved[0]  # type: ignore[index]
                    state_changes.append(change)
                elif not checked and has_checkpoint:
                    state_skipped.append({
                        "check": "resume_state", "step": resumed_at, "rank": rank,
                        "reason": "no checkpoint manifest, optimizer snapshot or loss scale recorded for this resume",
                    })
            # This attempt rolls back whatever earlier attempts recorded from its first step on.
            hashes = {k: v for k, v in hashes.items() if k < resumed_at}
            scales = {k: v for k, v in scales.items() if k < resumed_at}
            saves = [s for s in saves if s[0] < resumed_at]
            for step in sorted(steps):
                record = steps[step]
                data = _data(record)
                seen_data = seen_data or data is not None
                seen_checkpoint = seen_checkpoint or isinstance(record.get("checkpoint"), Mapping)
                if data is not None and (_int(data.get("batches")) or 0) > 0:
                    hashes[step] = (_hash(data), attempt)
                scale = _number(record.get("grad_scale"))
                if scale is not None:
                    scales[step] = scale
                lr = record.get("lr")
                if isinstance(lr, list) and lr and all(_number(v) is not None for v in lr):
                    lrs.add(tuple(float(v) for v in lr))
                checkpoint = record.get("checkpoint")
                save = checkpoint.get("save") if isinstance(checkpoint, Mapping) else None
                if isinstance(save, Mapping):
                    after = _int(save.get("after_step"))
                    saves.append((step if after is None else after, save))
    if not_checked is not None:
        not_checked.extend(data_skipped)
        not_checked.extend(state_skipped)

    def incident(kind: str, changes: list[dict[str, Any]], note: str) -> dict[str, Any] | None:
        if not changes:
            return None
        changes.sort(key=lambda c: c["step"])
        return {"kind": kind, "step": changes[0]["step"], "resumes": len(changes),
                "changes": changes[:CHANGES_LISTED], "note": note}

    return (
        incident("resume_data_mismatch", data_changes,
                 "an observation: a resume that starts its epoch over on purpose repeats data too"),
        incident("resume_incomplete", state_changes,
                 "an observation: a component left out or reset on purpose looks the same"),
    )
