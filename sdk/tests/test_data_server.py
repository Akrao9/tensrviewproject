"""The data and checkpoint kinds in `tensorquill-server` (`server/src/detect/data.rs`) give what
Python's `data_checks.py` gives, through the real binary: after every stretch of a run that
streams in pieces, across a server restart, over resumes, and across ranks in `watchdog.json`.

Needs the built server (`cargo build --manifest-path server/Cargo.toml`, or
`TENSORQUILL_SERVER_BIN`)."""

from __future__ import annotations

import hashlib
import json
import time

import pytest

pytest.importorskip("pyarrow.flight")

import tensorquill  # noqa: E402
from tensorquill.data_checks import detect_duplicate_batches  # noqa: E402
from tensorquill.evidence import detect_watchdog_incidents, resolve_attempts  # noqa: E402
from tensorquill.flight import send_file  # noqa: E402
from test_flight_e2e import BINARY, Server  # noqa: E402
from test_server_detectors import FAST, PORTED, Stream, differences, record  # noqa: E402

pytestmark = pytest.mark.skipif(BINARY is None, reason="tensorquill-server is not built")

DATA_KINDS = {"resume_data_mismatch", "resume_incomplete", "data_stalled", "epoch_order_repeated"}
CHECKS = ("resume_data", "resume_state")


def hx(tag: object) -> str:
    return hashlib.blake2b(str(tag).encode(), digest_size=8).hexdigest()


def reference(records):
    return [i for i in tensorquill.build_evidence(records)["incidents"] if i["kind"] in PORTED]


def not_checked(document_or_evidence):
    return [n for n in document_or_evidence.get("not_checked", []) if n["check"] in CHECKS]


ORDERS = ["a", "b", "b", "c", "d", "e", "f", "g", "h", "i", "j"]  # epoch 2 repeats epoch 1's order
FULL = {"model": {"tensors": 4}, "optimizer": {"entries": 2, "step": 25}, "scheduler": {"last_epoch": 25},
        "scaler": {"scale": 65536.0}, "rng": {}, "data": {}}


def tag(step: int) -> str:
    return f"{ORDERS[step // 10]}-{step % 10}"


def row(step, attempt, drawn, *, scale=None, **extra):
    data = {"batches": 0} if drawn is None else {
        "batches": 1, "batch_hash": hx(drawn), "epoch": step // 10, "sampler_epoch": 0, "shuffle": True}
    values = {"data": data, "scalars": {"train/loss": 2.0 - step / 1000 + attempt / 100},
              "lr": [1e-3 * min(1.0, (step + 1) / 20)], **extra}
    if scale is not None:
        values["grad_scale"] = scale
    return record(step, attempt, **values)


def attempts() -> list[tuple[str, list[dict]]]:
    """One rank over five attempts, with every data and checkpoint kind among things that must
    not confuse them."""
    first = []
    for step in range(60):
        r = row(step, 1, None if 30 <= step <= 32 else tag(step), scale=65536.0 if step < 40 else 32768.0)
        if step == 24:
            r["checkpoint"] = {"save": {"components": FULL, "after_step": 24}}
        if step == 49:
            r["checkpoint"] = {"save": {"components": {k: v for k, v in FULL.items() if k != "rng"}, "after_step": 49}}
        first.append(r)
    # Resumed at 50 from the save after 49 and replays the batches of five steps before, with a
    # partial checkpoint, a fresh optimizer and the scaler at its initial value.
    second = [row(s, 2, tag(s - 5), scale=65536.0) for s in range(50, 80)]
    second[0]["checkpoint"] = {
        "load": {"components": {"model": {}, "optimizer": {"entries": 2}, "data": {}}},
        "optimizer": {"entries_before": 0, "entries_after": 2, "step_after": 1},
    }
    second[19]["checkpoint"] = {"save": {"components": {**FULL, "scheduler": {"last_epoch": 70}}, "after_step": 69}}
    # Resumed at 70 from attempt 2's save after 69 (no load manifest): a good resume.
    third = [row(s, 3, tag(s - 5), scale=65536.0) for s in range(70, 90)]
    third[0]["checkpoint"] = {"optimizer": {"entries_before": 2, "entries_after": 2, "step_after": 71}}
    # Resumed at 95, past attempt 3's end: draws the batch attempt 3 drew at 85. It records
    # nothing about its state, so the last save (after 69) is assumed: its scheduler is far behind.
    fourth = [row(s, 4, tag(s - 15)) for s in range(95, 100)]
    # Resumed at 100 with batches never drawn: cannot be told from a correct resume.
    fifth = [row(s, 5, f"new-{s}") for s in range(100, 110)]
    return [
        ("before the repeat", first[:12]), ("the repeated epoch", first[12:23]), ("into the stall", first[23:31]),
        ("the rest of attempt 1", first[31:]), ("the first row of a bad resume", second[:1]),
        ("the bad resume goes on", second[1:]), ("a good resume", third), ("a resume past the end", fourth),
        ("a resume with new batches", fifth),
    ]


def test_the_fixture_has_every_data_kind_where_it_was_planted():
    records = [r for _, stretch in attempts() for r in stretch]
    evidence = tensorquill.build_evidence(records)
    by_kind = {i["kind"]: i for i in evidence["incidents"]}
    assert set(by_kind) == DATA_KINDS
    assert (by_kind["epoch_order_repeated"]["step"], by_kind["epoch_order_repeated"]["through_step"]) == (20, 22)
    assert (by_kind["data_stalled"]["step"], by_kind["data_stalled"]["reason"]) == (30, "no_batch")
    data = by_kind["resume_data_mismatch"]
    assert [(c["step"], c["basis"], c["finding"], c["matched_step"]) for c in data["changes"]] == [
        (50, "same_step", "repeated", 45), (95, "lookup", "repeated", 85)]
    state = by_kind["resume_incomplete"]
    assert [(c["step"], c["source"]) for c in state["changes"]] == [(50, "load"), (95, "save"), (100, "save")]
    assert state["changes"][1]["findings"] == [{"component": "scheduler", "finding": "position", "last_epoch": 70}]
    assert [(f["component"], f["finding"]) for f in state["changes"][0]["findings"]] == [
        ("scheduler", "missing"), ("scaler", "missing"), ("rng", "missing"), ("optimizer", "empty"),
        ("scaler", "initial_scale")]
    assert all("scalars" in i for i in by_kind.values())
    assert [(n["check"], n["step"]) for n in not_checked(evidence)] == [("resume_data", 100)]


def test_the_server_reports_what_python_reports_after_every_stretch_and_a_restart(tmp_path):
    stream = Stream(tmp_path, *FAST)
    rebuilds = []
    for name, stretch in attempts():
        if name == "a good resume":
            stream.restart()
        document = stream.send(stretch)
        assert differences(reference(stream.sent), document["incidents"]) == [], name
        python = not_checked(tensorquill.build_evidence(stream.sent))
        assert differences(python, not_checked(document)) == [], name
        rebuilds.append(document["rebuilds"])
    assert rebuilds == [1, 1, 1, 1, 2, 2, 1, 2, 3]  # each new attempt and the restart rebuild
    assert {"resume_data_mismatch", "data_stalled"} <= set(document["checked"])
    assert document["thresholds"]["data_stall_steps"] == 3 and document["thresholds"]["duplicate_window"] == 4096
    # The server's own Parquet gives Python the same answer.
    from_store = tensorquill.build_evidence(stream.store / "runs" / stream.run, rank=0)
    assert differences([i for i in from_store["incidents"] if i["kind"] in PORTED], document["incidents"]) == []
    assert stream.server.terminate() == 0


def edges() -> dict[str, list[dict]]:
    """Short runs for the rules' edges, each judged the same by both."""
    def run(name, tags, **fields):
        out = []
        for step, (epoch, drawn) in enumerate(tags):
            data = {"batches": 0} if drawn is None else {"batches": 1, "batch_hash": hx(drawn), "epoch": epoch, **fields}
            out.append(record(step, run=name, data=data))
        return out

    def passes(orders, length, **fields):
        return [(e, f"{o}-{p}") for e, o in enumerate(orders) for p in range(length)]

    return {
        "same-batch": run("same-batch", [(0, 1), (0, 2), (0, 2), (0, 2), (0, 2), (0, 3)]),
        "two-only": run("two-only", [(0, 1), (0, None), (0, None), (0, 2), (0, 3)]),
        "kept-order": run("kept-order", passes("aa", 4), shuffle=False),
        "short-epochs": run("short-epochs", passes("ab", 2) + [(2, "b-0"), (2, "b-1")], shuffle=True),
        "one-batch-epochs": run("one-batch-epochs", passes("aaa", 1), shuffle=True),
        "starts-alike": run("starts-alike", passes("a", 5) + [(1, "a-0"), (1, "a-1"), (1, "z-2")], shuffle=True),
        "unknown-shuffle": run("unknown-shuffle", passes("cc", 3)),
        # Epochs 0 and 2 in one order, with no epoch 1 recorded: not consecutive, not compared.
        "skipped-epoch": run("skipped-epoch", [(0, "a-0"), (0, "a-1"), (2, "a-0"), (2, "a-1")], shuffle=True),
    }


def test_edges_match(tmp_path):
    runs = edges()
    found = {name: reference(records) for name, records in runs.items()}
    assert {name: [(i["kind"], i["step"]) for i in incidents] for name, incidents in found.items()} == {
        "same-batch": [("data_stalled", 2)], "two-only": [], "kept-order": [], "short-epochs": [("epoch_order_repeated", 4)],
        "one-batch-epochs": [], "starts-alike": [], "unknown-shuffle": [("epoch_order_repeated", 3)],
        "skipped-epoch": [],
    }
    server = Server(tmp_path / "store", 0, *FAST)
    for name, records in runs.items():
        document = Stream(tmp_path, run=name, server=server).send(records)
        assert differences(found[name], document["incidents"]) == [], name
    assert server.terminate() == 0


def test_ranks_drawing_the_same_batch_show_in_the_runs_status(tmp_path):
    run_id = "dup"
    # Rank 1 draws rank 0's batches at steps 10-19 (an IterableDataset not split by rank);
    # rank 2 is rank 0's tensor-parallel peer (data rank 0) and always matches it. Rank 1's
    # second attempt resumes at 15 with its own batches: the resolved rows no longer duplicate there.
    def ranked(rank, attempt, steps, drawn, share=None):
        return [{"schema_version": 1, "run_id": run_id, "rank": rank, "attempt": attempt, "step": s, "ts": float(s),
                 "data": {"batches": 1, "batch_hash": hx(drawn(s)), **({} if share is None else {"data_rank": share})}}
                for s in steps]

    ranks = {
        0: ranked(0, 1, range(40), lambda s: f"r0-{s}", share=0),
        1: ranked(1, 1, range(30), lambda s: f"r0-{s}" if 10 <= s < 20 else f"r1-{s}")
           + ranked(1, 2, range(15, 40), lambda s: f"r1-{s}"),
        2: ranked(2, 1, range(40), lambda s: f"r0-{s}", share=0),
    }
    store = tmp_path / "store"
    server = Server(store, 0, *FAST, "--watchdog-seconds", "0.2")
    for rank, records in ranks.items():
        path = tmp_path / f"{run_id}.rank{rank}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        assert send_file(path, server.endpoint, run_id, rank, timeout=30.0)
    expected = detect_duplicate_batches(resolve_attempts([r for records in ranks.values() for r in records]))
    assert expected == {"kind": "duplicate_batches", "step": 10, "ranks": [0, 1, 2], "duplicated_steps": 5,
                        "ranks_compared": [0, 1, 2]}
    deadline = time.monotonic() + 15
    while True:
        try:
            status = json.loads((store / "runs" / run_id / "watchdog.json").read_text())
            if status.get("duplicate_batches", {}).get("ranks_compared") == [0, 1, 2]:
                break
        except (FileNotFoundError, ValueError):
            pass
        assert time.monotonic() < deadline, "the status never compared every rank"
        time.sleep(0.1)
    [python] = [i for i in detect_watchdog_incidents(status) if i["kind"] == "duplicate_batches"]
    assert differences(expected, python) == []
    assert [i for i in status["incidents"] if i["kind"] == "duplicate_batches"] == [python]
    # A rank's own evidence reads the run's status too.
    evidence = tensorquill.build_evidence(store / "runs" / run_id, rank=1)
    assert [i["step"] for i in evidence["incidents"] if i["kind"] == "duplicate_batches"] == [10]
    assert server.terminate() == 0


def test_a_resume_is_judged_against_the_history_after_every_rollback(tmp_path):
    stream = Stream(tmp_path, *FAST)
    stream.send([row(s, 1, tag(s)) for s in range(30)])
    stream.send([row(s, 2, f"b-{s}") for s in range(10, 15)])
    document = stream.send([row(s, 3, tag(s)) for s in range(20, 25)])
    python = not_checked(tensorquill.build_evidence(stream.sent))
    assert [(n["check"], n["step"]) for n in python] == [("resume_data", 20)]
    assert differences(python, not_checked(document)) == []
    assert differences(reference(stream.sent), document["incidents"]) == []
    assert stream.server.terminate() == 0


def test_a_resume_with_nothing_to_judge_is_listed_the_same_way(tmp_path):
    first = [row(s, 1, tag(s)) for s in range(10)]
    first[3]["checkpoint"] = {"save": {"components": FULL, "after_step": 30}}  # one the resume cannot have loaded
    second = [row(s, 2, tag(s)) for s in range(5, 12)]
    stream = Stream(tmp_path, *FAST)
    stream.send(first)
    document = stream.send(second)
    python = not_checked(tensorquill.build_evidence(stream.sent))
    assert [(n["check"], n["step"]) for n in python] == [("resume_state", 5)]
    assert differences(python, not_checked(document)) == []
    assert differences(reference(stream.sent), document["incidents"]) == []
    assert stream.server.terminate() == 0
