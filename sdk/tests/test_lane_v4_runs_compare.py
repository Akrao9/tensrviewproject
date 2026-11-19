"""Lane V4 (wave V): the run listing and `compare`, one test or more for each row it fixes.

- E2 / CLI-07: `compare`'s last step is the run as it stands after a resume, not the highest
  step of every attempt (`resolve_attempts`, as the detectors read the records).
- E4: an observation in a file of several ranks does not make the run `unhealthy`.
- E7 / CLI-12: the listing reads the compact evidence file `--compact` writes.
- CLI-08: one rank whose stored parts cannot be read does not end part 4 for every rank, and a
  rank whose parts hold no record is not judged.
- CLI-09: a store run with no judged rank lists its incidents as unknown, not as none.
- CLI-13: two files of one run and rank are both kept, read and named.
- CLI-20: a constant series does not rank first among series leaving at the same step.
- CLI-21: a run kept as local files and in a store under one folder is listed once.

The records are built here, so nothing depends on another lane's test file.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

import tensorquill
from tensorquill import compare as cmp
from tensorquill import divergence as dv
from tensorquill import evidence as ev
from tensorquill import runs as runs_module

SPIKED = "blocks.1.weight"


def records(steps: int | range, *, run: str, rank: int = 0, attempt: int = 1, spike: int | None = None,
            drift_from: int | None = None) -> list[dict]:
    """Step records of a small dense model. `spike` multiplies one parameter's gradient norm by
    60 at that step; `drift_from` moves the loss away from step on, for part 4."""
    out = []
    for step in (range(steps) if isinstance(steps, int) else steps):
        wobble = 1.0 + 0.05 * math.sin(step * 0.7)
        grads = {f"blocks.{block}.weight": 0.2 * wobble * (block + 1) for block in range(3)}
        if spike is not None and step == spike:
            grads[SPIKED] *= 60.0
        loss = 2.0 * wobble + (5.0 + step if drift_from is not None and step >= drift_from else 0.0)
        out.append({
            "schema_version": 1, "run_id": run, "rank": rank, "attempt": attempt, "step": step,
            "ts": 1.7e9 + attempt * 1e5 + step,
            "layers": {f"blocks.{block}": {"mean": wobble, "std": 2 * wobble} for block in range(3)},
            "grad_norm": math.sqrt(sum(v * v for v in grads.values())), "grad_norms": grads,
            "lr": [1e-3], "scalars": {"loss": loss},
        })
    return out


def write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def manifest(folder: Path, run_id: str, **fields) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run_id}.run.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run_id, "project": "p", "name": run_id, "created_at": "2026-11-17T10:00:00Z",
        **fields}))


def resumed(run: str, rank: int = 0) -> list[dict]:
    """Attempt 1 runs to step 299 with a spike at 250; attempt 2 resumes from the checkpoint at
    100 and stops at 199. The run as it stands ends at 199 and never reached the spike."""
    return records(300, run=run, rank=rank, attempt=1, spike=250) + records(range(100, 200), run=run, rank=rank, attempt=2)


# --- E2 / CLI-07 ------------------------------------------------------------------------------


def test_e2_a_resumed_local_run_ends_where_its_resolved_records_end_and_a_later_spike_is_ended_before(tmp_path):
    a = write(tmp_path / "a" / "runa.jsonl", resumed("runa"))
    b = write(tmp_path / "b" / "runb.jsonl", records(300, run="runb", spike=250))
    opened_a, opened_b = cmp.open_run(a), cmp.open_run(b)
    assert max(r["step"] for r in ev.load_records(a)) == 199
    assert opened_a["last_step"] == 199
    assert opened_b["last_step"] == 299
    part = cmp.compare_incidents([opened_a, opened_b], ["A", "B"])
    spike = next(e for e in part["entries"] if e["kind"] == "grad_spike")
    assert spike["in"] == ["B"] and spike["absent_from"] == {"A": "ended_before_step"}


def test_e2_a_resumed_rank_of_a_local_run_of_several_ranks_ends_where_its_resolved_records_end(tmp_path):
    folder = tmp_path / "out"
    write(folder / "two.rank0.jsonl", resumed("two", 0))
    write(folder / "two.rank1.jsonl", records(200, run="two", rank=1))
    manifest(folder, "two", world_size=2)
    opened = cmp.open_run("two", folder)
    assert opened["read_from"]["all_ranks"] and opened["last_step"] == 199


def store_rank(root: Path, run_id: str, ranks: tuple[int, ...]) -> Path:
    """A run folder of a store whose ranks have no report: their parts are read and judged."""
    folder = root / "runs" / run_id
    for rank in ranks:
        parts = folder / f"rank={rank}" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-0-299.parquet").write_bytes(b"read through load_records, which the test stands in for")
    (folder / "watchdog.json").write_text(json.dumps({
        "schema_version": 1, "summary": "finished", "world_size": len(ranks),
        "ranks": [{"rank": rank, "state": "finished", "last_step": 199} for rank in ranks]}))
    (folder / "run.json").write_text(json.dumps({"run_id": run_id, "project": "p", "name": run_id}))
    return folder


def fake_parts(monkeypatch, by_folder_rank: dict[tuple[str, int], list[dict] | Exception]) -> None:
    """`load_records` on a store's run folder gives these records (every attempt), or raises."""
    real = ev.load_records

    def load(source, *, rank=None, all_attempts=False):
        path = Path(source) if isinstance(source, (str, Path)) else None
        if path is None or not path.is_dir():
            return real(source, rank=rank, all_attempts=all_attempts)
        found = by_folder_rank[(path.name, rank)]
        if isinstance(found, Exception):
            raise found
        return list(found) if all_attempts else ev.resolve_attempts(list(found))

    monkeypatch.setattr(ev, "load_records", load)


def test_e2_a_resumed_rank_read_from_a_stores_parts_ends_where_its_resolved_records_end(tmp_path, monkeypatch):
    one = store_rank(tmp_path / "s1", "solo", (0,))
    two = store_rank(tmp_path / "s2", "pair", (0, 1))
    fake_parts(monkeypatch, {("solo", 0): resumed("solo"), ("pair", 0): resumed("pair", 0),
                             ("pair", 1): records(200, run="pair", rank=1)})
    alone = cmp.open_run(one)  # one rank: `_open_store`
    assert "Python detectors" in alone["read_from"]["judged_by"] and alone["last_step"] == 199
    whole = cmp.open_run(two)  # every rank, judged here: `_open_store_all_ranks`
    assert whole["read_from"]["judged_here"] == [0, 1] and whole["last_step"] == 199


# --- E4 ---------------------------------------------------------------------------------------


def evidence_of(rows: list[dict], *, add: list[dict] = ()) -> dict:
    found = ev.build_evidence(rows)
    found["incidents"] = list(found["incidents"]) + [dict(i) for i in add]
    return found


OBSERVATION = {"kind": "memory_pressure", "step": 10, "class": "observation", "device": "cuda:0", "reserved_fraction": 0.97}
INVARIANT = {"kind": "non_finite", "step": 12, "class": "invariant", "layers": ["blocks.2"], "gradients": []}


def mixed_run(folder: Path, *, shared_adds: list[dict], rank2_spike: bool = False) -> None:
    """A shared file holding ranks 0 and 1 (with one rank's evidence file beside it) and rank 2's
    own file, judged clean unless `rank2_spike`."""
    zero, one = records(60, run="mixed", rank=0), records(60, run="mixed", rank=1)
    shared = write(folder / "shared.jsonl", [r for pair in zip(zero, one) for r in pair])
    ev.write_evidence(evidence_of(zero, add=shared_adds), shared.with_suffix(".evidence.json"))
    own = write(folder / "mixed.rank2.jsonl", records(60, run="mixed", rank=2, spike=40 if rank2_spike else None))
    ev.write_evidence(ev.build_evidence(own), own.with_suffix(".evidence.json"))
    manifest(folder, "mixed", world_size=3)


def test_e4_an_observation_in_a_file_of_several_ranks_does_not_make_the_run_unhealthy(tmp_path):
    assert ev.incident_class(OBSERVATION["kind"]) == "observation"
    mixed_run(tmp_path / "out", shared_adds=[OBSERVATION])
    (row,) = tensorquill.list_runs(tmp_path / "out")
    assert row["incidents"] == {"observation": 1}
    # Ranks 0 and 1 are not judged, so the run is unknown; an observation never makes it unhealthy.
    assert row["run_state"] == "unknown"


def test_e4_an_invariant_in_a_file_of_several_ranks_still_makes_the_run_unhealthy(tmp_path):
    assert ev.incident_class(INVARIANT["kind"]) == "invariant"
    mixed_run(tmp_path / "out", shared_adds=[INVARIANT])
    (row,) = tensorquill.list_runs(tmp_path / "out")
    assert row["run_state"] == "unhealthy" and row["incidents"].get("invariant") == 1


def test_e4_a_rank_already_judged_unhealthy_stays_so_beside_a_shared_observation(tmp_path):
    mixed_run(tmp_path / "out", shared_adds=[OBSERVATION], rank2_spike=True)
    (row,) = tensorquill.list_runs(tmp_path / "out")
    assert row["run_state"] == "unhealthy"
    assert row["incidents"] == {"anomaly": 1, "observation": 1}


# --- E7 / CLI-12 ------------------------------------------------------------------------------


def compact_beside(records_file: Path, evidence: dict) -> Path:
    from tensorquill import compact

    return compact.write_compact(evidence, records_file.with_suffix(".evidence.json"))


def test_e7_a_run_with_only_the_compact_evidence_file_is_listed_from_it(tmp_path):
    folder = tmp_path / "out"
    path = write(folder / "c.rank0.jsonl", records(60, run="c", spike=40))
    written = compact_beside(path, ev.build_evidence(path))
    assert written.name == "c.rank0.evidence.compact.json" and not path.with_suffix(".evidence.json").exists()
    (row,) = tensorquill.list_runs(folder)
    assert row["evidence"]["state"] == "current" and row["evidence"].get("compact") is True
    assert row["evidence"]["path"] == str(written)
    assert row["first_incident"] == {"kind": "grad_spike", "step": 40} and row["run_state"] == "unhealthy"


def test_e7_each_rank_of_a_run_of_several_ranks_is_read_from_its_compact_evidence_file(tmp_path):
    folder = tmp_path / "out"
    for rank in (0, 1):
        path = write(folder / f"c.rank{rank}.jsonl", records(60, run="c", rank=rank, spike=40 if rank else None))
        compact_beside(path, ev.build_evidence(path))
    manifest(folder, "c", world_size=2)
    (row,) = tensorquill.list_runs(folder)
    assert row["evidence"]["judged_ranks"] == 2 and row["evidence"]["unknown_ranks"] == 0
    assert row["run_state"] == "unhealthy" and row["ranks_with_incidents"] == [1]


def test_e7_when_both_evidence_files_are_beside_the_records_the_newer_is_read_and_the_full_one_on_a_tie(tmp_path):
    folder = tmp_path / "out"
    path = write(folder / "c.rank0.jsonl", records(60, run="c", spike=40))
    full = path.with_suffix(".evidence.json")
    ev.write_evidence(ev.build_evidence(path), full)  # shows the spike
    small = compact_beside(path, ev.build_evidence(records(60, run="c")))  # clean
    base = path.stat().st_mtime
    os.utime(full, (base, base + 10))
    os.utime(small, (base, base + 20))
    (row,) = tensorquill.list_runs(folder)
    assert row["evidence"]["path"] == str(small) and row["incidents"] == {}
    os.utime(small, (base, base + 5))
    (row,) = tensorquill.list_runs(folder)
    assert row["evidence"]["path"] == str(full) and row["incidents"] == {"anomaly": 1}
    os.utime(small, (base, base + 10))
    (row,) = tensorquill.list_runs(folder)
    assert row["evidence"]["path"] == str(full)


# --- CLI-08 -----------------------------------------------------------------------------------


def test_cli08_one_rank_whose_parts_cannot_be_read_leaves_the_other_ranks_compared_in_part_4(tmp_path, monkeypatch):
    root = tmp_path / "store"
    store_rank(root, "base", (0, 1))
    store_rank(root, "after", (0, 1))
    fake_parts(monkeypatch, {
        ("base", 0): records(120, run="base"), ("base", 1): records(120, run="base", rank=1),
        ("after", 0): records(120, run="after", drift_from=70), ("after", 1): ValueError("a torn part"),
    })
    result = tensorquill.compare_runs("base", "after", source=root)
    read = result["runs"][1]["read_from"]
    assert read["unknown_ranks"] == [1] and "a torn part" in read["unknown_why"][1]
    part = result["divergence"]
    assert part["state"] == "diverged" and part["ranks_compared"] == [0]
    assert part["divergence"]["step"] == 70
    assert "a torn part" in part["ranks_not_read"][1] and "rank 1" in part["detail"]


def test_cli08_a_rank_that_fails_only_when_part_4_reads_it_does_not_end_part_4_for_the_others(tmp_path, monkeypatch):
    root = tmp_path / "store"
    store_rank(root, "base", (0, 1))
    store_rank(root, "after", (0, 1))
    fake_parts(monkeypatch, {
        ("base", 0): records(120, run="base"), ("base", 1): records(120, run="base", rank=1),
        ("after", 0): records(120, run="after", drift_from=70), ("after", 1): records(120, run="after", rank=1),
    })
    real = ev.load_records

    def flaky(source, *, rank=None, all_attempts=False):
        if not all_attempts and rank == 1 and Path(source).name == "after":
            raise OSError("the part went away")
        return real(source, rank=rank, all_attempts=all_attempts)

    monkeypatch.setattr(ev, "load_records", flaky)
    part = tensorquill.compare_runs("base", "after", source=root)["divergence"]
    assert part["state"] == "diverged" and part["ranks_compared"] == [0]
    assert "the part went away" in part["ranks_not_read"][1] and "rank 1" in part["detail"]


def test_cli08_a_rank_whose_parts_hold_no_record_is_unknown_not_judged(tmp_path, monkeypatch):
    root = tmp_path / "store"
    store_rank(root, "base", (0, 1))
    fake_parts(monkeypatch, {("base", 0): records(60, run="base"), ("base", 1): []})
    opened = cmp.open_run("base", root)
    read = opened["read_from"]
    assert read["judged_here"] == [0] and read["unknown_ranks"] == [1]
    assert "no record" in read["unknown_why"][1]
    assert 1 not in opened["records"]["ranks"]


def test_cli08_one_rank_alone_whose_parts_hold_no_record_is_unknown(tmp_path, monkeypatch):
    root = tmp_path / "store"
    store_rank(root, "solo", (0,))
    fake_parts(monkeypatch, {("solo", 0): []})
    opened = cmp.open_run("solo", root)
    assert opened["read_from"]["unknown_ranks"] == [0] and "no record" in opened["read_from"]["unknown_why"][0]


# --- CLI-09 -----------------------------------------------------------------------------------


def test_cli09_a_store_run_with_no_judged_rank_lists_its_incidents_as_unknown(tmp_path):
    root = tmp_path / "store"
    store_rank(root, "solo", (0,))  # no report: nothing judged
    store_rank(root, "pair", (0, 1))
    rows = {row["run_id"]: row for row in tensorquill.list_runs(root)}
    for run_id in ("solo", "pair"):
        row = rows[run_id]
        assert row["evidence"]["judged_ranks"] == 0
        assert row["incidents"] is None and row["first_incident"] is None
    text = runs_module.listing_text(list(rows.values()), root)
    assert " none " not in text and "?" in text


def test_cli09_a_store_run_with_no_judged_rank_still_lists_the_status_incidents_and_says_ranks_are_not_judged(tmp_path):
    root = tmp_path / "store"
    folder = store_rank(root, "pair", (0, 1))
    status = json.loads((folder / "watchdog.json").read_text())
    status["duplicate_batches"] = {"step": 5, "ranks": [0, 1], "pairs": 1}
    (folder / "watchdog.json").write_text(json.dumps(status))
    (row,) = tensorquill.list_runs(root)
    assert row["evidence"]["judged_ranks"] == 0
    assert row["incidents"] and sum(row["incidents"].values()) == 1
    assert row.get("rank_incidents_not_known") is True
    assert "not judged" in runs_module._incidents_text(row)


# --- CLI-13 -----------------------------------------------------------------------------------


def test_cli13_two_files_of_one_run_and_rank_are_both_kept_and_the_listing_names_them(tmp_path):
    folder = tmp_path / "custom"
    first = write(folder / "first-try.jsonl", records(3, run="cli13", attempt=1))
    second = write(folder / "second-try.jsonl", records(5, run="cli13", attempt=2))
    (found,) = runs_module.discover_runs(folder)
    kept = [found["files"][0], *found.get("more_files", {}).get(0, [])]
    assert sorted(kept) == [first, second]
    (row,) = tensorquill.list_runs(folder)
    # The later attempt replaced the earlier one from step 0 (`resolve_attempts` on the files' ends).
    assert row["last_step"] == 4
    said = json.dumps(row)
    assert "first-try.jsonl" in said and "second-try.jsonl" in said


def test_cli13_compare_reads_every_file_of_a_rank_and_resolves_the_attempts_across_them(tmp_path):
    folder = tmp_path / "custom"
    write(folder / "a-first.jsonl", records(300, run="resumed", attempt=1, spike=250))
    write(folder / "b-second.jsonl", records(range(100, 200), run="resumed", attempt=2))
    other = write(tmp_path / "other" / "clean.jsonl", records(300, run="clean"))
    opened = cmp.open_run("resumed", folder)
    assert opened["last_step"] == 199
    # The spike at 250 was rolled back: it is not the run's.
    assert not [i for i in opened["evidence"]["incidents"] if i.get("kind") == "grad_spike"]
    result = tensorquill.compare_runs("resumed", str(other), source=folder)
    assert result["divergence"]["runs"]["a"]["last_step"] == 199


def test_cli13_the_files_are_read_the_same_whichever_sorts_first(tmp_path):
    seen = []
    for first, second in (("a-first.jsonl", "b-second.jsonl"), ("b-first.jsonl", "a-second.jsonl")):
        folder = tmp_path / first
        write(folder / first, records(300, run="resumed", attempt=1, spike=250))
        write(folder / second, records(range(100, 200), run="resumed", attempt=2))
        opened = cmp.open_run("resumed", folder)
        seen.append((opened["last_step"], sorted(i["kind"] for i in opened["evidence"]["incidents"])))
    assert seen[0] == seen[1] == (199, [])


def test_cli13_a_copy_of_a_rank_file_is_read_once_not_as_a_repeat(tmp_path):
    folder = tmp_path / "out"
    path = write(folder / "x.rank0.jsonl", records(60, run="x"))
    (folder / "x.bak.jsonl").write_bytes(path.read_bytes())
    (found,) = runs_module.discover_runs(folder)
    assert found.get("more_files")  # both files are kept
    opened = cmp.open_run("x", folder)
    assert opened["last_step"] == 59
    assert not [i for i in opened["evidence"]["incidents"] if i.get("kind") == "repeated_steps"]
    part = tensorquill.compare_runs("x", str(write(tmp_path / "o" / "y.jsonl", records(60, run="y"))), source=folder)
    assert part["divergence"]["runs"]["a"]["records"] == 60
    # A step written twice within one file is still a repeat.
    twice = write(tmp_path / "t" / "t.rank0.jsonl", records(60, run="t") + records(range(30, 31), run="t"))
    assert [i["kind"] for i in cmp.open_run(twice)["evidence"]["incidents"]] == ["repeated_steps"]


def test_cli13_a_copy_of_one_ranks_file_in_a_run_of_several_ranks_is_read_once(tmp_path):
    folder = tmp_path / "out"
    write(folder / "two.rank0.jsonl", records(60, run="two"))
    one = write(folder / "two.rank1.jsonl", records(60, run="two", rank=1))
    (folder / "two-copy.jsonl").write_bytes(one.read_bytes())
    manifest(folder, "two", world_size=2)
    opened = cmp.open_run("two", folder)
    assert opened["read_from"]["judged_ranks"] == [0, 1]
    assert not [g for g in opened["evidence"]["incident_groups"] if g["incident"].get("kind") == "repeated_steps"]


# --- CLI-20 -----------------------------------------------------------------------------------


def scalar_run(run: str, *, part: bool) -> list[dict]:
    """A moving loss and a constant scalar. With `part`, from step 60 the loss jumps far out of
    agreement and the constant scalar moves by a hair."""
    out = []
    for step in range(120):
        loss = 2.0 + 0.1 * math.sin(step * 0.7)
        steady = 1.0
        if part and step >= 60:
            loss += 50.0
            steady += 1e-9
        out.append({"run_id": run, "rank": 0, "step": step, "scalars": {"loss": loss, "steady": steady}})
    return out


def test_cli20_a_hair_on_a_constant_series_does_not_rank_before_a_large_departure_at_the_same_step():
    found = dv.first_divergence(scalar_run("a", part=False), scalar_run("b", part=True))
    assert found["answer"] == "diverged"
    names = [(s["series"], s["step"]) for s in found["diverged_series"]]
    assert set(names) == {("scalars/loss", 60), ("scalars/steady", 60)}
    assert found["divergence"]["name"] == "loss"
    assert names[0] == ("scalars/loss", 60)


def test_cli20_a_non_finite_departure_still_ranks_first_at_its_step():
    a = scalar_run("a", part=False)
    b = scalar_run("b", part=True)
    for row in b[60:]:
        row["scalars"]["other"] = math.inf
    for row in a + b[:60]:
        row["scalars"]["other"] = 3.0 + 0.1 * math.cos(row["step"])
    found = dv.first_divergence(a, b)
    assert found["divergence"]["name"] == "other"


# --- CLI-21 -----------------------------------------------------------------------------------


def test_cli21_a_run_kept_as_local_files_and_in_a_store_under_one_folder_is_listed_once(tmp_path):
    root = tmp_path / "both"
    write(root / "cli21.rank0.jsonl", records(60, run="cli21"))
    manifest(root, "cli21", world_size=1, name="twice")
    store_rank(root, "cli21", (0,))
    found = runs_module.discover_runs(root)
    assert [(r["run_id"], r["source"]) for r in found] == [("cli21", "local")]
    assert found[0]["also_in_store"] == root / "runs" / "cli21"
    (row,) = tensorquill.list_runs(root)
    assert row["source"] == "local" and row["also_in_store"] == str(root / "runs" / "cli21")
    assert "also in the store" in runs_module.listing_text([row], root)
    # By ID and by name it is the one run, read from its local files.
    assert runs_module.find_run(root, "cli21")["source"] == "local"
    assert runs_module.find_run(root, "twice")["source"] == "local"
    # Given as the folder, it is one run, not two.
    assert cmp.open_run(root)["run_id"] == "cli21"


def test_cli21_a_store_run_whose_local_side_is_only_a_manifest_is_listed_once_from_the_store(tmp_path):
    root = tmp_path / "both"
    manifest(root, "cli21", world_size=1)
    store_rank(root, "cli21", (0,))
    found = runs_module.discover_runs(root)
    assert [(r["run_id"], r["source"]) for r in found] == [("cli21", "store")]
