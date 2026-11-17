"""Listing runs and comparing them (plan: "Projects, run names and comparing runs", piece 2).

- `python -m tensorquill runs` and `tensorquill.list_runs`: one row a run, by project, from
  local files and from a server's store, read without a part and without a detector;
- `python -m tensorquill compare` and `tensorquill.compare_runs`: the setup, what was checked
  and the incidents, in that order; a name that fits several runs is refused with their IDs;
- the two verbs reach both entry points, and a path of the same name is written `./runs`.

The seeded runs are the toy Llama of `examples/seeded_failures.py` on CPU (about a minute;
skipped without `transformers`). The real-server test needs the binary.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tensorquill
from tensorquill import compare as cmp
from tensorquill import evidence as ev
from tensorquill import runs as runs_module
from test_mcp_server import NOT_FINITE, SPIKE, child_env, dense, report, tree, write_capture

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SEEDED = "model.layers.2.self_attn.k_proj.weight"  # the middle block of the 4-block toy


def local_run(folder: Path, run_id: str, records: list[dict] | None = None, *, rank: int = 0, evidence: bool = False,
              **manifest) -> Path:
    """A run as `tensorquill.init` leaves it in `folder`: the rank's `.jsonl` file, the manifest
    when any of its fields is given, and the evidence file when asked for."""
    folder.mkdir(parents=True, exist_ok=True)
    records = dense(60, run=run_id, rank=rank) if records is None else records
    path = write_capture(folder / f"{run_id}.rank{rank}.jsonl", records)
    if manifest:
        (folder / f"{run_id}.run.json").write_text(json.dumps({
            "schema_version": 1, "run_id": run_id, "group": None, "tags": [], "world_size": 1,
            "sdk_version": "0.0.1", "torch_version": "2.15.0", "run_id_source": "explicit", "config": {},
            "created_at": "2026-11-17T10:00:00Z", **manifest,
        }))
    if evidence:
        ev.write_evidence(ev.build_evidence(path), path.with_suffix(".evidence.json"))
    return path


def spiked(run_id: str, steps: int = 60, **more) -> list[dict]:
    return dense(steps, run=run_id, spike=(40, "blocks.1.weight", 60.0), **more)


def cli(*argv: str, cwd: Path, entry: str = "module") -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "tensorquill"] if entry == "module" else [
        sys.executable, "-c", "from tensorquill.evidence import main; main()"]  # what the console script calls
    return subprocess.run([*command, *argv], cwd=cwd, env=child_env(), capture_output=True, text=True, timeout=120)


# --- Listing --------------------------------------------------------------------------------


@pytest.fixture
def out(tmp_path) -> Path:
    """Two runs of one project with different names, one with a spike, and a run recorded
    before the manifest existed."""
    folder = tmp_path / "tensorquill-out"
    local_run(folder, "run-1", project="granite", name="baseline", evidence=True, created_at="2026-11-17T10:00:00Z")
    local_run(folder, "run-2", spiked("run-2"), project="granite", name="after-the-fix", evidence=True,
              created_at="2026-11-17T11:00:00Z", config={"lr": 0.01})
    local_run(folder, "old-run", evidence=True)
    return folder


def test_two_runs_of_one_project_are_listed_under_it_by_name_and_an_older_run_under_uncategorized(out):
    rows = tensorquill.list_runs(out)
    assert [(r["project"], r["name"], r["run_id"]) for r in rows] == [
        ("granite", "baseline", "run-1"), ("granite", "after-the-fix", "run-2"), ("uncategorized", "old-run", "old-run")]
    baseline, fixed, old = rows
    assert (baseline["run_state"], baseline["last_step"], baseline["incidents"], baseline["first_incident"]) == (
        "healthy", 59, {}, None)
    assert (fixed["run_state"], fixed["incidents"], fixed["first_incident"]) == (
        "unhealthy", {"anomaly": 1}, {"kind": "grad_spike", "step": 40})
    assert fixed["kinds_checked"]["of"] == 32 and 0 < fixed["kinds_checked"]["checked"] < 32
    assert (baseline["started"], baseline["has_manifest"], baseline["source"]) == ("2026-11-17T10:00:00Z", True, "local")
    assert baseline["evidence"]["state"] == "current"
    # No manifest: its start is its first record's time.
    assert (old["has_manifest"], old["started"]) == (False, "2023-11-14T22:13:20Z")
    assert [r["run_id"] for r in tensorquill.list_runs(out, project="granite")] == ["run-1", "run-2"]
    assert tensorquill.list_runs(out, project="nobody") == []


def test_the_listing_runs_no_detector_and_says_when_it_has_no_evidence_or_an_old_one(tmp_path, monkeypatch):
    folder = tmp_path / "out"
    path = local_run(folder, "run-1", spiked("run-1"), project="granite", name="baseline")
    monkeypatch.setattr(ev, "build_evidence", lambda *a, **k: pytest.fail("the listing ran a detector"))
    monkeypatch.setattr(ev, "load_records", lambda *a, **k: pytest.fail("the listing read every record"))
    (row,) = tensorquill.list_runs(folder)
    assert (row["run_state"], row["incidents"], row["first_incident"], row["kinds_checked"]) == (None, None, None, None)
    assert row["last_step"] == 59 and row["evidence"]["state"] == "missing"
    assert f"python -m tensorquill {path}" in row["evidence"]["detail"]
    text = runs_module.listing_text([row], folder)
    assert "?" in text and "no evidence file beside run-1.rank0.jsonl" in text
    monkeypatch.undo()

    ev.write_evidence(ev.build_evidence(path), path.with_suffix(".evidence.json"))
    assert tensorquill.list_runs(folder)[0]["evidence"]["state"] == "current"
    # The run goes on after the evidence file was written.
    with path.open("a") as file:
        file.write(json.dumps({**dense(1, run="run-1")[0], "step": 60}) + "\n")
        file.write('{"step": 61, "an unfinished li')
    os.utime(path, (path.stat().st_atime, path.with_suffix(".evidence.json").stat().st_mtime + 5))
    (row,) = tensorquill.list_runs(folder)
    assert row["evidence"]["state"] == "older_than_records" and row["last_step"] == 60
    assert row["first_incident"] == {"kind": "grad_spike", "step": 40}
    # A compact evidence file still has what a row needs, and is said to be compact.
    path.write_text("".join(json.dumps(r) + "\n" for r in spiked("run-1")))
    assert cli(str(path), "--compact", "-o", str(path.with_suffix(".evidence.json")), cwd=folder).returncode == 0
    (row,) = tensorquill.list_runs(folder)
    assert row["evidence"].get("compact") is True and row["first_incident"] == {"kind": "grad_spike", "step": 40}


def test_a_manifest_with_no_records_yet_and_a_file_with_a_name_of_its_own_are_both_listed(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "starting.run.json").write_text(json.dumps({"run_id": "starting", "project": "granite", "name": "just-began"}))
    write_capture(folder / "trainer.jsonl", dense(5, run="hf-run"))
    (folder / "hf-run.run.json").write_text(json.dumps({"run_id": "hf-run", "project": "granite", "name": "trainer"}))
    rows = {row["run_id"]: row for row in tensorquill.list_runs(folder)}
    assert rows["starting"]["evidence"]["state"] == "no_records" and rows["starting"]["name"] == "just-began"
    assert (rows["hf-run"]["name"], rows["hf-run"]["last_step"], rows["hf-run"]["path"]) == (
        "trainer", 4, str(folder / "trainer.jsonl"))
    with pytest.raises(FileNotFoundError):
        tensorquill.list_runs(tmp_path / "absent")


def hand_store(root: Path, runs: dict[str, tuple[dict | None, list[dict]]]) -> Path:
    """A server's store written by hand (as `test_mcp_server.store`): each run with one rank
    whose report is current, one part that is not Parquet at all, and its status."""
    for run_id, (manifest, incidents) in runs.items():
        folder = root / "runs" / run_id
        parts = folder / "rank=0" / "attempt=1"
        parts.mkdir(parents=True)
        (parts / "part-0-59.parquet").write_bytes(b"this is not a Parquet file")
        (folder / "rank=0" / "incidents.json").write_text(json.dumps({**report(0, incidents), "run_id": run_id}))
        assert ev.read_server_report(folder / "rank=0")["state"] == "current"
        (folder / "watchdog.json").write_text(json.dumps({
            "schema_version": 1, "summary": "finished", "world_size": 1,
            "ranks": [{"rank": 0, "state": "finished", "last_step": 59}],
        }))
        if manifest is not None:
            (folder / "run.json").write_text(json.dumps({"run_id": run_id, **manifest}))
    return root


def test_a_servers_store_is_listed_from_its_small_files_without_opening_a_part(tmp_path):
    store = hand_store(tmp_path / "store", {
        "run-1": ({"project": "granite", "name": "baseline", "created_at": "2026-11-17T10:00:00Z"}, []),
        "run-2": ({"project": "granite", "name": "after-the-fix", "created_at": "2026-11-17T11:00:00Z"}, [SPIKE, NOT_FINITE]),
        "old-run": (None, []),
    })
    for source in (store, store / "runs"):
        rows = tensorquill.list_runs(source)
        assert [(r["project"], r["name"], r["run_id"], r["source"]) for r in rows] == [
            ("granite", "baseline", "run-1", "store"), ("granite", "after-the-fix", "run-2", "store"),
            ("uncategorized", "old-run", "old-run", "store")]
        fixed = rows[1]
        assert (fixed["run_state"], fixed["last_step"], fixed["incidents"], fixed["first_incident"], fixed["ranks"]) == (
            "unhealthy", 59, {"anomaly": 1, "invariant": 1}, {"kind": "grad_spike", "step": 40}, [0])
    assert [r["run_id"] for r in tensorquill.list_runs(store / "runs" / "run-2")] == ["run-2"]
    # A folder may hold local files and a store both; a run in both is two rows.
    local_run(store, "run-1", project="granite", name="baseline")
    assert [(r["run_id"], r["source"]) for r in tensorquill.list_runs(store, project="granite")] == [
        ("run-1", "local"), ("run-1", "store"), ("run-2", "store")]


def test_the_printed_listing_groups_by_project_and_the_json_option_gives_the_rows(out):
    listed = cli("runs", str(out), cwd=out.parent)
    assert listed.returncode == 0, listed.stderr
    text = listed.stdout
    assert text.index("project granite (2 runs)") < text.index("baseline") < text.index("after-the-fix") < text.index(
        "project uncategorized (1 run)")
    assert "grad_spike at step 40" in text and "1 anomaly" in text and "old-run: no run manifest" in text
    # The default folder is tensorquill-out, where init writes.
    assert cli("runs", cwd=out.parent).stdout == text.replace(str(out), "tensorquill-out")
    only = cli("runs", str(out), "--project", "uncategorized", "--json", cwd=out.parent)
    assert [row["run_id"] for row in json.loads(only.stdout)] == ["old-run"]
    absent = cli("runs", "nowhere", cwd=out.parent)
    assert absent.returncode == 2 and "does not exist" in absent.stderr


# --- The two verbs and a path of the same name ---------------------------------------------


@pytest.mark.parametrize("entry", ["module", "console script"])
def test_runs_and_compare_are_verbs_at_both_entry_points_whatever_is_on_disk(entry, out):
    cwd = out.parent
    assert "project granite" in cli("runs", str(out), cwd=cwd, entry=entry).stdout
    compared = cli("compare", "baseline", "after-the-fix", cwd=cwd, entry=entry)
    assert compared.returncode == 0 and "Comparing 2 runs" in compared.stdout, compared.stderr
    assert json.loads((cwd / "compare.json").read_text())["view"] == "compare"

    # A capture that is itself named `compare`, and a folder named `runs`.
    write_capture(cwd / "compare", spiked("odd-name"))
    (cwd / "runs").mkdir()
    local_run(cwd / "runs", "inner", project="inner-project", name="inner")
    as_verb = cli("runs", cwd=cwd, entry=entry)
    assert "project granite" in as_verb.stdout and "inner-project" not in as_verb.stdout
    assert "read as the verb" in as_verb.stderr and "./runs" in as_verb.stderr
    as_verb = cli("compare", "run-1", "run-2", cwd=cwd, entry=entry)
    assert as_verb.returncode == 0 and "./compare" in as_verb.stderr
    # Written as a path, each is read as before: the capture gets its evidence file.
    as_path = cli("./compare", cwd=cwd, entry=entry)
    assert as_path.returncode == 0 and "wrote evidence" in as_path.stdout, as_path.stderr
    assert json.loads((cwd / "compare.evidence.json").read_text())["incidents"][0]["kind"] == "grad_spike"
    assert "inner-project" in cli("runs", "./runs", cwd=cwd, entry=entry).stdout
    # A path that only starts with a verb is a path.
    write_capture(cwd / "runs.jsonl", dense(5, run="plain"))
    assert cli("runs.jsonl", cwd=cwd, entry=entry).returncode == 0
    (cwd / "compare.json").unlink()


# --- Finding a run by ID or name --------------------------------------------------------------


def test_an_id_wins_and_a_name_that_fits_several_runs_is_refused_with_their_ids(tmp_path, monkeypatch):
    folder = tmp_path / "out"
    local_run(folder, "run-1", project="granite", name="baseline")
    local_run(folder, "run-2", spiked("run-2"), project="granite", name="baseline")
    local_run(folder, "run-3", project="llama", name="baseline")
    local_run(folder, "baseline-x", project="llama", name="run-1")  # another run's ID as its name
    assert runs_module.find_run(folder, "run-1")["run_id"] == "run-1"
    assert runs_module.find_run(folder, "baseline", project="llama")["run_id"] == "run-3"
    with pytest.raises(runs_module.RunLookupError) as refused:
        runs_module.find_run(folder, "baseline")
    assert all(run_id in str(refused.value) for run_id in ("run-1", "run-2", "run-3")) and "fits 3 runs" in str(refused.value)
    with pytest.raises(ValueError, match=r"fits 2 runs in project 'granite'.*run-1 .*run-2 "):
        tensorquill.compare_runs("baseline", "run-3", source=folder, project="granite")
    with pytest.raises(runs_module.RunLookupError, match="no run with the ID or name 'nobody'.*Runs there: .*run-2"):
        tensorquill.compare_runs("run-1", "nobody", source=folder)
    refused = cli("compare", "baseline", "run-3", "--source", str(folder), cwd=tmp_path)
    assert refused.returncode == 2 and "run-1" in refused.stderr and "run-2" in refused.stderr and "give the run ID" in refused.stderr
    assert not (tmp_path / "compare.json").exists()
    # The same run twice, and one run alone, are nothing to compare.
    with pytest.raises(ValueError, match="same run"):
        tensorquill.compare_runs("run-3", "baseline", source=folder, project="llama")
    with pytest.raises(ValueError, match="at least two"):
        tensorquill.compare_runs("run-1", source=folder)
    # Only the refusal itself stops the search: a path or a run whose own name holds the word
    # "fits" is found as any other.
    local_run(folder, "retrofits", spiked("retrofits"), project="llama", name="it fits 2 runs")
    misfits = local_run(tmp_path / "misfits", "elsewhere", spiked("elsewhere"))
    by_path = tensorquill.compare_runs(str(misfits), "run-3", source=folder)
    assert [run["run_id"] for run in by_path["runs"]] == ["elsewhere", "run-3"]
    with pytest.raises(runs_module.RunLookupError, match="no run with the ID or name") as missing:
        tensorquill.compare_runs(str(tmp_path / "misfits" / "absent.jsonl"), "run-3", source=folder)
    assert not isinstance(missing.value, runs_module.AmbiguousRunName)
    with pytest.raises(runs_module.AmbiguousRunName):
        runs_module.find_run(folder, "baseline")
    # A name that fits several runs is never taken for a path, even when one is there.
    write_capture(tmp_path / "baseline", dense(5, run="a-file"))
    monkeypatch.chdir(tmp_path)
    assert Path("baseline").is_file()
    with pytest.raises(ValueError, match="fits 3 runs"):
        tensorquill.compare_runs("baseline", "run-3", source=folder)


# --- Comparing: the setup ---------------------------------------------------------------------


def test_part_1_lists_each_setup_field_and_config_key_that_differs_and_leaves_out_a_run_without_a_manifest(tmp_path):
    folder = tmp_path / "out"
    local_run(folder, "run-1", project="granite", name="a", group="sweep", torch_version="2.14.0",
              config={"lr": 0.01, "warmup": 100, "layers": 4})
    local_run(folder, "run-2", project="granite", name="b", group="sweep", torch_version="2.15.0", world_size=2,
              config={"lr": 0.1, "layers": 4, "seed": 7})
    local_run(folder, "old-run")
    result = tensorquill.compare_runs("run-1", "run-2", "old-run", source=folder)
    setup = result["setup"]
    assert setup["differs"] == [
        {"field": "world_size", "values": {"A": 1, "B": 2}},
        {"field": "torch_version", "values": {"A": "2.14.0", "B": "2.15.0"}},
        {"field": "config.lr", "values": {"A": 0.01, "B": 0.1}},
        {"field": "config.seed", "values": {"A": None, "B": 7}, "not_set_in": ["A"]},
        {"field": "config.warmup", "values": {"A": 100, "B": None}, "not_set_in": ["B"]},
    ]
    assert setup["same"] == ["project", "group", "sdk_version"] and setup["config_keys_same"] == ["layers"]
    assert setup["no_manifest"] == ["C"]
    assert [run["label"] for run in result["runs"]] == ["A", "B", "C"]
    assert "config.lr: A 0.01, B 0.1" in result["summary"] and "no run manifest for C" in result["summary"]
    assert result["divergence"]["state"] == "not_asked"  # two runs only


# --- Comparing: what was checked, before any incident -------------------------------------------


def toy_run(path: Path, run_id: str, steps: int, capture: str, **labels) -> Path:
    """A run of the toy MoE model, the same model and batches every time, with the capture
    mode's own settings."""
    from conftest import ToyMoE

    torch.manual_seed(0)
    model = ToyMoE()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    with tensorquill.init(model, run_id=run_id, routers={"router": 4}, capture=capture, every=10,
                          sink=tensorquill.JsonlSink(path), **labels):
        for _ in range(steps):
            optimizer.zero_grad()
            model(torch.randn(16, 8)).pow(2).mean().backward()
            optimizer.step()
    return path


def before(result: dict, first: str, second: str) -> bool:
    keys = list(result)
    return keys.index(first) < keys.index(second)


def test_a_router_only_run_against_a_full_capture_run_says_what_was_checked_before_any_incident(tmp_path, toy_moe):
    folder = tmp_path / "out"
    folder.mkdir()
    toy_run(folder / "routers.rank0.jsonl", "routers", 70, "routers", project="toy", name="cheap")
    full = toy_run(folder / "full.rank0.jsonl", "full", 70, "full", project="toy", name="debug")
    # A fault only full capture can see: one block's output goes NaN at step 50.
    records = ev.load_records(full)
    layer = next(iter(records[50]["layers"]))
    records[50]["layers"][layer]["abs_max"] = float("nan")
    write_capture(full, records)

    result = tensorquill.compare_runs("cheap", "debug", source=folder)
    differs = {item["kind"]: item["looked"] for item in result["checked"]["differs"]}
    for kind in ("non_finite", "grad_spike", "activation_spike", "activation_drift", "dead_activations"):
        assert differs[kind] == {"A": "not_configured", "B": "checked"}, kind
    assert "moe_imbalance" not in differs and "starved_experts" not in differs  # both runs watch the router
    assert result["checked"]["ranks_read"] == {"A": 0, "B": 0}
    assert result["checked"]["same"] + len(differs) == 32

    (nan,) = [entry for entry in result["incidents"]["entries"] if entry["kind"] == "non_finite"]
    assert nan["in"] == ["B"] and nan["first_step"] == {"B": 50} and nan["where"]["B"]["layers"] == [layer]
    assert nan["absent_from"] == {"A": "not_configured"}, "A's silence is not a 'none'"
    # The order is part of the format, in the file and in the text.
    assert before(result, "setup", "checked") and before(result, "checked", "incidents") and before(result, "incidents", "divergence")
    text = result["summary"]
    assert text.index("1. Setup") < text.index("2. What was checked") < text.index(
        "non_finite: not configured in A; checked") < text.index("3. Incidents") < text.index(
        "B only: non_finite") < text.index("A did not look: the kind is not configured there") < text.index("4. Where the runs part")
    # Part 4 is asked of these two (their records were read): a router-only run and a full capture share
    # too little to be a pair, so it gives its reason and never a step.
    assert result["divergence"]["state"] in ("no_answer", "no_divergence") and result["divergence"]["divergence"] is None
    assert result["divergence"]["detail"] and result["divergence"]["labels"] == {"a": "A", "b": "B"}


def test_a_short_run_against_a_long_one_says_which_kinds_have_not_had_enough_history(tmp_path):
    folder = tmp_path / "out"
    local_run(folder, "short", dense(5, run="short"), project="p", name="short")
    local_run(folder, "medium", dense(30, run="medium"), project="p", name="medium")
    local_run(folder, "long", spiked("long"), project="p", name="long")
    result = tensorquill.compare_runs("short", "long", "medium", source=folder)
    spike = next(item for item in result["checked"]["differs"] if item["kind"] == "grad_spike")
    assert spike["looked"] == {"A": "not_enough_history", "B": "checked", "C": "checked"}
    assert spike["records"] == {"A": 5, "B": 60, "C": 30} and spike["records_needed"] == {"A": 13, "B": 13, "C": 13}
    (entry,) = result["incidents"]["entries"]
    assert (entry["kind"], entry["in"], entry["first_step"]) == ("grad_spike", ["B"], {"B": 40})
    # Neither silence is a "none": A's rule had no history, and C looked but stopped at step 29.
    assert entry["absent_from"] == {"A": "not_enough_history", "C": "ended_before_step"}
    text = result["summary"]
    assert "grad_spike: checked, not yet with enough history (5 of the 13 records its slowest rule needs) in A" in text
    assert text.index("2. What was checked") < text.index("not yet with enough history") < text.index("3. Incidents")
    assert "C ended before this step" in text and "A checked this kind without enough history yet" in text


# --- Comparing: the incidents -------------------------------------------------------------------


def test_incidents_are_matched_by_kind_and_what_they_name_with_each_runs_first_step(tmp_path):
    folder = tmp_path / "out"
    local_run(folder, "a", dense(80, run="a", spike=(40, "blocks.1.weight", 60.0)), project="p", name="a")
    late = dense(80, run="b", spike=(55, "blocks.1.weight", 60.0))
    late[70]["layers"]["blocks.2"]["abs_max"] = float("inf")
    local_run(folder, "b", late, project="p", name="b")
    other = dense(80, run="c", spike=(40, "blocks.0.weight", 600.0))
    local_run(folder, "c", other, project="p", name="c")
    result = tensorquill.compare_runs("a", "b", "c", source=folder)
    found = {(e["kind"], json.dumps(e["where"][e["in"][0]], sort_keys=True)): e for e in result["incidents"]["entries"]}
    same = next(e for e in result["incidents"]["entries"] if e["in"] == ["A", "B"])
    assert (same["kind"], same["first_step"]) == ("grad_spike", {"A": 40, "B": 55})
    assert same["where"]["A"] == same["where"]["B"] == {"parameters": ["blocks.1.weight"]}
    assert same["absent_from"] == {"C": "checked"}
    only_c = next(e for e in result["incidents"]["entries"] if e["in"] == ["C"])
    assert only_c["where"]["C"] == {"parameters": ["blocks.0.weight"]} and only_c["absent_from"] == {"A": "checked", "B": "checked"}
    only_b = next(e for e in result["incidents"]["entries"] if e["in"] == ["B"])
    assert (only_b["kind"], only_b["class"], only_b["first_step"]) == ("non_finite", "invariant", {"B": 70})
    assert len(found) == 3 and result["incidents"]["counts"] == {"A": 1, "B": 2, "C": 1}
    assert result["incidents"]["only_in"] == {"A": [], "B": [result["incidents"]["entries"].index(only_b)],
                                              "C": [result["incidents"]["entries"].index(only_c)]}
    assert result["incidents"]["in_all"] == []
    assert "A, B: grad_spike (anomaly), first at A step 40, B step 55: parameters blocks.1.weight" in result["summary"]

    both = tensorquill.compare_runs("a", "b", source=folder)
    assert "3. Incidents: 0 in A only, 1 in B only, 1 in both" in both["summary"]
    assert [e["in"] for e in both["incidents"]["entries"]] == [["A", "B"], ["B"]]


def test_the_moe_identity_is_the_layer_and_router_and_each_run_keeps_its_own_experts():
    one = {"kind": "moe_imbalance", "step": 100, "layer": 3, "router": "blocks.3.router", "overloaded": [2], "dead": []}
    two = {**one, "step": 400, "overloaded": [5, 6], "dead": [1]}
    assert cmp.incident_identity(one) == cmp.incident_identity(two) != cmp.incident_identity({**one, "layer": 4})
    assert cmp.incident_where(two) == {"layer": 3, "router": "blocks.3.router", "overloaded": [5, 6], "dead": [1]}
    attention = {"kind": "attention_collapse", "step": 2000, "confirmed_step": 3000, "modules": [{"module": "h.1.attn"}]}
    assert cmp.incident_identity(attention) == cmp.incident_identity({**attention, "step": 5000, "confirmed_step": 6000})
    unknown = {"kind": "a_kind_from_later", "step": 7, "thing": "x", "scalars": {"loss": 1.0}}
    assert cmp.incident_identity(unknown) == cmp.incident_identity({**unknown, "step": 9, "scalars": {"loss": 2.0}})
    assert cmp.incident_identity(unknown) != cmp.incident_identity({**unknown, "thing": "y"})


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    """`seeded_failures.py --toy` on CPU: a healthy run and the two seeded ones, each with the
    script's own steps, model and batches."""
    pytest.importorskip("transformers")
    sys.path.insert(0, str(EXAMPLES))
    try:
        import seeded_failures as sf
    finally:
        sys.path.remove(str(EXAMPLES))
    out = tmp_path_factory.mktemp("seeded") / "out"
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        sf.main(["--toy", "--out", str(out), "--case", "all"])
    assert "All cases passed" in printed.getvalue(), printed.getvalue()
    return out


@pytest.mark.parametrize("case, step", [("grad_spike", 250), ("stalled_weights", 249)])
def test_a_healthy_run_and_a_seeded_run_compare_as_one_incident_in_the_second_only_on_the_seeded_parameter(seeded, case, step):
    healthy, planted = seeded / "healthy" / "healthy.jsonl", seeded / case / f"{case}.jsonl"
    result = tensorquill.compare_runs(healthy, planted)
    assert [(run["name"], run["run_id"], run["project"], run["incidents"]) for run in result["runs"]] == [
        ("healthy", "seeded-healthy", "seeded-failures", 0), (case, f"seeded-{case}", "seeded-failures", 1)]
    assert result["runs"][0]["run_state"] == "healthy" and result["runs"][1]["run_state"] == "unhealthy"
    (entry,) = result["incidents"]["entries"]
    assert (entry["kind"], entry["in"], entry["first_step"]) == (case, ["B"], {"B": step})
    assert entry["where"]["B"] == {"parameters": [SEEDED]}
    assert entry["absent_from"] == {"A": "checked"}, "the healthy run looked, with enough history, and found none"
    assert result["incidents"]["only_in"] == {"A": [], "B": [0]} and result["incidents"]["in_all"] == []
    # The same capture settings, so nothing differs in what was checked; the seed is in the config.
    assert result["checked"]["differs"] == [] and result["checked"]["same"] == 32
    assert {"field": "config.case", "values": {"A": "healthy", "B": case}} in result["setup"]["differs"]
    assert "3. Incidents: 0 in A only, 1 in B only, 0 in both" in result["summary"]
    assert f"B only: {case} ({entry['class']}), first at B step {step}: parameters {SEEDED}" in result["summary"]
    assert entry["class"] == ev.INCIDENT_CLASSES[case]
    assert "A checked this kind and found none" in result["summary"]


def test_the_seeded_runs_are_found_by_name_in_one_folder_and_the_command_writes_compare_json(seeded, tmp_path):
    folder = tmp_path / "together"
    folder.mkdir()
    for case in ("healthy", "grad_spike", "stalled_weights"):
        (folder / f"seeded-{case}.rank0.jsonl").write_bytes((seeded / case / f"{case}.jsonl").read_bytes())
        (folder / f"seeded-{case}.run.json").write_bytes((seeded / case / f"seeded-{case}.run.json").read_bytes())
    rows = tensorquill.list_runs(folder, project="seeded-failures")
    assert [row["name"] for row in rows] == ["healthy", "grad_spike", "stalled_weights"]
    done = cli("compare", "healthy", "grad_spike", "stalled_weights", "--source", str(folder), "--project",
               "seeded-failures", "-o", "out/three.json", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    written = json.loads((tmp_path / "out" / "three.json").read_text())
    assert done.stdout.rstrip().endswith("wrote comparison -> out/three.json") and written["summary"] in done.stdout
    assert [(e["kind"], e["in"]) for e in written["incidents"]["entries"]] == [("stalled_weights", ["C"]), ("grad_spike", ["B"])]
    assert all(e["absent_from"] in ({"A": "checked", "B": "checked"}, {"A": "checked", "C": "checked"})
               for e in written["incidents"]["entries"])
    assert "Where the runs part: where runs part is answered for two runs only" in done.stdout


# --- Comparing: which rank, and where it was read ------------------------------------------------


def test_a_run_of_many_ranks_is_read_on_rank_0_and_says_the_others_were_not_compared(tmp_path):
    folder = tmp_path / "out"
    local_run(folder, "one", project="p", name="one")
    local_run(folder, "two", dense(60, run="two"), project="p", name="two")
    local_run(folder, "two", dense(60, run="two", rank=1, spike=(40, "blocks.1.weight", 60.0)), rank=1)
    result = tensorquill.compare_runs("one", "two", source=folder)
    assert result["runs"][1]["read_from"]["rank"] == 0 and result["runs"][1]["read_from"]["ranks"] == [0, 1]
    assert result["incidents"]["entries"] == [] and result["incidents"]["other_ranks_not_compared"] == {"B": [1]}
    assert "rank 0 of 2" in result["summary"] and "incidents on other ranks were not compared: B ranks 1" in result["summary"]
    assert "(A rank 0, B rank 0)" in result["summary"]
    with pytest.raises(ValueError, match="no rank 3"):
        tensorquill.compare_runs("one", "two", source=folder, rank=3)
    # Rank 1, asked for: the run with one rank is still read on the rank it has.
    on_one = tensorquill.compare_runs("one", "two", source=folder, rank=1)
    assert [run["read_from"]["rank"] for run in on_one["runs"]] == [0, 1]
    assert [(e["kind"], e["in"]) for e in on_one["incidents"]["entries"]] == [("grad_spike", ["B"])]
    assert on_one["incidents"]["other_ranks_not_compared"] == {"B": [0]}


def test_a_capture_that_is_still_being_written_is_compared_on_its_whole_lines_and_says_so(tmp_path):
    folder = tmp_path / "out"
    local_run(folder, "done", project="p", name="done")
    going = local_run(folder, "going", spiked("going"), project="p", name="going")
    with going.open("a") as file:
        file.write('{"schema_version": 1, "run_id": "going", "rank": 0, "step": 60, "an unfinished li')
    result = tensorquill.compare_runs("done", "going", source=folder)
    assert result["runs"][1]["read_from"]["unfinished_lines_left_out"] is True and result["runs"][1]["last_step"] == 59
    assert "unfinished_lines_left_out" not in result["runs"][0]["read_from"]
    assert [(e["kind"], e["in"]) for e in result["incidents"]["entries"]] == [("grad_spike", ["B"])]
    assert "the run may still be going" in result["summary"]
    (folder / "empty.rank0.jsonl").write_text("")
    with pytest.raises(ValueError, match="run empty has no whole record"):
        tensorquill.compare_runs("done", "empty", source=folder)


def test_runs_given_as_paths_a_store_read_from_its_report_and_a_report_that_cannot_be_used(tmp_path):
    store = hand_store(tmp_path / "store", {
        "run-1": ({"project": "granite", "name": "baseline", "config": {"lr": 0.01}}, []),
        "run-2": ({"project": "granite", "name": "after-the-fix", "config": {"lr": 0.1}}, [SPIKE]),
    })
    before_tree = tree(tmp_path)
    result = tensorquill.compare_runs("baseline", "after-the-fix", source=store)
    # The reports are current, so the parts (which are not Parquet at all) were never opened.
    assert all("no part was read" in run["read_from"]["judged_by"] for run in result["runs"])
    assert [(run["name"], run["last_step"], run["read_from"]["kind"]) for run in result["runs"]] == [
        ("baseline", 59, "server store"), ("after-the-fix", 59, "server store")]
    (entry,) = result["incidents"]["entries"]
    assert (entry["kind"], entry["in"], entry["where"]["B"]) == ("grad_spike", ["B"], {"parameters": ["blocks.1.weight"]})
    # These hand-written reports are from before `kind_checks`: said, and never read as "checked".
    assert result["checked"]["not_said_by"] == ["A", "B"] and entry["absent_from"] == {"A": "not_said"}
    assert "A's evidence does not say whether it looked" in result["summary"]
    assert {"field": "config.lr", "values": {"A": 0.01, "B": 0.1}} in result["setup"]["differs"]
    assert tree(tmp_path) == before_tree, "compare_runs writes nothing"

    # As paths: a run folder, a capture and an evidence file.
    capture = local_run(tmp_path / "files", "local-run", spiked("local-run"), project="granite", name="local")
    evidence_file = tmp_path / "files" / "saved.json"
    ev.write_evidence(ev.build_evidence(capture), evidence_file)
    mixed = tensorquill.compare_runs(store / "runs" / "run-1", capture, evidence_file, source=tmp_path / "nowhere")
    assert [run["read_from"]["kind"] for run in mixed["runs"]] == ["server store", "local records", "evidence file"]
    assert [run["name"] for run in mixed["runs"]] == ["baseline", "local", "local"]
    assert [e["in"] for e in mixed["incidents"]["entries"]] == [["B", "C"]]
    with pytest.raises(ValueError, match="IDs and names only|no run with the ID or name"):
        tensorquill.compare_runs(str(capture), "run-1", source=store, paths=False)
    with pytest.raises(ValueError, match="does not exist"):
        tensorquill.compare_runs(tmp_path / "absent.jsonl", capture)
    with pytest.raises(ValueError, match="holds 2 runs"):
        tensorquill.compare_runs(store, capture)

    # A report that is behind its store cannot be used, so the parts are read, and here they cannot be.
    (store / "runs" / "run-2" / "rank=0" / "attempt=1" / "part-60-99.parquet").write_bytes(b"not Parquet either")
    with pytest.raises(Exception) as failed:
        tensorquill.compare_runs("baseline", "after-the-fix", source=store)
    assert not isinstance(failed.value, AssertionError)


def test_two_runs_in_a_real_servers_store_are_listed_and_compared_by_name(tmp_path, toy_moe, monkeypatch):
    pytest.importorskip("pyarrow.flight")
    import test_flight_e2e as e2e

    if e2e.BINARY is None:
        pytest.skip("tensorquill-server is not built")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MLFLOW_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    store = tmp_path / "store"
    server = e2e.Server(store, 0, "--flush-seconds", "0.2", "--watchdog-seconds", "0.2")
    try:
        monkeypatch.setenv("TENSORQUILL_ENDPOINT", server.endpoint)
        toy_run(tmp_path / "local" / "run-1.rank0.jsonl", "run-1", 40, "routers", project="granite",
                name="cheap", config={"capture": "routers"})
        toy_run(tmp_path / "local" / "run-2.rank0.jsonl", "run-2", 40, "full", project="granite",
                name="debug", config={"capture": "full"})
        monkeypatch.delenv("TENSORQUILL_ENDPOINT")
    finally:
        assert server.terminate() == 0
    rows = tensorquill.list_runs(store, project="granite")
    assert [(row["name"], row["run_id"], row["source"], row["last_step"]) for row in rows] == [
        ("cheap", "run-1", "store", 39), ("debug", "run-2", "store", 39)]
    assert all(row["run_state"] in ev.RUN_STATES and row["kinds_checked"]["of"] == 32 for row in rows)
    assert rows[0]["kinds_checked"]["checked"] < rows[1]["kinds_checked"]["checked"]
    assert rows[0]["kinds_checked"]["rank"] == 0  # until the all-ranks view says what was checked
    listed = cli("runs", str(store), "--project", "granite", cwd=tmp_path)
    assert "cheap" in listed.stdout and "debug" in listed.stdout and "project granite (2 runs)" in listed.stdout

    from_store = tensorquill.compare_runs("cheap", "debug", source=store)
    from_local = tensorquill.compare_runs("cheap", "debug", source=tmp_path / "local")
    assert all("no part was read" in run["read_from"]["judged_by"] for run in from_store["runs"])
    assert from_store["setup"]["differs"] == from_local["setup"]["differs"] == [
        {"field": "config.capture", "values": {"A": "routers", "B": "full"}}]
    # The server's report and the detectors here agree on what each run checked of the server's kinds.
    server_kinds = set(ev.SERVER_KINDS)
    pick = lambda result: {item["kind"]: item["looked"] for item in result["checked"]["differs"] if item["kind"] in server_kinds}  # noqa: E731
    assert pick(from_store) == pick(from_local) and pick(from_store)["grad_spike"] == {"A": "not_configured", "B": "checked"}
    brief = lambda result: [(e["kind"], e["in"], e["first_step"], e["where"], e["absent_from"])  # noqa: E731
                            for e in result["incidents"]["entries"]]
    assert brief(from_store) == brief(from_local) and brief(from_store)

    # Without the server's report the parts are read and judged here: the same answer, and it
    # says who judged. With `read_parts=False` (the MCP tool) the run is refused with the reason.
    (store / "runs" / "run-2" / "rank=0" / "incidents.json").unlink()
    judged_here = tensorquill.compare_runs("cheap", "debug", source=store)
    assert "no part was read" in judged_here["runs"][0]["read_from"]["judged_by"]
    assert "the Python detectors, here, on the stored parts" in judged_here["runs"][1]["read_from"]["judged_by"]
    assert "the server's report is missing" in judged_here["runs"][1]["read_from"]["judged_by"]
    assert brief(judged_here) == brief(from_local) and pick(judged_here) == pick(from_local)
    assert judged_here["runs"][1]["last_step"] == 39
    with pytest.raises(runs_module.RunLookupError, match="run-2 rank 0 has no current report from its server: it is missing"):
        tensorquill.compare_runs("cheap", "debug", source=store, read_parts=False)
