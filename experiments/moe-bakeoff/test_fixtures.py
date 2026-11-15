"""Check that the SDK's detectors find each fixture's planted failure at the step it is planted.

Every fixture goes through the whole pipeline (`generate.export_fixture`: `JsonlSink`,
`load_records`, `build_evidence`, `export_rerun`), and the assertions read the results back
from disk. The expected answers are `Fixture.answer` and `Fixture.events` in generate.py,
which hold the planted values; nothing here is taken from the detectors' output.

Run from the repository root with the project environment (it has the SDK and rerun-sdk):

    .venv/bin/python -m pytest experiments/moe-bakeoff -q
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import tensorquill
import torch
from rerun.chunk import RrdReader
from torch import nn

import generate
import verify_rerun
from generate import FIXTURES, router_name

NAMES = [fixture.name for fixture in FIXTURES]
MOE = [fixture for fixture in FIXTURES if fixture.answer["kind"] == "moe_imbalance"]
DENSE = [fixture for fixture in FIXTURES if fixture.answer["kind"] != "moe_imbalance"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Run every fixture through the pipeline once: {name: (directory, evidence read back)}."""
    root = tmp_path_factory.mktemp("fixtures")
    out = {}
    for fixture in FIXTURES:
        directory = root / fixture.name
        generate.export_fixture(fixture, directory)
        evidence = json.loads((directory / f"{fixture.name}.evidence.json").read_text(encoding="utf-8"))
        out[fixture.name] = (directory, evidence)
    return out


def summary_text(store):
    return [value for chunk in store.stream().filter(content="/incident/summary", is_static=True).to_chunks()
            for value in chunk.to_record_batch()["TextDocument:text"].to_pylist()]


def logged(store):
    """Everything a recording holds, by entity and component: {(entity, component): [(step, value)]}."""
    held = {}
    for column in store.schema().component_columns():
        if column.is_property:
            continue
        rows = []
        for chunk in store.stream().filter(content=column.entity_path, is_static=column.is_static).to_chunks():
            batch = chunk.to_record_batch()
            if column.component in batch.column_names:
                steps = batch["training_step"].to_pylist() if "training_step" in batch.column_names else [None] * batch.num_rows
                rows += zip(steps, batch[column.component].to_pylist())
        held[column.entity_path, column.component] = sorted(rows, key=lambda row: -1 if row[0] is None else row[0])
    return held


def flagged(evidence):
    return {(s["step"], s["layer"]) for s in evidence["samples"] if s["overloaded"] or s["dead"]}


@pytest.mark.parametrize("fixture", FIXTURES, ids=NAMES)
def test_the_detectors_report_the_planted_failure_and_nothing_else(fixture, exported):
    _, evidence = exported[fixture.name]
    key = fixture.answer
    assert [incident["kind"] for incident in evidence["incidents"]] == [key["kind"]]
    incident = evidence["incidents"][0]
    assert incident["step"] == key["step"]
    if key["kind"] == "moe_imbalance":
        assert (incident["layer"], incident["router"]) == (key["layer"], router_name(key["layer"]))
        assert incident["overloaded"] == key["overloaded"]
        assert incident["dead"] == key["dead"]
        first = tensorquill.first_incident(evidence)
        assert (first["step"], first["layer"]) == (key["step"], key["layer"])
        assert evidence["top_k"] == {router_name(layer): fixture.top_k for layer in range(1, evidence["layer_count"] + 1)}
    else:
        assert evidence["layer_count"] == 0  # a dense run has no routers
        if key["kind"] == "grad_spike":
            assert incident["parameters"][0]["name"] == key["parameter"]
            assert incident["parameters"][0]["rising_since"] == key["rising_since"]
            assert incident["spike_count"] == 1
            assert incident["ratio"] >= tensorquill.evidence.SPIKE_FACTOR
        else:
            assert incident["modules"][0]["module"] == key["module"]
            assert incident["modules"][0]["stat"] == key["stat"]
            assert incident["affected"] == 1  # the block with massive activations is not reported
        # The incident carries what the loss did, for the summary and the Logged tab.
        assert "loss" in incident["scalars"]
    opening = {
        "moe_imbalance": "First sampled bad step", "grad_spike": "Gradient spike at step",
        "activation_spike": "Activation spike at step",
    }[key["kind"]]
    assert tensorquill.incident_summary(evidence).startswith(f"{opening} {key['step']}")


@pytest.mark.parametrize("fixture", MOE, ids=[fixture.name for fixture in MOE])
def test_only_the_planted_samples_are_flagged_and_decoys_are_left_alone(fixture, exported):
    _, evidence = exported[fixture.name]
    expected, found = fixture.flagged_samples(), flagged(evidence)
    first = (fixture.answer["step"], fixture.answer["layer"])
    assert first in found and first == min(expected)
    assert found == expected
    for sample in evidence["samples"]:
        if (sample["step"], sample["layer"]) in expected and sample["layer"] == fixture.answer["layer"]:
            assert sample["overloaded"] == fixture.answer["overloaded"]
            assert sample["dead"] == fixture.answer["dead"]


@pytest.mark.parametrize("fixture", FIXTURES, ids=NAMES)
def test_the_run_is_clean_until_the_planted_step_and_caught_when_it_arrives(fixture):
    step = fixture.answer["step"]
    records = fixture.records()
    before = tensorquill.build_evidence([r for r in records if r["step"] < step])
    assert before["incidents"] == [] and not flagged(before)
    so_far = tensorquill.build_evidence([r for r in records if r["step"] <= step])
    assert [(i["kind"], i["step"]) for i in so_far["incidents"]] == [(fixture.answer["kind"], step)]


@pytest.mark.parametrize("fixture", FIXTURES, ids=NAMES)
def test_the_command_line_gives_the_same_evidence_and_recording(fixture, exported, tmp_path):
    directory, _ = exported[fixture.name]
    subprocess.run(
        [sys.executable, "-m", "tensorquill", str(directory / f"{fixture.name}.jsonl"),
         "-o", str(tmp_path / "evidence.json"), "--rerun", str(tmp_path / "run.rrd")],
        check=True, capture_output=True, cwd=tmp_path,
    )
    assert (tmp_path / "evidence.json").read_bytes() == (directory / f"{fixture.name}.evidence.json").read_bytes()
    ours, theirs = RrdReader(directory / f"{fixture.name}.rrd").store(), RrdReader(tmp_path / "run.rrd").store()
    # The files differ in their start time and IDs (and in how Rerun splits them into chunks),
    # so compare what they hold.
    held = logged(ours)
    assert held and held == logged(theirs)


@pytest.mark.parametrize("fixture", FIXTURES, ids=NAMES)
def test_each_recording_holds_only_its_own_fixture(fixture, exported):
    # export_rerun starts a global recording per call, so seven exports in one process must
    # not leak into one another.
    directory, evidence = exported[fixture.name]
    reader = RrdReader(directory / f"{fixture.name}.rrd")
    assert [(e.application_id, e.recording_id) for e in reader.recordings()] == [("tensorquill", fixture.name)]
    assert len(reader.blueprints()) == 1
    store = reader.store()
    if fixture.answer["kind"] == "moe_imbalance":
        assert set(verify_rerun.read_samples(store, "/routing/status", "Image:format")) == set(fixture.steps)
    else:
        assert set(verify_rerun.read_samples(store, "/training/grad_norm", "Scalars:scalars")) == set(fixture.steps)
    assert summary_text(store) == [[tensorquill.incident_summary(evidence)]]


class ToyRouter(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.proj = nn.Linear(8, n_experts)

    def forward(self, h):
        logits = self.proj(h)
        return logits, logits.argmax(dim=-1)


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 8)
        self.router = ToyRouter(4)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        h = torch.relu(self.inp(x))
        _, choice = self.router(h)
        return self.out(h * choice[:, None])


def types(record):
    return {key: type(value) for key, value in record.items()}


def test_moe_records_have_the_shape_of_a_real_capture():
    torch.manual_seed(0)
    model, sink = ToyMoE(), tensorquill.MemorySink()
    session = tensorquill.init(model, run_id="real", sink=sink, every=1, auto_step=False,
                               routers={"router": tensorquill.Router(4, top_k=1)})
    model(torch.randn(4, 8)).sum().backward()
    session.step(3)
    session.close()
    (real,) = sink.records
    for fixture in MOE:
        for ours in fixture.records()[:1]:
            assert types(ours) == types(real)
            router, real_router = next(iter(ours["moe"])), next(iter(real["moe"]))
            assert {type(count) for count in ours["moe"][router]} == {type(count) for count in real["moe"][real_router]}
            assert types(ours["moe_stats"][router]) == types(real["moe_stats"][real_router])
            assert type(ours["moe_top_k"][router]) is type(real["moe_top_k"][real_router])
            assert ours["moe_stats"][router] == tensorquill.capture._router_stats(ours["moe"][router])


def test_dense_records_have_the_shape_of_a_real_capture():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer, sink = torch.optim.SGD(model.parameters(), lr=0.1), tensorquill.MemorySink()
    session = tensorquill.init(model, run_id="real", sink=sink, every=1, capture="full")
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        session.log(loss=loss.detach())
        optimizer.step()
    session.close()
    real = sink.records[-1]
    for fixture in DENSE:
        ours = fixture.records()[-1]
        assert types(ours) == types(real)
        assert set(next(iter(ours["layers"].values()))) == set(next(iter(real["layers"].values())))
        assert {type(v) for v in ours["lr"]} == {type(v) for v in real["lr"]}
        assert types(ours["scalars"]) == types(real["scalars"])


def test_the_seeded_fixture_is_the_readme_answer_key():
    by_key = {(r["step"], layer): counts for r in generate.BY_NAME["seeded"].records()
              for layer, counts in enumerate(r["moe"].values(), 1)}
    counts = by_key[1850, 31]
    assert counts[:14] == [0] * 14 and counts[41] == 46_080 and sum(counts) == 128_000
    assert by_key[1800, 31] == [1000] * 128
    assert all(c == [1000] * 128 for (step, layer), c in by_key.items() if (step, layer) not in {(1850, 31), (1900, 31)})
    evidence = tensorquill.build_evidence(generate.BY_NAME["seeded"].records())
    assert evidence["samples"][(1850 - 1750) // 50 * 60 + 30]["diversity"] == pytest.approx(0.3094112778)


def test_generation_is_the_same_in_another_process_and_writes_the_answer_key(exported, tmp_path):
    subprocess.run([sys.executable, str(Path(generate.__file__)), str(tmp_path), "--no-rerun"],
                   check=True, capture_output=True)
    for fixture in FIXTURES:
        directory, evidence = exported[fixture.name]
        for suffix in (".jsonl", ".evidence.json"):
            assert (tmp_path / fixture.name / f"{fixture.name}{suffix}").read_bytes() == \
                (directory / f"{fixture.name}{suffix}").read_bytes(), f"{fixture.name}{suffix}"
    assert json.loads((tmp_path / "stats.json").read_text()) == exported["seeded"][1]
    keys = json.loads((tmp_path / "answer-key.json").read_text())
    assert {name: key["answer"] for name, key in keys.items()} == {f.name: f.answer for f in FIXTURES}
