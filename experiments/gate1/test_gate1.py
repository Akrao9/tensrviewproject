"""Check the Gate 1 harness: the material per arm, the key, the answer format and the scoring.

Run from the repository root with the project environment (the build step needs the SDK):

    .venv/bin/python -m pytest experiments/gate1 -q

The key's planted values are compared with `Fixture.answer` and `Fixture.events` in
experiments/moe-bakeoff/generate.py, never with the detectors' output.
"""

import copy
import csv
import io
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

import harness

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(harness.BAKEOFF))
import generate  # noqa: E402

VALIDATION = HERE / "validation"
# What the hand-written wrong set should score, file by file: the questions it gets wrong.
WRONG = {
    "unseen-1": {"dead_experts"},
    "unseen-2": {"step"},
    "unseen-3": {"magnitude"},
    "unseen-4": {"overloaded_experts", "magnitude"},
    "unseen-5": {"location", "magnitude"},
    "unseen-6": {"step", "location", "overloaded_experts", "magnitude"},
}
MOE_NAMES = [f.name for f in generate.FIXTURES if f.answer["kind"] == "moe_imbalance"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("gate1")
    key = harness.build(out)
    return out, key


def records_of(out: Path, name: str) -> list[dict]:
    """The step records, read with the standard library (no SDK)."""
    path = out / "fixtures" / name / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- The key ---------------------------------------------------------------------------


def test_the_key_holds_the_planted_answers(built):
    _, key = built
    assert set(key["fixtures"]) == set(generate.BY_NAME)
    for fixture in generate.FIXTURES:
        k, a = key["fixtures"][fixture.name], fixture.answer
        assert k["held_out"] == (fixture.name != "seeded")
        assert (k["kind"], k["step"]) == (a["kind"], a["step"])
        if a["kind"] == "moe_imbalance":
            assert k["location"] == {"layer": a["layer"]}
            assert k["overloaded_experts"] == sorted(a["overloaded"])
            assert k["dead_experts"] == sorted(a["dead"])
            planted = [e.hot for e in fixture.events if e.flagged and e.active(a["step"], a["layer"])][0]
            if a["overloaded"]:
                assert k["magnitude"]["rule"] == "share"
                assert k["magnitude"]["value"] == pytest.approx(max(planted[x] for x in a["overloaded"]), abs=1e-4)
            else:
                assert k["magnitude"] == {"rule": "null", "value": None}
        else:
            assert k["location"] == {"name": a.get("parameter") or a.get("module")}
            assert "overloaded_experts" not in k and "dead_experts" not in k
            planted = generate.SPIKE_FACTORS[a["step"]] if a["kind"] == "grad_spike" else 100.0
            assert k["magnitude"]["rule"] == "ratio"
            assert planted / 1.25 <= k["magnitude"]["value"] <= planted * 1.25


def test_the_unseen_fixtures_are_held_out_and_seeded_is_practice():
    assert set(harness.HELD_OUT) | set(harness.PRACTICE) == set(generate.BY_NAME)
    assert not set(harness.HELD_OUT) & set(harness.PRACTICE)


def test_every_planted_kind_is_in_the_prompts_list():
    assert {f.answer["kind"] for f in generate.FIXTURES} <= set(harness.KINDS)
    assert harness.NO_INCIDENT not in harness.KINDS
    assert len(set(harness.KINDS)) == len(harness.KINDS)


# --- What each arm is given ------------------------------------------------------------


def test_every_folder_has_the_same_prompt_and_a_template(built):
    out, _ = built
    prompt = harness.prompt_text()
    for fixture in generate.FIXTURES:
        for arm in harness.ARMS:
            folder = out / "trial" / fixture.name / arm
            assert (folder / "prompt.md").read_text(encoding="utf-8") == prompt
            template = json.loads((folder / "answer.template.json").read_text(encoding="utf-8"))
            assert template == harness.answer_template(fixture.name, arm)


def test_arm_a_gets_the_scalars_csv_and_nothing_else(built):
    out, _ = built
    for fixture in generate.FIXTURES:
        folder = out / "trial" / fixture.name / "A"
        assert {p.name for p in folder.iterdir()} == {"prompt.md", "answer.template.json", "scalars.csv"}
        rows = list(csv.DictReader(io.StringIO((folder / "scalars.csv").read_text(encoding="utf-8"))))
        header = (folder / "scalars.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        # The fixtures log the loss and nothing else through `session.log`.
        assert header == ["step", "loss", "lr", "grad_norm"]
        records = records_of(out, fixture.name)
        if fixture.name in MOE_NAMES:
            assert rows == []  # the MoE fixtures logged no loss, lr or total gradient norm
        else:
            assert [int(r["step"]) for r in rows] == [r["step"] for r in records]
            assert [float(r["grad_norm"]) for r in rows] == [r["grad_norm"] for r in records]
            assert [float(r["loss"]) for r in rows] == [r["scalars"]["loss"] for r in records]


def test_arm_a_cannot_leak_the_answer(built):
    """No incident name, layer, router, expert, parameter or module statistic reaches arm A."""
    out, key = built
    for fixture in generate.FIXTURES:
        # The prompt and the template are the same for every fixture apart from the neutral
        # fixture name and the arm (checked above), so the CSV is the only run-specific file.
        folder = out / "trial" / fixture.name / "A"
        text = (folder / "scalars.csv").read_text(encoding="utf-8")
        for word in (*harness.KINDS, "router", "expert", "layers.", "blocks.", "moe", "overloaded", "dead",
                     "diversity", "abs_max", "zero_frac", "incident", "spike"):
            assert word not in text.lower(), (fixture.name, word)
        k = key["fixtures"][fixture.name]
        if "name" in k["location"]:
            assert k["location"]["name"] not in text


def test_arm_a_does_not_change_when_everything_but_the_scalars_does(built):
    """By construction: scramble every per-layer, per-parameter and per-expert field."""
    out, _ = built
    rng = random.Random(0)
    for fixture in generate.FIXTURES:
        records = records_of(out, fixture.name)
        scrambled = copy.deepcopy(records)
        for record in scrambled:
            record["layers"] = {f"blocks.{rng.randrange(99)}": {"abs_max": rng.random() * 1e6}}
            record["grad_norms"] = {"blocks.0.x.weight": rng.random() * 1e6}
            if "moe" in record:
                record["moe"] = {"layers.01.router": [rng.randrange(100) for _ in range(8)]}
                record["moe_stats"] = {"layers.01.router": {"diversity": rng.random()}}
                record["moe_top_k"] = {"layers.01.router": 3}
        assert harness.scalar_export(scrambled) == harness.scalar_export(records)
        assert harness.scalar_export(records) == (out / "trial" / fixture.name / "A" / "scalars.csv").read_text(encoding="utf-8")


def test_arm_a_shows_the_scalars_it_is_given():
    """The export is not blind: a logged value, the learning rate and the total norm come through."""
    records = [{"step": 3, "lr": [0.1, 0.2], "grad_norm": 2.5, "scalars": {"loss": 1.0, "acc": 0.5}},
               {"step": 4, "lr": None, "grad_norm": None, "scalars": {}}]
    text = harness.scalar_export(records)
    assert text.splitlines() == ["step,loss,lr_0,lr_1,grad_norm,acc", "3,1.0,0.1,0.2,2.5,0.5"]


def test_arm_b_is_a_placeholder_that_names_what_the_app_loads(built):
    out, _ = built
    for fixture in generate.FIXTURES:
        folder = out / "trial" / fixture.name / "B"
        assert {p.name for p in folder.iterdir()} == {"prompt.md", "answer.template.json", "RUNE_APP_PLACEHOLDER.md"}
        text = (folder / "RUNE_APP_PLACEHOLDER.md").read_text(encoding="utf-8")
        assert f"`{fixture.name}.evidence.json`" in text and f"`{fixture.name}.jsonl`" in text
        assert f"`fixtures/{fixture.name}/`" in text
        assert str(out) not in text  # no path into the folder that holds the key


def test_arm_c_gets_the_evidence_and_its_summary(built):
    import tensorquill

    out, _ = built
    for fixture in generate.FIXTURES:
        folder = out / "trial" / fixture.name / "C"
        assert {p.name for p in folder.iterdir()} == {"prompt.md", "answer.template.json", "evidence.json", "summary.txt"}
        source = out / "fixtures" / fixture.name / f"{fixture.name}.evidence.json"
        assert (folder / "evidence.json").read_bytes() == source.read_bytes()
        evidence = json.loads(source.read_text(encoding="utf-8"))
        assert (folder / "summary.txt").read_text(encoding="utf-8") == tensorquill.incident_summary(evidence) + "\n"


def test_arm_r_gets_what_c_gets_and_the_recording(built):
    out, _ = built
    for fixture in generate.FIXTURES:
        folder, c = out / "trial" / fixture.name / "R", out / "trial" / fixture.name / "C"
        assert {p.name for p in folder.iterdir()} == {"prompt.md", "answer.template.json", "evidence.json",
                                                      "summary.txt", "run.rrd"}
        for name in ("evidence.json", "summary.txt"):
            assert (folder / name).read_bytes() == (c / name).read_bytes()
        assert (folder / "run.rrd").read_bytes() == (out / "fixtures" / fixture.name / f"{fixture.name}.rrd").read_bytes()


def test_the_build_can_skip_arm_r(tmp_path):
    harness.build(tmp_path, ["unseen-6"], rerun_arm=False)
    assert sorted(p.name for p in (tmp_path / "trial" / "unseen-6").iterdir()) == ["A", "B", "C"]
    assert not list(tmp_path.rglob("*.rrd"))


def test_nothing_in_a_trial_folder_holds_the_key(built):
    out, _ = built
    for path in (out / "trial").rglob("*"):
        assert path.name not in {"key.json", "answer-key.json"}
    assert not (out / "trial" / "key.json").exists()


# --- Scoring the hand-written answer sets ------------------------------------------------


def test_the_perfect_set_scores_100_percent(built):
    _, key = built
    rows = harness.score(key, harness.load_answers([VALIDATION / "perfect"]))
    assert sorted(r["fixture"] for r in rows) == sorted(harness.HELD_OUT)
    assert all(r["correct"] for r in rows)
    assert sum(r["right"] for r in rows) == sum(r["scored"] for r in rows) == 4 * 6 + 2 * 4
    assert harness.arm_stats(rows)["accuracy"] == 100.0


def test_the_wrong_set_scores_as_expected(built):
    _, key = built
    rows = harness.score(key, harness.load_answers([VALIDATION / "wrong"]))
    for row in rows:
        assert {q for q, m in row["marks"].items() if m is False} == WRONG[row["fixture"]], row["fixture"]
    assert not any(r["correct"] for r in rows)
    assert (sum(r["right"] for r in rows), sum(r["scored"] for r in rows)) == (21, 32)
    assert harness.arm_stats(rows)["accuracy"] == pytest.approx(65.625)


def test_the_command_line_writes_the_results_table(built, tmp_path):
    out, _ = built
    table = tmp_path / "results.md"
    subprocess.run([sys.executable, str(HERE / "harness.py"), "score", str(out / "key.json"), str(VALIDATION),
                    "-o", str(table)], check=True)
    text = table.read_text(encoding="utf-8")
    assert "| C | TensorQuill's evidence.json and its text summary | 6 | 6 | 100.0% | 6 of 6 |" in text
    assert "| A | tracker scalars only" in text and "| 65.6% | 0 of 6 |" in text
    assert "| B | the Rune viewer app with the run's evidence.json | not run |" in text
    assert "| R | stock Rerun 0.43.2" in text and "| not run |" in text
    assert "Arm B: not run." in text


def test_finding_nothing_scores_nothing(built):
    """`none` with empty lists and `null` earns no point, though some keys hold `[]` or `null`."""
    _, key = built
    nothing = dict(kind="none", step=None, location=None, overloaded_experts=[], dead_experts=[], magnitude=None)
    rows = harness.score(key, [(name, answer(fixture=name, **nothing)) for name in harness.HELD_OUT])
    assert (sum(r["right"] for r in rows), sum(r["scored"] for r in rows)) == (0, 32)
    # The same lists with the right kind do score where the key has them.
    marks = harness.score_answer({**nothing, "kind": "moe_imbalance"}, key["fixtures"]["unseen-4"])
    assert marks["overloaded_experts"] and marks["magnitude"] and not marks["dead_experts"]


def test_scoring_needs_no_sdk(built):
    """`score` runs on the standard library: it must not import tensorquill or torch."""
    out, _ = built
    code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import harness, json, pathlib;"
            f"key = json.loads(pathlib.Path({str(out / 'key.json')!r}).read_text());"
            f"harness.results_table(harness.score(key, harness.load_answers([pathlib.Path({str(VALIDATION)!r})])));"
            "assert 'tensorquill' not in sys.modules and 'torch' not in sys.modules, sorted(sys.modules)")
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


# --- Scoring rules -----------------------------------------------------------------------


def answer(fixture="unseen-1", arm="C", trial=1, participant="agent:test", **answers):
    base = {q: None for q in harness.QUESTION_IDS}
    base.update(answers)
    return {"format": harness.ANSWER_FORMAT, "fixture": fixture, "arm": arm, "trial": trial,
            "participant": participant, "started_at": "2026-11-16T10:00:00Z",
            "submitted_at": "2026-11-16T10:01:00Z", "tool_calls": 5, "answers": base, "notes": ""}


@pytest.mark.parametrize("value,expected", [
    (17, {"layer": 17}), ("17", {"layer": 17}), ("layers.17.router", {"layer": 17}), ("Layer 17", {"layer": 17}),
    ("layers.017.router", {"layer": 17}), (17.0, {"layer": 17}), (" blocks.13 ", {"name": "blocks.13"}),
    ("blocks.7.mlp.fc2.weight", {"name": "blocks.7.mlp.fc2.weight"}), (None, None), ("", None), (True, None),
])
def test_locations_are_normalised(value, expected):
    assert harness.normalise_location(value) == expected


def test_a_rule_per_question(built):
    _, key = built
    k1, k2, k4 = key["fixtures"]["unseen-1"], key["fixtures"]["unseen-2"], key["fixtures"]["unseen-4"]
    good1 = dict(kind="moe_imbalance", step=1300, location=17, overloaded_experts=[53],
                 dead_experts=list(range(20, 28)), magnitude=0.44)
    assert all(harness.score_answer(good1, k1).values())
    # A wrong kind also loses questions 4 to 6, which count only with the right kind.
    marks = harness.score_answer({**good1, "kind": "starved_experts"}, k1)
    assert {q for q, m in marks.items() if m is False} == {"kind", "overloaded_experts", "dead_experts", "magnitude"}
    for change, wrong in [({"step": 1200}, "step"), ({"step": True}, "step"),
                          ({"location": 18}, "location"), ({"overloaded_experts": [53, 54]}, "overloaded_experts"),
                          ({"dead_experts": list(range(20, 28)) + [1]}, "dead_experts"),
                          ({"dead_experts": None}, "dead_experts"), ({"overloaded_experts": [True]}, "overloaded_experts"),
                          ({"magnitude": 0.4511}, "magnitude"), ({"magnitude": None}, "magnitude"),
                          ({"magnitude": float("nan")}, "magnitude")]:
        marks = harness.score_answer({**good1, **change}, k1)
        assert {q for q, m in marks.items() if m is False} == {wrong}, change
    # Order and duplicates of expert IDs do not matter; the share tolerance is 0.01 either side.
    assert all(harness.score_answer({**good1, "dead_experts": list(range(27, 19, -1)) + [20], "magnitude": 0.449}, k1).values())
    assert all(harness.score_answer({**good1, "kind": " MoE_Imbalance ", "step": 1300.0}, k1).values())
    # Dense: experts are not scored; the ratio is right within a factor of 1.5.
    good2 = dict(kind="grad_spike", step=310, location="blocks.7.mlp.fc2.weight", overloaded_experts=[3],
                 dead_experts=None, magnitude=80)
    marks = harness.score_answer(good2, k2)
    assert marks["overloaded_experts"] is None and marks["dead_experts"] is None
    assert all(m for m in marks.values() if m is not None)
    assert not harness.score_answer({**good2, "magnitude": k2["magnitude"]["value"] * 1.51}, k2)["magnitude"]
    assert not harness.score_answer({**good2, "magnitude": k2["magnitude"]["value"] / 1.51}, k2)["magnitude"]
    assert not harness.score_answer({**good2, "location": "blocks.7.mlp.fc2"}, k2)["location"]
    # No overloaded expert: the magnitude must be null, and an empty list is the right answer.
    good4 = dict(kind="moe_imbalance", step=2000, location=8, overloaded_experts=[],
                 dead_experts=k4["dead_experts"], magnitude=None)
    assert all(harness.score_answer(good4, k4).values())
    assert not harness.score_answer({**good4, "magnitude": 0.34}, k4)["magnitude"]
    assert not harness.score_answer({**good4, "overloaded_experts": None}, k4)["overloaded_experts"]


@pytest.mark.parametrize("change,message", [
    ({"format": "other"}, "not a"), ({"arm": "D"}, "arm must be"), ({"trial": 0}, "trial must be"),
    ({"trial": True}, "trial must be"), ({"participant": ""}, "participant"), ({"participant": "opus"}, "participant"),
    ({"participant": "agent:"}, "participant"), ({"tool_calls": -1}, "tool_calls"),
    ({"started_at": "2026-11-16T10:00:00"}, "time zone"), ({"submitted_at": "2026-11-16T09:00:00Z"}, "before"),
    ({"answers": {"kind": "none"}}, "exactly the keys"),
])
def test_bad_answer_files_are_refused(change, message, tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({**answer(), **change}), encoding="utf-8")
    with pytest.raises(harness.AnswerError, match=message):
        harness.load_answers([path])


def test_duplicates_and_unknown_fixtures_are_refused(built):
    _, key = built
    with pytest.raises(harness.AnswerError, match="same fixture"):
        harness.score(key, [("a", answer()), ("b", answer())])
    with pytest.raises(harness.AnswerError, match="not in the key"):
        harness.score(key, [("a", answer(fixture="unseen-9"))])
    # Another trial or participant on the same fixture is a separate answer.
    assert harness.score(key, [("h", answer(participant="human:author"))])[0]["agent"] is False
    assert len(harness.score(key, [("a", answer()), ("b", answer(trial=2)), ("c", answer(participant="human:x"))])) == 3


def test_the_practice_fixture_is_listed_but_not_counted(built):
    _, key = built
    rows = harness.score(key, [("p", answer(fixture="seeded", kind="moe_imbalance")), ("u", answer())])
    table = harness.results_table(rows)
    assert "1 answer files on held-out fixtures, 1 on the practice fixture" in table
    assert "| seeded (practice) |" in table
    assert "| C | TensorQuill's evidence.json and its text summary | 1 | 1 |" in table


# --- The gate lines ----------------------------------------------------------------------


def test_the_gate_lines_compare_arms_on_shared_fixtures_only(built):
    _, key = built
    k = key["fixtures"]["unseen-1"]
    right = dict(kind="moe_imbalance", step=1300, location=17, overloaded_experts=[53],
                 dead_experts=k["dead_experts"], magnitude=0.44)
    k6 = key["fixtures"]["unseen-6"]
    right6 = dict(kind="moe_imbalance", step=700, location=3, overloaded_experts=[7, 22], dead_experts=[], magnitude=0.39)
    files = [
        ("a1", {**answer(arm="A", **right), "tool_calls": 20}),
        ("a2", {**answer(arm="A", trial=2, **right), "tool_calls": 30}),
        ("b1", {**answer(arm="B", **right), "tool_calls": 10}),
        # Only arm A answered unseen-6, so it stays out of every B-against-A comparison.
        ("a6", {**answer(fixture="unseen-6", arm="A", **right6), "tool_calls": 1}),
        ("h1", {**answer(arm="A", participant="human:author", **right),
                "started_at": "2026-11-16T10:00:00Z", "submitted_at": "2026-11-16T10:03:00Z"}),
        ("h2", {**answer(arm="B", participant="human:author", trial=2, **right),
                "started_at": "2026-11-16T10:00:00Z", "submitted_at": "2026-11-16T10:02:00Z"}),
    ]
    lines = harness.gate_checks(harness.score(key, files))
    assert "Accuracy, B against A on unseen-1: 100.0% against 100.0% (+0.0 pp): within ±5 pp or higher in B." in lines
    assert "Median agent tool calls to a correct answer, A over B: 25.0 / 10 = 2.50× (meets 2.0×)." in lines
    assert "Median human time to a correct answer, A over B: 180.0 / 120.0 = 1.50× (below 2.0×)." in lines
    assert "Speed line against A (met): arm B must cut either measure by at least 2.0× against arm A." in lines
    assert "B against C: arm C not run." in lines and "B against R: arm R not run." in lines
    # Lower accuracy in B by more than 5 points fails the accuracy line.
    files.append(("b2", {**answer(arm="B", trial=3, **{**right, "step": 1}), "tool_calls": 10}))
    lines = harness.gate_checks(harness.score(key, files))
    assert any(line.startswith("Accuracy, B against A on unseen-1:") and line.endswith("FAILS.") for line in lines)
