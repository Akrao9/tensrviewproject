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
    # The validation sets are human trials in arms C and A, so the amended gate is not checked.
    assert "- C against R: not checked; arms C and R have no agent trials on held-out fixtures." in text


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


# --- The gate as amended Nov 17, 2026: arm C against arm R, agent trials only ------------------


def trials(arm, *, calls=10, participant="agent:model-x", fixtures=harness.HELD_OUT, broken=0, break_field="step",
           name=""):
    """Three trials per fixture in `arm`, each the perfect set's answer, with `broken` items made wrong.

    Each broken item is one file's `step` (or `location`), never `kind`, which would also cost the
    questions that count only with the right kind. Every trial has the same tool-call count, so a
    trial that drops out of the correct set does not move the median.
    """
    files = []
    for trial in (1, 2, 3):
        for fixture in fixtures:
            base = json.loads((VALIDATION / "perfect" / f"{fixture}.json").read_text(encoding="utf-8"))
            base.update(arm=arm, trial=trial, participant=participant, tool_calls=calls)
            if broken:
                base["answers"][break_field] = 1  # no fixture's step or layer is 1
                broken -= 1
            files.append((f"{name or arm}-{fixture}-{trial}", base))
    assert broken == 0, "more broken items than files"
    return files


def c_against_r(key, files):
    lines = harness.evidence_gate_checks(harness.score(key, files))
    return lines, next(line for line in lines if line.startswith("Gate (C against R"))


def part(lines, prefix):
    return next(line for line in lines if line.startswith(prefix))


def test_c_against_r_passes_when_c_is_as_accurate_with_fewer_tool_calls(built):
    _, key = built
    files = trials("C", calls=8) + trials("R", calls=12)
    rows = harness.score(key, files)
    for arm in ("C", "R"):
        assert sum(r["scored"] for r in rows if r["arm"] == arm) == 96
    lines, verdict = c_against_r(key, files)
    names = ", ".join(harness.HELD_OUT)
    assert f"Accuracy, C against R on {names}: 96 of 96 (100.0%) against 96 of 96 (100.0%), +0.0 pp: met " \
           "(C may be at most 5.0 pp behind R)." in lines
    assert "Median agent tool calls to a correct answer, C against R: 8.0 against 12.0 (18 and 18 correct trials): met " \
           "(C must be no more than R)." in lines
    assert verdict == "Gate (C against R, amendment of Nov 17, 2026): PASSES. Both parts must hold."
    assert "Model: model-x in both arms." in lines
    assert any(line.startswith("Time limit per trial: not recorded") for line in lines)
    assert not any("held-out fixtures only" in line for line in lines)  # all six are shared


@pytest.mark.parametrize("behind,passes", [(4, True), (5, False)])
def test_c_may_be_four_items_of_96_behind_r_but_not_five(built, behind, passes):
    _, key = built
    for field in ("step", "location"):
        files = trials("C", broken=behind, break_field=field) + trials("R")
        lines, verdict = c_against_r(key, files)
        accuracy = part(lines, "Accuracy, C against R")
        assert f"{96 - behind} of 96" in accuracy and "96 of 96" in accuracy
        assert accuracy.endswith("met (C may be at most 5.0 pp behind R).") == passes, accuracy
        assert "FAILS" not in part(lines, "Median agent tool calls")  # equal medians: only accuracy decides
        assert verdict.endswith("PASSES. Both parts must hold." if passes else "FAILS. Both parts must hold.")
    # 4 of 96 is 4.2 points and 5 of 96 is 5.2, as the amendment says.
    assert f"-{100 * behind / 96:.1f} pp" in accuracy


def test_c_against_r_fails_on_tool_calls_alone(built):
    _, key = built
    lines, verdict = c_against_r(key, trials("C", calls=13) + trials("R", calls=12))
    assert part(lines, "Accuracy, C against R").endswith("met (C may be at most 5.0 pp behind R).")
    assert part(lines, "Median agent tool calls") == ("Median agent tool calls to a correct answer, C against R: "
                                                      "13.0 against 12.0 (18 and 18 correct trials): FAILS "
                                                      "(C must be no more than R).")
    assert "FAILS" in verdict


def test_equal_tool_call_medians_pass(built):
    _, key = built
    files = trials("C", calls=10) + trials("R", calls=10)
    # Medians, not means: C's 9, 10, 30 and R's 10, 10, 10 per fixture both have median 10.
    for i, (_, answer_file) in enumerate(files[:18]):
        answer_file["tool_calls"] = (9, 10, 30)[i // 6]
    lines, verdict = c_against_r(key, files)
    assert "10.0 against 10.0" in part(lines, "Median agent tool calls") and verdict.endswith("PASSES. Both parts must hold.")


def test_no_correct_trial_in_c_fails_the_tool_call_part(built):
    _, key = built
    # Every C trial has a wrong step; to keep the accuracy part out of it, R loses as many items.
    files = trials("C", broken=18) + trials("R", broken=18, break_field="location", calls=1)
    lines, verdict = c_against_r(key, files)
    assert part(lines, "Accuracy, C against R").endswith("met (C may be at most 5.0 pp behind R).")
    assert part(lines, "Median agent tool calls") == \
        "Median agent tool calls to a correct answer: FAILS; arm C has no correct trial."
    assert "FAILS" in verdict
    # Neither arm with a correct trial fails too.
    lines, verdict = c_against_r(key, trials("C", broken=18) + trials("R", broken=18))
    assert part(lines, "Median agent tool calls").endswith("arm C has no correct trial.") and "FAILS" in verdict


def test_no_correct_trial_in_r_and_one_in_c_meets_the_tool_call_part(built):
    _, key = built
    files = trials("C", calls=50) + trials("R", broken=18, calls=1)
    lines, verdict = c_against_r(key, files)
    assert part(lines, "Median agent tool calls") == \
        "Median agent tool calls to a correct answer: met; arm C has 18 correct trials (median 50.0), and arm R " \
        "has no correct trial."
    assert verdict.endswith("PASSES. Both parts must hold.")
    # Met with no condition on counts: C's correct trials need none when R has no correct trial.
    lines, verdict = c_against_r(key, trials("C", calls=None) + trials("R", broken=18, calls=1))
    assert part(lines, "Median agent tool calls") == "Median agent tool calls to a correct answer: met; arm C has " \
        "18 correct trials (median –), and arm R has no correct trial."
    assert verdict.endswith("PASSES. Both parts must hold.")


def test_c_against_r_counts_only_the_fixtures_both_arms_answered(built):
    _, key = built
    # C alone answered unseen-1 and unseen-2, every trial wrong and slow; R alone answered unseen-5 and
    # unseen-6. Counted, C's unseen-1 and unseen-2 trials would fail both parts.
    files = (trials("C", fixtures=["unseen-1", "unseen-2"], broken=6, calls=99, name="C-only")
             + trials("C", fixtures=["unseen-3", "unseen-4"], calls=5)
             + trials("R", fixtures=["unseen-3", "unseen-4", "unseen-5", "unseen-6"], calls=5))
    lines, verdict = c_against_r(key, files)
    assert lines[0] == ("C against R on 2 of the 6 held-out fixtures only (unseen-3, unseen-4); "
                        "the others lack agent trials in one arm or both.")
    assert part(lines, "Provisional: Accuracy, C against R") == (
        "Provisional: Accuracy, C against R on unseen-3, unseen-4: 36 of 36 (100.0%) against 36 of 36 (100.0%), "
        "+0.0 pp: met (C may be at most 5.0 pp behind R).")
    assert "5.0 against 5.0 (6 and 6 correct trials)" in part(lines, "Provisional: Median agent tool calls")
    # Lane U8 (audit X01): this case read PASSES until Nov 19, 2026. Two of six fixtures are not the
    # preregistered design, so the arithmetic is shown and the verdict is not given.
    assert "PASSES" not in verdict and "not decided; the preregistered design needs" in verdict
    assert ("arm R trials 1, 2 and 3 on each held-out fixture (unseen-1 has none; unseen-2 has none)" in verdict)


def test_c_against_r_ignores_human_trials_and_the_practice_fixture(built):
    _, key = built
    files = trials("C", calls=8) + trials("R", calls=12)
    # Counted, the wrong ones would put C 18 items behind R, and the right ones, with 99 calls each,
    # would lift C's median above R's 12.
    files += [(f"human-{name}", {**body, "participant": "human:author", "tool_calls": None})
              for name, body in trials("C", broken=18, name="h")]
    files += [(f"human-right-{name}", {**body, "participant": "human:someone", "trial": body["trial"] + 3})
              for name, body in trials("C", calls=99, name="hr")]
    files += [(f"human-right2-{name}", {**body, "participant": "human:someone", "trial": body["trial"] + 6})
              for name, body in trials("C", calls=99, name="hr2")]
    files += [("practice", {**answer(fixture="seeded", arm="C", kind="none"), "tool_calls": 500})]
    lines, verdict = c_against_r(key, files)
    assert "96 of 96 (100.0%) against 96 of 96" in part(lines, "Accuracy, C against R")
    assert "8.0 against 12.0" in part(lines, "Median agent tool calls")
    assert verdict.endswith("PASSES. Both parts must hold.")
    assert "Model: model-x in both arms." in lines  # the human's name is not a model
    # With only human trials in one arm, the gate is not checked at all.
    only_humans = [(n, {**b, "participant": "human:author"}) for n, b in trials("R")]
    assert harness.evidence_gate_checks(harness.score(key, trials("C") + only_humans)) == \
        ["C against R: not checked; arm R has no agent trials on held-out fixtures."]


def test_a_correct_trial_without_a_tool_call_count_leaves_the_part_undecided(built):
    _, key = built
    files = trials("C", calls=8) + trials("R", calls=None)
    lines, verdict = c_against_r(key, files)
    assert part(lines, "Median agent tool calls") == ("Median agent tool calls to a correct answer: cannot be judged; "
                                                      "correct agent trials without a tool-call count: 18 in arm R.")
    assert verdict == ("Gate (C against R, amendment of Nov 17, 2026): not decided, since the tool-call part "
                       "cannot be judged. Both parts must hold.")


def test_the_results_table_says_when_the_models_differ(built):
    _, key = built
    files = trials("C", participant="agent:model-x") + trials("R", participant="agent:model-y")
    lines = harness.evidence_gate_checks(harness.score(key, files))
    assert ("Model: the arms' answer files differ or mix models (C: model-x; R: model-y). The amendment requires "
            "the same model in every arm, so no verdict is given.") in lines
    # Lane U8 (audit X01): different models read PASSES until Nov 19, 2026.
    verdict = part(lines, "Gate (C against R")
    assert verdict == ("Gate (C against R, amendment of Nov 17, 2026): not decided; the preregistered design needs "
                       "one model in both arms (the answer files name model-x, model-y). Both parts must hold.")
    table = harness.results_table(harness.score(key, files))
    assert "## Gate 1: arm C against arm R (amendment of Nov 17, 2026; agent trials only)" in table
    assert "- Model: the arms' answer files differ" in table
    assert "PASSES" not in table


# --- Lane U8 (audit X01): a release verdict only for the preregistered design ------------------
#
# Six held-out fixtures, three trials each (numbered 1 to 3), one model in every arm. Anything
# less still shows the two parts' arithmetic, marked provisional, and the verdict reads "not
# decided" with what is missing. Every case below read PASSES before Nov 19, 2026.


def whole_table(key, files):
    return harness.results_table(harness.score(key, files))


def test_one_perfect_paired_trial_on_one_fixture_is_not_a_verdict(built):
    """The audit's case: one perfect agent answer per arm on unseen-1."""
    _, key = built
    files = [(f"{arm}-1", answer_file) for arm, answer_file in
             ((arm, {**json.loads((VALIDATION / "perfect" / "unseen-1.json").read_text(encoding="utf-8")),
                     "arm": arm, "trial": 1, "participant": "agent:model-x", "tool_calls": calls})
              for arm, calls in (("C", 3), ("R", 5)))]
    lines, verdict = c_against_r(key, files)
    assert lines[1] == ("The answer files are not the preregistered design, so the two parts below are provisional "
                        "arithmetic and no release verdict is given.")
    assert part(lines, "Provisional: Accuracy, C against R on unseen-1: 6 of 6 (100.0%) against 6 of 6")
    assert part(lines, "Provisional: Median agent tool calls").endswith("3 against 5 (1 and 1 correct trials): met "
                                                                        "(C must be no more than R).")
    missing = ("unseen-1 has trials 1; unseen-2 has none; unseen-3 has none; unseen-4 has none; unseen-5 has none; "
               "unseen-6 has none")
    assert verdict == ("Gate (C against R, amendment of Nov 17, 2026): not decided; the preregistered design needs "
                       f"arm C trials 1, 2 and 3 on each held-out fixture ({missing}); and arm R trials 1, 2 and 3 on "
                       f"each held-out fixture ({missing}). Both parts must hold.")
    assert "PASSES" not in whole_table(key, files)


def test_one_fixture_with_all_three_trials_is_not_a_verdict(built):
    _, key = built
    files = trials("C", fixtures=["unseen-1"], calls=3) + trials("R", fixtures=["unseen-1"], calls=5)
    lines, verdict = c_against_r(key, files)
    assert "not decided" in verdict and "unseen-1 has" not in verdict and "unseen-2 has none" in verdict
    assert "PASSES" not in whole_table(key, files)


@pytest.mark.parametrize("arm", ["C", "R"])
def test_six_fixtures_with_a_repetition_missing_is_not_a_verdict(built, arm):
    _, key = built
    files = trials("C", calls=8) + trials("R", calls=12)
    files = [(n, b) for n, b in files if not (b["arm"] == arm and b["fixture"] == "unseen-4" and b["trial"] == 3)]
    lines, verdict = c_against_r(key, files)
    assert verdict == ("Gate (C against R, amendment of Nov 17, 2026): not decided; the preregistered design needs "
                       f"arm {arm} trials 1, 2 and 3 on each held-out fixture (unseen-4 has trials 1, 2). "
                       "Both parts must hold.")
    assert not any("held-out fixtures only" in line for line in lines)  # all six are still shared
    assert part(lines, "Provisional: Accuracy, C against R")
    assert "PASSES" not in whole_table(key, files)


def test_an_extra_trial_is_not_the_preregistered_design(built):
    _, key = built
    extra = [(f"C-fourth-{n}", {**b, "trial": 4}) for n, b in trials("C", fixtures=["unseen-2"], calls=8)][:1]
    files = trials("C", calls=8) + trials("R", calls=12) + extra
    _, verdict = c_against_r(key, files)
    assert "arm C trials 1, 2 and 3 on each held-out fixture (unseen-2 has trials 1, 2, 3, 4)" in verdict
    assert "PASSES" not in verdict


def test_two_models_within_one_arm_is_not_a_verdict(built):
    _, key = built
    files = trials("C", calls=8) + trials("R", calls=12)
    files = [(n, {**b, "participant": "agent:model-z"}) if b["arm"] == "R" and b["fixture"] == "unseen-6" else (n, b)
             for n, b in files]
    lines, verdict = c_against_r(key, files)
    assert "one model in both arms (the answer files name model-x, model-z)" in verdict
    assert "PASSES" not in whole_table(key, files)


def test_the_complete_design_gets_its_verdict_and_no_provisional_mark(built):
    _, key = built
    for c_calls, expected in ((8, "PASSES"), (13, "FAILS")):
        files = trials("C", calls=c_calls) + trials("R", calls=12)
        lines, verdict = c_against_r(key, files)
        assert verdict == f"Gate (C against R, amendment of Nov 17, 2026): {expected}. Both parts must hold."
        assert not any(line.startswith("Provisional") or "preregistered design" in line for line in lines)
    assert harness.design_gaps(*[[r for r in harness.score(key, files) if r["arm"] == arm] for arm in ("C", "R")]) == []


# --- The speed line's ratio with a zero denominator (amendment of Nov 20, 2026) ------------------
# Hand-built score rows, so none of these needs the fixture build. Arm R is compared with arm B,
# which is one of the pairs the plan's speed line is checked on.


def row(arm, *, calls=None, seconds=None, agent=True, correct=True, trial=1, fixture="unseen-1"):
    """One row as `harness.score` makes it: a held-out trial, right on every question when `correct`."""
    marks = {q: True for q in harness.QUESTION_IDS}
    if not correct:
        marks["step"] = False
    return {"file": f"{arm}-{fixture}-{trial}", "fixture": fixture, "arm": arm, "trial": trial,
            "participant": "agent:model-x" if agent else "human:author", "held_out": True, "agent": agent,
            "seconds": seconds, "tool_calls": calls, "marks": marks,
            "right": sum(marks.values()), "scored": len(marks), "correct": correct}


CALLS = "Median agent tool calls to a correct answer"
SECONDS = "Median human time to a correct answer"


def speed(rows):
    """The speed lines of B against R: each measure's line, and the verdict line or None."""
    lines = harness.gate_checks(rows)
    calls = next(line for line in lines if line.startswith(CALLS))
    seconds = next(line for line in lines if line.startswith(SECONDS))
    verdict = next((line for line in lines if line.startswith("Speed line against R")), None)
    return calls, seconds, verdict


def test_zero_tool_calls_over_zero_is_not_measured():
    calls, _, verdict = speed([row("B", calls=0), row("R", calls=0)])
    assert calls == (f"{CALLS}, R over B: 0 / 0: not measured (arm B's median is zero, so the ratio is "
                     f"undefined; it counts neither for nor against the speed line).")
    # Nothing else is measured, so the speed line has no verdict (as when no measure has correct trials).
    assert verdict is None


def test_a_positive_count_over_zero_is_not_measured_and_does_not_meet_the_line():
    calls, _, verdict = speed([row("B", calls=0), row("R", calls=7)])
    assert calls.startswith(f"{CALLS}, R over B: 7 / 0: not measured")
    assert "meets" not in calls and "inf" not in calls
    assert verdict is None


def test_zero_over_a_positive_count_is_an_ordinary_ratio_below_the_line():
    calls, _, verdict = speed([row("B", calls=5), row("R", calls=0)])
    assert calls == f"{CALLS}, R over B: 0 / 5 = 0.00× (below 2.0×)."
    assert verdict.startswith("Speed line against R (not met)")


@pytest.mark.parametrize("r_calls,ratio,meets", [(20, "2.00", True), (15, "1.50", False)])
def test_an_ordinary_ratio_meets_the_line_at_two_and_not_below(r_calls, ratio, meets):
    calls, _, verdict = speed([row("B", calls=10), row("R", calls=r_calls)])
    assert calls == f"{CALLS}, R over B: {r_calls} / 10 = {ratio}× ({'meets' if meets else 'below'} 2.0×)."
    assert verdict.startswith(f"Speed line against R ({'met' if meets else 'not met'})")


def test_a_missing_count_beside_an_observed_zero():
    # A count of null is left out of the median and is not read as zero; the observed zero is kept.
    calls, _, _ = speed([row("B", calls=None), row("B", calls=0, trial=2), row("R", calls=6)])
    assert calls.startswith(f"{CALLS}, R over B: 6 / 0: not measured (arm B's median is zero")
    # With only a missing count, B has no median at all: the line says so, and it is not the zero's line.
    calls, _, _ = speed([row("B", calls=None), row("R", calls=6)])
    assert calls == f"{CALLS}, B against R: not measured (no correct trials in both arms)."
    # The missing count is not turned into a zero: B's median over [null, 0, 4] is 2, not 0.
    calls, _, _ = speed([row("B", calls=None), row("B", calls=0, trial=2), row("B", calls=4, trial=3),
                         row("R", calls=6)])
    assert calls == f"{CALLS}, R over B: 6 / 2.0 = 3.00× (meets 2.0×)."
    # Every file's own count is shown as it is: 0 as 0 and null as a dash.
    table = harness.results_table([row("B", calls=None), row("B", calls=0, trial=2), row("R", calls=6)])
    assert "| unseen-1 | B | 1 | agent:model-x | 6 / 6 | yes | – | – | – |" in table
    assert "| unseen-1 | B | 2 | agent:model-x | 6 / 6 | yes | – | 0 | – |" in table
    assert "R over B: 6 / 0: not measured" in table


def test_a_zero_human_seconds_denominator_is_not_measured():
    _, seconds, verdict = speed([row("B", seconds=0.0, agent=False), row("R", seconds=90.0, agent=False)])
    assert seconds.startswith(f"{SECONDS}, R over B: 90.0 / 0.0: not measured")
    assert verdict is None


@pytest.mark.parametrize("r_seconds,met", [(180.0, True), (90.0, False)])
def test_the_speed_line_is_judged_on_the_measured_measure_alone(r_seconds, met):
    # Tool calls not measured (B's median is zero); human seconds measured, 3.0× or 1.5×.
    rows = [row("B", calls=0), row("R", calls=9),
            row("B", seconds=60.0, agent=False, trial=2), row("R", seconds=r_seconds, agent=False, trial=2)]
    calls, seconds, verdict = speed(rows)
    assert calls.startswith(f"{CALLS}, R over B: 9 / 0: not measured")
    assert seconds.endswith(f"({'meets' if met else 'below'} 2.0×).")
    assert verdict == (f"Speed line against R ({'met' if met else 'not met'}): arm B must cut either measure "
                       f"by at least 2.0× against arm R.")
