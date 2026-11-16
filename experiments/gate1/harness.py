"""Gate 1 trial harness: build each fixture's material per arm, and score the answers.

It calls no model and no service. A person, or a coding agent the author runs, reads a trial
folder and writes an answer file; this script only prepares the folders and scores the files.

    python experiments/gate1/harness.py build [OUT]            # needs the SDK (and torch)
    python experiments/gate1/harness.py score OUT/key.json ANSWERS... [-o results.md]

`build` runs every fixture through `experiments/moe-bakeoff/generate.py` (the SDK's own
pipeline) into `OUT/fixtures/`, writes the scorer's key to `OUT/key.json`, and one folder per
fixture and arm under `OUT/trial/<fixture>/<arm>/`. `score` reads answer files (or folders of
them), checks their format, scores each against the key and writes a Markdown results table.
Scoring uses the standard library only. The rules are pre-registered in PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parent / "moe-bakeoff"
ANSWER_FORMAT = "tensorquill-gate1-answer/1"
KEY_FORMAT = "tensorquill-gate1-key/1"

# The visible practice fixture; the other six are held out and are the only ones the gate counts.
PRACTICE = ("seeded",)
HELD_OUT = ("unseen-1", "unseen-2", "unseen-3", "unseen-4", "unseen-5", "unseen-6")

ARMS = {
    "A": "tracker scalars only (loss, learning rate, total gradient norm, as MLflow or W&B hold them)",
    "B": "the Rune viewer app with the run's evidence.json",
    "C": "TensorQuill's evidence.json and its text summary",
    # Not one of the three arms this harness was asked for: plan Step 8's own arm A. Optional.
    "R": "stock Rerun 0.43.2 with the bake-off blueprint, evidence.json and its summary (plan Step 8's arm A)",
}

# Frozen on Nov 16, 2026 from the SDK's 32 incident kinds (every kind `build_evidence`, the server
# and the watcher can report). The list is the same for every fixture and arm, so it says nothing
# about any one run.
KINDS = (
    "activation_drift", "activation_spike", "allocator_retries", "attention_collapse",
    "attention_logit_growth", "attention_uniform", "collective_desync", "collective_hang", "data_stalled",
    "dead_activations", "duplicate_batches", "epoch_order_repeated", "gpu_throttled", "grad_spike",
    "hardware_errors", "input_starvation", "job_stalled", "loss_scale_trouble", "memory_growth",
    "memory_imbalance", "memory_pressure", "moe_imbalance", "non_finite", "rank_silent",
    "repeated_steps", "replica_drift", "resume_data_mismatch", "resume_incomplete",
    "resume_lr_change", "stalled_weights", "starved_experts", "traincheck_violation",
)
NO_INCIDENT = "none"

QUESTIONS = (
    ("kind", "Which kind of failure is the run's first incident? Answer with one name from the list "
             f"below, or `{NO_INCIDENT}` if you find no failure."),
    ("step", "At which training step does that failure first appear in the recorded data? An integer. "
             "If the run records only some steps, give the first recorded step that shows it."),
    ("location", "Where is it? For a router (mixture-of-experts) failure, the layer number as the run "
                 "numbers its routers (`layers.17.router` is layer 17). For a gradient failure, the full "
                 "parameter name. For an activation failure, the module name."),
    ("overloaded_experts", "For a router failure: the IDs of the experts on that layer that receive "
                           "more than their share at that step, as the data numbers them; `[]` if none. "
                           "For any other failure: `[]`."),
    ("dead_experts", "For a router failure: the IDs of the experts on that layer that receive no tokens "
                     "at that step; `[]` if none. For any other failure: `[]`."),
    ("magnitude", "How large is it at that step? For a router failure: the largest share of the "
                  "router's assignments held by one of the experts you named as overloaded (that "
                  "expert's count divided by the sum of all experts' counts on the layer, from 0 to 1), "
                  "or `null` if you named none. For a gradient or activation failure: the named "
                  "parameter's gradient norm, or the module's reading, divided by its typical value "
                  "before the failure (a ratio, such as 12.5)."),
)
QUESTION_IDS = tuple(q for q, _ in QUESTIONS)

# Tolerances for `magnitude`, pre-registered.
SHARE_TOLERANCE = 0.01  # absolute, on a share of assignments
RATIO_FACTOR = 1.5  # the answer must lie within [key / 1.5, key * 1.5]
BASELINE_STEPS = 50  # the "typical value" is the median of this many recorded steps before the onset

GATE_ACCURACY_PP = 5.0
GATE_SPEEDUP = 2.0


# --- What each arm is given ------------------------------------------------------------


def scalar_export(records: list[dict]) -> str:
    """Arm A: the run's scalars as a tracker holds them, as CSV text.

    Only a step's `scalars` (what `session.log` recorded, the loss among them), its learning
    rate and its total gradient norm are read. Nothing per layer, per parameter, per module or
    per expert is, and no detector runs. Rows are the steps with at least one value; a run that
    logged none gives the header alone.
    """
    rows: list[dict[str, float]] = []
    names: set[str] = set()
    groups = 0
    for record in records:
        row: dict[str, float] = {}
        lr = record.get("lr")
        if isinstance(lr, (int, float)):
            lr = [lr]
        if lr:
            groups = max(groups, len(lr))
            for i, value in enumerate(lr):
                if value is not None:
                    row[f"lr_{i}"] = float(value)
        if record.get("grad_norm") is not None:
            row["grad_norm"] = float(record["grad_norm"])
        for name, value in (record.get("scalars") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[str(name)] = float(value)
                names.add(str(name))
        if row:
            row["step"] = record["step"]
            rows.append(row)
    if groups <= 1:
        for row in rows:
            if "lr_0" in row:
                row["lr"] = row.pop("lr_0")
        lr_columns = ["lr"]
    else:
        lr_columns = [f"lr_{i}" for i in range(groups)]
    # The loss, learning rate and gradient norm columns are always there, empty when not logged.
    columns = ["step", "loss", *lr_columns, "grad_norm", *sorted(names - {"loss"})]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: r["step"]):
        writer.writerow({c: ("" if row.get(c) is None else repr(row[c]) if isinstance(row[c], float) else row[c])
                         for c in columns})
    return out.getvalue()


def prompt_text() -> str:
    """The prompt: byte-identical for every fixture and every arm."""
    lines = [
        "# Diagnose this training run",
        "",
        "A training run went wrong, or may have. The files in this folder (and, if you were given one,",
        "the viewer app) are everything you have about it. Answer the six questions below and write",
        "your answers into `answer.json`, starting from `answer.template.json`: fill in `answers`,",
        "keep the other fields as the person running the trial tells you. Use JSON values exactly as",
        "asked: a string, an integer, a list of integers, a number, or `null`.",
        "",
        "## Questions",
        "",
    ]
    for i, (qid, text) in enumerate(QUESTIONS, 1):
        lines += [f"{i}. `{qid}`: {text}", ""]
    lines += ["## Failure kinds", "", ", ".join(f"`{k}`" for k in KINDS) + f", or `{NO_INCIDENT}`.", ""]
    return "\n".join(lines)


def answer_template(fixture: str, arm: str) -> dict:
    return {
        "format": ANSWER_FORMAT,
        "fixture": fixture,
        "arm": arm,
        "trial": None,
        "participant": None,
        "started_at": None,
        "submitted_at": None,
        "tool_calls": None,
        "answers": {qid: None for qid in QUESTION_IDS},
        "notes": "",
    }


ARM_B_PLACEHOLDER = """\
# Arm B: the Rune viewer app (placeholder)

This slot is not runnable yet. Do not give it to a participant until the Rune app can do
what is listed here; until then the scorer reports arm B as "not run".

The participant gets the Rune app, started on this run, and this folder's `prompt.md` and
`answer.template.json`. The app must load at run time (not baked in at build time):

- `{evidence}`: this run's evidence (the same file arm C gets). For a router run it holds the
  per-step, per-layer expert counts the heatmap and the expert load table are drawn from.
- `{records}`: this run's step records. The evidence of a dense run has no per-step samples,
  so the app needs these for its dense view: per-parameter gradient norms (`grad_norms`) and
  per-module activation statistics (`layers`).

Both files are in the operator's build folder, under `fixtures/{name}/`. Neither is copied
here, so nothing in this folder can be read in place of the app; that build folder also holds
the scorer's key, so never hand it to a participant. The app opens at the first incident sample, with the
detector summary and the status heatmap visible, before the timer starts.
"""


def build(out: Path, names: list[str] | None = None, *, rerun_arm: bool = True) -> dict:
    """Generate the fixtures, write the key, and write every trial folder. Needs the SDK.

    `rerun_arm=False` skips arm R (and rerun-sdk, which only that arm needs).
    """
    sys.path.insert(0, str(BAKEOFF))
    import generate  # noqa: E402  (experiments/moe-bakeoff/generate.py; imports tensorquill)
    import tensorquill  # noqa: E402

    chosen = [generate.BY_NAME[n] for n in names] if names else list(generate.FIXTURES)
    keys = {}
    for fixture in chosen:
        source = out / "fixtures" / fixture.name
        evidence = generate.export_fixture(fixture, source, rerun=rerun_arm)
        records = tensorquill.load_records(source / f"{fixture.name}.jsonl")
        keys[fixture.name] = fixture_key(fixture.name, fixture.answer, records)
        evidence_text = (source / f"{fixture.name}.evidence.json").read_text(encoding="utf-8")
        summary = tensorquill.incident_summary(evidence) + "\n"
        for arm in ARMS:
            folder = out / "trial" / fixture.name / arm
            if arm == "R" and not rerun_arm:
                continue
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            (folder / "prompt.md").write_text(prompt_text(), encoding="utf-8")
            (folder / "answer.template.json").write_text(
                json.dumps(answer_template(fixture.name, arm), indent=2) + "\n", encoding="utf-8")
            if arm == "A":
                (folder / "scalars.csv").write_text(scalar_export(records), encoding="utf-8")
            elif arm == "B":
                (folder / "RUNE_APP_PLACEHOLDER.md").write_text(ARM_B_PLACEHOLDER.format(
                    name=fixture.name, evidence=f"{fixture.name}.evidence.json",
                    records=f"{fixture.name}.jsonl"), encoding="utf-8")
            else:
                (folder / "evidence.json").write_text(evidence_text, encoding="utf-8")
                (folder / "summary.txt").write_text(summary, encoding="utf-8")
                if arm == "R":
                    shutil.copyfile(source / f"{fixture.name}.rrd", folder / "run.rrd")
    key = {"format": KEY_FORMAT, "fixtures": keys}
    (out / "key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return key


# --- The key ---------------------------------------------------------------------------


def _router_number(router: str) -> int:
    match = re.fullmatch(r"layers\.0*(\d+)\.router", router)
    if not match:
        raise ValueError(f"not a fixture router name: {router}")
    return int(match.group(1))


def fixture_key(name: str, answer: dict, records: list[dict]) -> dict:
    """The six answers for one fixture, from its planted answer and its own records.

    Kind, step, location and experts are the planted values (`Fixture.answer` in generate.py).
    The magnitude is read from the records at the planted step, never from the detectors.
    """
    kind, step = answer["kind"], answer["step"]
    by_step = {r["step"]: r for r in records}
    key: dict = {"held_out": name in HELD_OUT, "kind": kind, "step": step}
    if kind == "moe_imbalance":
        layer = answer["layer"]
        key["location"] = {"layer": layer}
        key["overloaded_experts"] = sorted(answer["overloaded"])
        key["dead_experts"] = sorted(answer["dead"])
        counts = next(c for router, c in by_step[step]["moe"].items() if _router_number(router) == layer)
        total = sum(counts)
        key["magnitude"] = (
            {"rule": "share", "value": max(counts[e - 1] for e in answer["overloaded"]) / total,
             "tolerance": SHARE_TOLERANCE}
            if answer["overloaded"] else {"rule": "null", "value": None}
        )
    elif kind == "grad_spike":
        param = answer["parameter"]
        key["location"] = {"name": param}
        onset = answer.get("rising_since", step)
        before = [r["grad_norms"][param] for r in records if r["step"] < onset][-BASELINE_STEPS:]
        key["magnitude"] = {"rule": "ratio", "value": by_step[step]["grad_norms"][param] / statistics.median(before),
                            "factor": RATIO_FACTOR}
    elif kind == "activation_spike":
        module, stat = answer["module"], answer["stat"]
        key["location"] = {"name": module}
        before = [r["layers"][module][stat] for r in records if r["step"] < step][-BASELINE_STEPS:]
        key["magnitude"] = {"rule": "ratio", "value": by_step[step]["layers"][module][stat] / statistics.median(before),
                            "factor": RATIO_FACTOR}
    else:
        raise ValueError(f"{name}: no key rule for kind {kind}")
    return key


# --- Answers and scoring ---------------------------------------------------------------


class AnswerError(ValueError):
    pass


def _time(value, field: str, where: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnswerError(f"{where}: {field} must be an ISO 8601 string or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnswerError(f"{where}: {field}: {error}") from None
    if parsed.tzinfo is None:
        raise AnswerError(f"{where}: {field} needs a time zone (for example a trailing Z)")
    return parsed


def check_answer(answer: dict, where: str) -> None:
    """Raise AnswerError unless `answer` is in the fixed answer format."""
    if not isinstance(answer, dict) or answer.get("format") != ANSWER_FORMAT:
        raise AnswerError(f"{where}: not a {ANSWER_FORMAT} answer")
    if not isinstance(answer.get("fixture"), str):
        raise AnswerError(f"{where}: fixture must be a string")
    if answer.get("arm") not in ARMS:
        raise AnswerError(f"{where}: arm must be one of {', '.join(ARMS)}")
    trial = answer.get("trial")
    if not isinstance(trial, int) or isinstance(trial, bool) or trial < 1:
        raise AnswerError(f"{where}: trial must be a positive integer")
    if not isinstance(answer.get("participant"), str) or not re.fullmatch(r"(agent|human):\S.*", answer["participant"]):
        raise AnswerError(f"{where}: participant must be `agent:<model>` or `human:<name>`")
    calls = answer.get("tool_calls")
    if calls is not None and (not isinstance(calls, int) or isinstance(calls, bool) or calls < 0):
        raise AnswerError(f"{where}: tool_calls must be a non-negative integer or null")
    start = _time(answer.get("started_at"), "started_at", where)
    end = _time(answer.get("submitted_at"), "submitted_at", where)
    if start and end and end < start:
        raise AnswerError(f"{where}: submitted_at is before started_at")
    answers = answer.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(QUESTION_IDS):
        raise AnswerError(f"{where}: answers must have exactly the keys {', '.join(QUESTION_IDS)}")


def load_answers(paths: list[Path]) -> list[tuple[str, dict]]:
    """Read answer files; a folder is searched for `*.json` files (templates are skipped)."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*.json") if not p.name.endswith(".template.json"))
        else:
            files.append(path)
    out = []
    for file in files:
        answer = json.loads(file.read_text(encoding="utf-8"))
        check_answer(answer, str(file))
        out.append((str(file), answer))
    return out


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalise_location(value) -> dict | None:
    """`17`, `"17"`, `"layer 17"`, `"layers.17.router"` are layer 17; any other string is a name."""
    if _is_int(value):
        return {"layer": value}
    if isinstance(value, float) and value.is_integer():
        return {"layer": int(value)}
    if isinstance(value, str) and value.strip():
        match = re.fullmatch(r"(?:layers?[ ._]?)?0*(\d+)(?:\.router)?", value.strip(), re.IGNORECASE)
        if match:
            return {"layer": int(match.group(1))}
        return {"name": value.strip()}
    return None


def _expert_set(value) -> set[int] | None:
    if not isinstance(value, list) or not all(_is_int(v) for v in value):
        return None
    return set(value)


def score_answer(answers: dict, key: dict) -> dict[str, bool | None]:
    """{question: True/False, or None when the question is not scored on this fixture}.

    Questions 4 to 6 count only when the kind is right, so "no failure found" (`none`, `[]`,
    `[]`, `null`) earns nothing on a fixture whose key happens to have an empty list or `null`.
    """
    out: dict[str, bool | None] = {}
    kind = answers.get("kind")
    out["kind"] = isinstance(kind, str) and kind.strip().lower() == key["kind"]
    step = answers.get("step")
    out["step"] = (_is_int(step) or (isinstance(step, float) and step.is_integer())) and int(step) == key["step"]
    out["location"] = normalise_location(answers.get("location")) == key["location"]
    for qid in ("overloaded_experts", "dead_experts"):
        if qid in key:
            out[qid] = _expert_set(answers.get(qid)) == set(key[qid])
        else:
            out[qid] = None  # not scored on a dense fixture
    magnitude, rule = answers.get("magnitude"), key["magnitude"]
    if rule["rule"] == "null":
        out["magnitude"] = magnitude is None
    elif rule["rule"] == "share":
        out["magnitude"] = _is_number(magnitude) and abs(magnitude - rule["value"]) <= rule["tolerance"] + 1e-12
    else:
        out["magnitude"] = (_is_number(magnitude) and magnitude > 0
                            and rule["value"] / rule["factor"] <= magnitude <= rule["value"] * rule["factor"])
    if not out["kind"]:
        for qid in ("overloaded_experts", "dead_experts", "magnitude"):
            if out[qid] is not None:
                out[qid] = False
    return out


def score(key: dict, answers: list[tuple[str, dict]]) -> list[dict]:
    """One row per answer file: its fields, per-question marks, and whether it is all correct."""
    if key.get("format") != KEY_FORMAT:
        raise ValueError(f"not a {KEY_FORMAT} key")
    rows, seen = [], {}
    for where, answer in answers:
        name = answer["fixture"]
        if name not in key["fixtures"]:
            raise AnswerError(f"{where}: fixture {name!r} is not in the key")
        ident = (name, answer["arm"], answer["trial"], answer["participant"])
        if ident in seen:
            raise AnswerError(f"{where}: same fixture, arm, trial and participant as {seen[ident]}")
        seen[ident] = where
        marks = score_answer(answer["answers"], key["fixtures"][name])
        scored = [m for m in marks.values() if m is not None]
        start = _time(answer.get("started_at"), "started_at", where)
        end = _time(answer.get("submitted_at"), "submitted_at", where)
        rows.append({
            "file": where, "fixture": name, "arm": answer["arm"], "trial": answer["trial"],
            "participant": answer["participant"], "held_out": key["fixtures"][name]["held_out"],
            "agent": answer["participant"].startswith("agent:"),
            "seconds": (end - start).total_seconds() if start and end else None,
            "tool_calls": answer.get("tool_calls"),
            "marks": marks, "right": sum(scored), "scored": len(scored), "correct": all(scored),
        })
    return rows


# --- The results table -----------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def arm_stats(rows: list[dict]) -> dict:
    right, scored = sum(r["right"] for r in rows), sum(r["scored"] for r in rows)
    correct = [r for r in rows if r["correct"]]
    return {
        "trials": len(rows),
        "fixtures": len({r["fixture"] for r in rows}),
        "accuracy": 100.0 * right / scored if scored else None,
        "correct": len(correct),
        "agent_calls_correct": _median([r["tool_calls"] for r in correct if r["agent"] and r["tool_calls"] is not None]),
        "agent_calls_all": _median([r["tool_calls"] for r in rows if r["agent"] and r["tool_calls"] is not None]),
        "agent_seconds_correct": _median([r["seconds"] for r in correct if r["agent"] and r["seconds"] is not None]),
        "human_seconds_correct": _median([r["seconds"] for r in correct if not r["agent"] and r["seconds"] is not None]),
    }


def gate_checks(rows: list[dict]) -> list[str]:
    """The plan's two Gate 1 lines, on held-out fixtures that both arms of a pair answered."""
    lines = []
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    if not by_arm["B"]:
        return ["Arm B: not run. Neither Gate 1 line can be checked until the Rune app (arm B) is run."]
    for other in ("A", "C", "R"):
        if not by_arm[other]:
            lines.append(f"B against {other}: arm {other} not run.")
            continue
        shared = {r["fixture"] for r in by_arm["B"]} & {r["fixture"] for r in by_arm[other]}
        if not shared:
            lines.append(f"B against {other}: no held-out fixture answered in both arms.")
            continue
        b = arm_stats([r for r in by_arm["B"] if r["fixture"] in shared])
        o = arm_stats([r for r in by_arm[other] if r["fixture"] in shared])
        names = ", ".join(sorted(shared))
        if b["accuracy"] is None or o["accuracy"] is None:
            lines.append(f"B against {other} on {names}: no scored answers.")
            continue
        diff = b["accuracy"] - o["accuracy"]
        ok = abs(diff) <= GATE_ACCURACY_PP or diff > 0
        lines.append(f"Accuracy, B against {other} on {names}: {b['accuracy']:.1f}% against "
                     f"{o['accuracy']:.1f}% ({diff:+.1f} pp): {'within ±5 pp or higher in B' if ok else 'FAILS'}.")
        if other not in ("A", "R"):
            continue
        speedups = []
        for label, field in (("median agent tool calls to a correct answer", "agent_calls_correct"),
                             ("median human time to a correct answer", "human_seconds_correct")):
            if b[field] is None or o[field] is None:
                lines.append(f"{label[0].upper() + label[1:]}, B against {other}: not measured (no correct trials in both arms).")
                continue
            ratio = o[field] / b[field] if b[field] else math.inf
            speedups.append(ratio >= GATE_SPEEDUP)
            lines.append(f"{label[0].upper() + label[1:]}, {other} over B: {_fmt(o[field])} / {_fmt(b[field])} = "
                         f"{ratio:.2f}× ({'meets' if ratio >= GATE_SPEEDUP else 'below'} 2.0×).")
        if speedups:
            lines.append(f"Speed line against {other} ({'met' if any(speedups) else 'not met'}): arm B must cut "
                         f"either measure by at least 2.0× against arm {other}.")
    return lines


def results_table(rows: list[dict]) -> str:
    held = [r for r in rows if r["held_out"]]
    practice = [r for r in rows if not r["held_out"]]
    out = ["# Gate 1 results", "",
           f"{len(held)} answer files on held-out fixtures, {len(practice)} on the practice fixture "
           "(listed last, not counted).", "",
           "## By arm (held-out fixtures)", "",
           "| Arm | Material | Trials | Fixtures | Accuracy | All-correct trials | Median agent tool calls (correct / all) "
           "| Median agent seconds to correct | Median human seconds to correct |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, material in ARMS.items():
        mine = [r for r in held if r["arm"] == arm]
        if not mine:
            out.append(f"| {arm} | {material} | not run | | | | | | |")
            continue
        s = arm_stats(mine)
        out.append(f"| {arm} | {material} | {s['trials']} | {s['fixtures']} | {_fmt(s['accuracy'], 1, '%')} | "
                   f"{s['correct']} of {s['trials']} | {_fmt(s['agent_calls_correct'])} / {_fmt(s['agent_calls_all'])} | "
                   f"{_fmt(s['agent_seconds_correct'])} | {_fmt(s['human_seconds_correct'])} |")
    out += ["", "## By question (held-out fixtures; right / scored)", "",
            "| Arm | " + " | ".join(QUESTION_IDS) + " |", "| --- |" + " --- |" * len(QUESTION_IDS)]
    for arm in ARMS:
        mine = [r for r in held if r["arm"] == arm]
        if not mine:
            continue
        cells = []
        for qid in QUESTION_IDS:
            marks = [r["marks"][qid] for r in mine if r["marks"][qid] is not None]
            cells.append(f"{sum(marks)} / {len(marks)}" if marks else "–")
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    out += ["", "## Gate 1 checks (plan Step 9)", ""]
    out += [f"- {line}" for line in gate_checks(held)]
    out += ["", "## Every answer file", "",
            "| Fixture | Arm | Trial | Participant | Right / scored | All correct | Seconds | Tool calls | Wrong |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(held, key=lambda r: (r["fixture"], r["arm"], r["trial"], r["participant"])) + \
            sorted(practice, key=lambda r: (r["fixture"], r["arm"], r["trial"], r["participant"])):
        wrong = ", ".join(q for q, m in r["marks"].items() if m is False) or "–"
        name = r["fixture"] + ("" if r["held_out"] else " (practice)")
        out.append(f"| {name} | {r['arm']} | {r['trial']} | {r['participant']} | {r['right']} / {r['scored']} | "
                   f"{'yes' if r['correct'] else 'no'} | {_fmt(r['seconds'], 0)} | {_fmt(r['tool_calls'])} | {wrong} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="generate fixtures, the key and the trial folders (needs the SDK)")
    b.add_argument("out", type=Path, nargs="?", default=HERE / "out")
    b.add_argument("--fixture", action="append", help="only this fixture (repeatable)")
    b.add_argument("--no-rerun-arm", action="store_true", help="skip arm R (stock Rerun) and its recordings")
    s = sub.add_parser("score", help="score answer files against the key")
    s.add_argument("key", type=Path)
    s.add_argument("answers", type=Path, nargs="+", help="answer files, or folders searched for *.json")
    s.add_argument("-o", "--output", type=Path, help="write the Markdown table here (default: print it)")
    args = parser.parse_args(argv)
    if args.command == "build":
        key = build(args.out, args.fixture, rerun_arm=not args.no_rerun_arm)
        print(f"wrote {len(key['fixtures'])} fixtures × {len(ARMS) - args.no_rerun_arm} arms under {args.out / 'trial'}; "
              f"key in {args.out / 'key.json'} (keep it from participants)")
        return
    key = json.loads(args.key.read_text(encoding="utf-8"))
    table = results_table(score(key, load_answers(args.answers)))
    if args.output:
        args.output.write_text(table, encoding="utf-8")
    else:
        sys.stdout.write(table)


if __name__ == "__main__":
    main()
