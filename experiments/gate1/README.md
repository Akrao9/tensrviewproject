# Gate 1 trial harness

Tooling for Steps 7 and 8 of the plan's Gate 1: the pre-registered questions, and a harness that prepares each fixture's material per arm and scores the answers. The questions, rules, arms and thresholds are in [PREREGISTRATION.md](PREREGISTRATION.md). This folder holds the answers, so keep it, and `experiments/moe-bakeoff/`, away from participants.

The harness calls no model and no service. A person, or a coding agent the author runs, reads a trial folder and writes `answer.json`.

```sh
.venv/bin/python experiments/gate1/harness.py build /path/outside/the/repo/gate1
.venv/bin/python experiments/gate1/harness.py score /path/outside/the/repo/gate1/key.json answers/ -o results.md
.venv/bin/python -m pytest experiments/gate1 -q
```

- `build` runs `experiments/moe-bakeoff/generate.py`'s fixtures through the SDK into `OUT/fixtures/`, and writes the key to `OUT/key.json`. It also writes `OUT/trial/<fixture>/<arm>/` for arms A (tracker scalars), B (Rune app placeholder), C (evidence and summary) and R (stock Rerun, optional; `--no-rerun-arm` skips it). It needs the SDK, and `rerun-sdk` for arm R.
- `score` checks each answer file's format, scores it, and prints or writes a Markdown table: by arm, by question, the gate lines, and every file. It uses the standard library only. The gate lines come in two sections:
  - **Arm C against arm R** (`evidence_gate_checks`), the gate as amended Nov 17, 2026. Agent trials only, on the held-out fixtures both arms answered (named when fewer than six). Accuracy: C's minus R's must be −5.0 points or more, compared exactly, so at 96 items each C may be 4 behind and not 5. Tool calls: C's median agent tool calls to a correct answer must be no more than R's; no correct trial in C fails the part, none in R with one in C meets it, and when both have correct trials, one without a count leaves the part undecided. Both parts must hold. The section also says whether the two arms' files name the same model (the text after `agent:`). The time limit is not in the answer format, so the section says it is not recorded. With no agent trials in C or R, as in the validation sets, it says "not checked".
  - **The plan's two Step 9 lines** (`gate_checks`): arm B against A, C and R, shown as "not run" while arm B has no answers.
- `validation/perfect/` and `validation/wrong/` are hand-written answer sets. The first scores 100%; the second scores 21 of 32, with the mistakes listed in the pre-registration.
- `test_gate1.py` checks the key against the planted values, the contents of each arm's folder (arm A cannot carry the answer), the two answer sets, and the scoring rules.
