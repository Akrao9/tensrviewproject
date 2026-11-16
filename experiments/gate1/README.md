# Gate 1 trial harness

Tooling for Steps 7 and 8 of the plan's Gate 1: the pre-registered questions, and a harness that prepares each fixture's material per arm and scores the answers. The questions, rules, arms and thresholds are in [PREREGISTRATION.md](PREREGISTRATION.md). This folder holds the answers, so keep it, and `experiments/moe-bakeoff/`, away from participants.

The harness calls no model and no service. A person, or a coding agent the author runs, reads a trial folder and writes `answer.json`.

```sh
.venv/bin/python experiments/gate1/harness.py build /path/outside/the/repo/gate1
.venv/bin/python experiments/gate1/harness.py score /path/outside/the/repo/gate1/key.json answers/ -o results.md
.venv/bin/python -m pytest experiments/gate1 -q
```

- `build` runs `experiments/moe-bakeoff/generate.py`'s fixtures through the SDK into `OUT/fixtures/`, and writes the key to `OUT/key.json`. It also writes `OUT/trial/<fixture>/<arm>/` for arms A (tracker scalars), B (Rune app placeholder), C (evidence and summary) and R (stock Rerun, optional; `--no-rerun-arm` skips it). It needs the SDK, and `rerun-sdk` for arm R.
- `score` checks each answer file's format, scores it, and prints or writes a Markdown table: by arm, by question, the plan's two Gate 1 lines, and every file. It uses the standard library only.
- `validation/perfect/` and `validation/wrong/` are hand-written answer sets. The first scores 100%; the second scores 21 of 32, with the mistakes listed in the pre-registration.
- `test_gate1.py` checks the key against the planted values, the contents of each arm's folder (arm A cannot carry the answer), the two answer sets, and the scoring rules.
