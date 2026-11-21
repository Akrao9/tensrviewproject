# Gate 1 trial: pre-registration

Written Nov 16, 2026, before any trial was run, for Steps 7 and 8 of [the plan](../../TensorQuill%20v3%20—%20Product%20and%20Build%20Plan.md) ("Weeks 2–3: what must the Rune app do first? (Gate 1)"). What is fixed here (the questions, the scoring rules, the arms, the fixtures, what is measured) does not change once the first trial starts. Any later change goes under "Amendments" at the end, dated, with its reason.

This document holds the answers. Keep it away from participants and agents. Keep away the rest of this folder too, along with `experiments/moe-bakeoff/` (its README, `generate.py` and `artifacts/answer-key.json`).

## Fixtures

The fixtures are made by `experiments/moe-bakeoff/generate.py` through the SDK's own pipeline (Step 6). `harness.py build` runs that pipeline again into its output folder. It does not read the git-ignored `artifacts/` folder, so every trial gets evidence from the SDK version it runs with.

| Fixture | Role | Run |
| --- | --- | --- |
| `seeded` | **Practice**, visible. Its answers are in the bake-off README. It may be used to show a participant the answer format. It is never counted. | MoE, 60 layers × 128 experts, top-1, 4 sampled steps |
| `unseen-1` | **Held out** | MoE, 48 × 64, top-1, 8 sampled steps |
| `unseen-2` | **Held out** | Dense, 12 blocks, 400 steps |
| `unseen-3` | **Held out** | MoE, 24 × 64, top-4, 31 sampled steps |
| `unseen-4` | **Held out** | MoE, 32 × 128, top-1, 9 sampled steps |
| `unseen-5` | **Held out** | Dense, 20 monitored blocks, 400 steps |
| `unseen-6` | **Held out** | MoE, 24 × 32, top-1, 7 sampled steps |

The gate is judged on the six held-out fixtures only. Their names say nothing about the failure.

## The six questions

Every participant gets the same prompt, for every fixture and every arm. It is byte-identical. `harness.py` writes it as `prompt.md`, and a test checks that every folder holds the same text. The prompt lists the 32 incident kinds the SDK can report as of Nov 16, 2026, plus `none`. A constant list like this tells the participant nothing about any one run.

| # | Field | Question (as in the prompt) |
| --- | --- | --- |
| 1 | `kind` | Which kind of failure is the run's first incident? One name from the list, or `none`. |
| 2 | `step` | At which training step does that failure first appear in the recorded data? For a run that records only some steps, the first recorded step that shows it. |
| 3 | `location` | Where is it? A router failure: the layer number as the run numbers its routers (`layers.17.router` is layer 17). A gradient failure: the full parameter name. An activation failure: the module name. |
| 4 | `overloaded_experts` | A router failure: the IDs of the experts on that layer that get more than their share at that step (`[]` if none). Any other failure: `[]`. |
| 5 | `dead_experts` | A router failure: the IDs of the experts on that layer that get no tokens at that step (`[]` if none). Any other failure: `[]`. |
| 6 | `magnitude` | How large is it at that step? A router failure: the largest share of the router's assignments held by one of the experts named in question 4, counted as that expert's count over the sum of the layer's counts (0 to 1), or `null` if question 4 named none. A gradient or activation failure: the named parameter's gradient norm, or the module's reading, divided by its typical value before the failure. |

Questions 1–5 are the plan's Step 7 requirements ("the first failing step, layer, and failure kind/expert IDs"). Question 6 checks that the participant has read the data and not just the headline. A free-text "what to do next" was considered and left out, because no answer key or exact rule can score it.

### Scoring rules

| Field | Correct when | Scored on |
| --- | --- | --- |
| `kind` | A string equal to the key, ignoring case and surrounding spaces | All fixtures |
| `step` | An integer equal to the key (`1300.0` counts as 1300; `true` does not) | All fixtures |
| `location` | Router failures: these forms all mean the same layer and must equal the key's layer: `17`, `"17"`, `"layer 17"` and `"layers.17.router"` (leading zeros are ignored). Other failures: a string equal to the key's name after trimming spaces, case-sensitive, with no partial names | All fixtures |
| `overloaded_experts` | A list of integers whose set equals the key's set. Order and repeats do not matter. `null` is wrong, even when the key is empty | MoE fixtures only |
| `dead_experts` | Same rule | MoE fixtures only |
| `magnitude` | Router failures with an overloaded expert: a number within ±0.01 of the key's share. Router failures with none: `null`. Dense failures: a positive number within a factor of 1.5 of the key's ratio (`key / 1.5 ≤ answer ≤ key × 1.5`) | All fixtures |

A missing value, `null` (except where `null` is the answer), or a value of the wrong type is wrong. **Questions 4, 5 and 6 count as right only when question 1 (`kind`) is right.** Without this rule, an answer of "no failure found" (`none`, `[]`, `[]`, `null`) would earn 4 of 32 points on fixtures whose key has an empty list or `null`. With it, that answer earns 0 of 32, which a test checks. On the two dense fixtures, questions 4 and 5 are **not scored** and are left out of the denominator. One complete set of the six held-out fixtures therefore has 4 × 6 + 2 × 4 = **32 scored items**.

**Accuracy** of an arm is the share of scored items answered right, over all its answer files on held-out fixtures. A trial is **correct** when every scored question in it is right. Time and tool calls "to a correct answer" are taken over correct trials only.

### The key

The key comes from the planted values in `generate.py` (`Fixture.answer`). The magnitude is read from the fixture's own step records at the planted step, never from the detectors. For router failures it is the counts at the incident layer. For dense failures it is the reading over the median of the 50 recorded steps before the onset; for `unseen-2` the onset is the step the gradient began rising (308).

| Fixture | `kind` | `step` | `location` | `overloaded_experts` | `dead_experts` | `magnitude` |
| --- | --- | --- | --- | --- | --- | --- |
| `unseen-1` | `moe_imbalance` | 1300 | 17 | 53 | 20–27 | 0.440 ± 0.01 |
| `unseen-2` | `grad_spike` | 310 | `blocks.7.mlp.fc2.weight` | (not scored) | (not scored) | 85.5, accepted 57.0–128.3 |
| `unseen-3` | `moe_imbalance` | 400 | 11 | 9–14 | none | 0.090 ± 0.01 |
| `unseen-4` | `moe_imbalance` | 2000 | 8 | none | 40 experts: 3, 9, 14, 21, 22, 30, 37, 41, 44, 50, 52, 58, 61, 63, 67, 70, 73, 78, 80, 84, 88, 91, 95, 96, 99, 102, 105, 108, 110, 113, 115, 117, 119, 121, 123–128 | `null` |
| `unseen-5` | `activation_spike` | 260 | `blocks.13` | (not scored) | (not scored) | 95.8, accepted 63.9–143.8 |
| `unseen-6` | `moe_imbalance` | 700 | 3 | 7, 22 | none | 0.390 ± 0.01 |

The traps in the fixtures are deliberate and fixed in advance. The wrong answer set in `validation/wrong/` exercises each one.

- `unseen-1`: a second overload starts at step 1500 in layer 9 (expert 5, 40%). It is later, so it is not the first incident.
- `unseen-2`: the gradient starts rising at step 308. The spike, and the incident, is at step 310.
- `unseen-3`: on a top-4 router each of experts 9–14 is picked by 36% of tokens. That is 9% of the router's assignments, and the question asks for the share of assignments. Expert 3 holds 17.5% of assignments in every layer and is healthy.
- `unseen-4`: expert 1 holds 34% of assignments, below the 35% overload rule. It is not overloaded, so the answer to question 6 is `null`.
- `unseen-5`: `blocks.1` carries very large activations at every step, and it is healthy.
- `unseen-6`: from step 500 to step 600, layer 9 sends 34% to expert 13. That is just under the rule and is healthy.

## The arms

What each participant gets, per fixture. `harness.py build` writes one folder per fixture and arm under `OUT/trial/<fixture>/<arm>/`. Each folder holds `prompt.md`, `answer.template.json` and that arm's material.

| Arm | Material | Files |
| --- | --- | --- |
| **A** | Tracker scalars only: loss, learning rate and total gradient norm per step, as MLflow or W&B would hold them | `scalars.csv` |
| **B** | The Rune viewer app with the run's evidence | `RUNE_APP_PLACEHOLDER.md` (not runnable yet, see below) |
| **C** | TensorQuill's `evidence.json` and its text summary (`incident_summary`) | `evidence.json`, `summary.txt` |
| R (optional) | Stock Rerun 0.43.2 with the bake-off blueprint, `evidence.json` and its summary | `evidence.json`, `summary.txt`, `run.rrd` |

**Arm A** is built from the step records' `step`, `lr`, `grad_norm` and `scalars` fields, and nothing else. Nothing per layer, per parameter, per module or per expert reaches it, and no detector runs. A test checks this in three ways:

- The CSV has only the columns `step`, `loss`, `lr` and `grad_norm`.
- No incident kind, router, expert, layer, block or statistic name appears in it.
- The CSV stays byte-identical when every per-layer, per-parameter and per-expert field in the records is scrambled.

**Arm A has nothing for the four MoE fixtures.** Their records carry no loss, no learning rate and no total gradient norm, so `scalars.csv` holds the header alone. Those records were made in Step 6 and are not changed here, because the server parity tests use them too. For the dense fixtures, arm A shows the following:

- `unseen-2`: the loss jump and a total-norm spike at step 310, about 10× its median. This can give `kind` and `step`. It cannot name the parameter, and the total-norm ratio falls outside the accepted range for `magnitude`.
- `unseen-5`: no visible sign of the failure.

Arm A's accuracy is therefore near zero by construction: "no failure found" scores 0 of 32, and the best arm A can do from its data is 2 of 32 (`unseen-2`'s kind and step). It is a floor, not a competitor.

**Arm B** is the author's Rune app and is not runnable yet. Its placeholder file names what the app must load at run time: the run's `evidence.json`, and for the dense fixtures the step records (`<name>.jsonl`), since dense evidence has no per-step samples. The placeholder names the two files and the build sub-folder they are in (`fixtures/<name>/`), not an absolute path, because that build folder also holds the key. It copies neither file, so arm B cannot be run by reading files instead of using the app. Until arm B has answers, the results table shows it as "not run" and the gate lines are not checked.

**Arm C** gets exactly the evidence file the SDK wrote, together with its summary. The summary states the answers to questions 1–5 outright, and question 6's share or ratio for most fixtures. Arm C measures what the evidence alone gives, and it is expected near the ceiling.

### How this differs from plan Step 8

Plan Step 8 defines different arms:

> (A) stock Rerun 0.43.2 with the bake-off blueprint + `evidence.json`, (B) the Rune app prototype + `evidence.json`, and (C) `evidence.json` alone.

This trial was asked for with arm A as tracker scalars and arm C as evidence plus summary. Both definitions are kept:

- Arms A, B and C are as above.
- The plan's own arm A is available as arm **R**.

The plan's speed threshold reads "relative to Arm A", and the plan meant stock Rerun. Against the scalar arm A it cannot be measured, because arm A will have no correct trials and so no time or tool calls to a correct answer. The results table therefore checks arm B against A, C and R alike. The plan's comparison is **B against R**. Which arm the author reports the gate against is the author's call, made before the first trial and entered under "Amendments".

Plan Step 7 also asks for questions "that neither default screen answers without interaction". These six do not meet that bar. The detector summary states questions 1–5 on both viewers' default screens, and arm C's `summary.txt` states them too. The bake-off README suggested verification questions that need interaction, for example one expert's load change in a neighbouring layer, or whether the overload persists at the next sample. If the author wants those, they are a seventh-and-later question set, added under "Amendments" before the first trial, with the same answer format.

## What is measured

Each answer file records the following:

- `participant`: `agent:<model>` or `human:<name>`. The scorer refuses any other form, because the prefix decides whether a trial counts toward the agent medians or the human ones.
- `trial`: 1–3.
- `started_at` and `submitted_at`: ISO 8601 with a time zone.
- `tool_calls`: an integer for agents, `null` for people.

The harness reports per arm:

- **Accuracy**: right items over scored items, held-out fixtures only.
- **All-correct trials.**
- **Median agent tool calls**: to a correct answer (the gate measure), and over all agent trials (reported, not judged).
- **Median seconds to a correct answer**: for agents, and separately for people (the author).

It also reports per-question accuracy per arm, and every answer file with the questions it got wrong.

Protocol, from the plan:

- Fresh coding-agent sessions, with the same model and system prompt in every arm.
- Three trials per fixture per arm.
- The author timed on two arms, with fixtures counterbalanced. The plan names arms A and B, meaning stock Rerun and Rune; here that is arm R against arm B, see above.

From the bake-off README:

- Both viewers start at the first incident sample, with the detector summary and the status heatmap visible before the timer starts.
- Each condition gets only its own folder.

Fixed by the author before the first trial, entered under "Amendments":

- The time limit per trial.
- The model.
- The counterbalancing order.

## Success thresholds (plan Step 9, quoted)

> - [ ] Accuracy across the 6 fixtures is within ±5 percentage points between arms (or higher in Arm B).
> - [ ] Arm B cuts median agent tool calls or median human time-to-correct-answer by **≥2.0×** relative to Arm A.
> - *Decision rule (changed Nov 14, 2026):* the Rune app is built either way (Step 22). The trial decides its order of work: the interactions that cut time or tool calls most come first, and the release ships with Rerun as the viewer until the Rune app reaches parity.

How the harness checks them:

- **Accuracy line**: for each other arm X, on the held-out fixtures both B and X answered, B's accuracy minus X's must be within ±5 points, or positive.
- **Speed line**: on the same shared fixtures, X's median over B's median, for agent tool calls to a correct answer and for human seconds to a correct answer. At least one of the two must be ≥ 2.0.

## Running the trial

1. `python experiments/gate1/harness.py build OUT` from the repository root, with the SDK installed (`pip install -e 'sdk[rerun]'`; `--no-rerun-arm` skips arm R and rerun-sdk). `OUT` defaults to `experiments/gate1/out/`, which is git-ignored. Use a folder **outside the repository** for anything a participant will see, because the repository holds the answers.
2. For each trial, copy `OUT/trial/<fixture>/<arm>/` into a fresh, empty folder outside the repository. Start a fresh agent session or a person there. Give arm B the Rune app, and arm R the Rerun viewer on `run.rrd`. Never give anyone `OUT/key.json` or `OUT/fixtures/`.
3. The participant writes `answer.json` from the template. The operator fills in `trial`, `participant`, `started_at`, `submitted_at` and `tool_calls`, then collects the file, for example as `answers/<fixture>.<arm>.<trial>.<participant>.json`.
4. `python experiments/gate1/harness.py score OUT/key.json answers/ -o results.md`. Scoring needs only the Python standard library.

The harness calls no model and no service. Whoever runs the trial runs the agents.

## Validation of the harness (Nov 16, 2026)

Two answer sets were written by hand, before any trial.

`validation/perfect/` holds the key's answers, written as a participant would write them, with magnitudes rounded. It scores 32 of 32 (100%), with all 6 trials correct.

`validation/wrong/` holds one plausible mistake or more per fixture, each built on a trap above. It scores 21 of 32 (65.6%), with no trial correct:

| Fixture | Mistake | Score |
| --- | --- | --- |
| `unseen-1` | The dead list stops at expert 26 | 5 of 6 |
| `unseen-2` | Step 308, the onset of the rise | 3 of 4 |
| `unseen-3` | Share of tokens, 0.36 | 5 of 6 |
| `unseen-4` | Decoy expert 1 named overloaded, with its share | 4 of 6 |
| `unseen-5` | `blocks.1`, with a ratio of 800 | 2 of 4 |
| `unseen-6` | The layer-9 decoy at step 500 | 2 of 6 |

The arm labels on these two sets (C and A) only exercise the table; they say nothing about what those arms would score. `test_gate1.py` checks both totals, file by file and question by question.

## Amendments

### Nov 16, 2026: the summary's count line says which kinds are short of history

Entered before any trial was run. It changes what arms C and R are shown (`summary.txt`, and `kind_checks` in `evidence.json`). It changes no question, no answer key and no scoring.

- **What changed.** Each kind in `kind_checks` now carries `enough_history`, and the 25 kinds read from step records carry `records_needed` beside `records`. The summary's count line names the checked kinds whose rule has not had enough history. This is the plan's decision "Enough history".
- **Why it shows here.** The five MoE fixtures are 4 to 31 records long, and `moe_imbalance` needs 11 records and `starved_experts` 50. The two dense fixtures have 400 records and their line is unchanged.

| Fixture | Records | The count line in `summary.txt` |
| --- | --- | --- |
| `seeded` | 4 | `3 of 32 kinds checked, 2 of them without enough history yet (moe_imbalance 4 of 11 records but reported by a faster rule, starved_experts 4 of 50), 29 not configured` |
| `unseen-1` | 8 | the same, with `8 of 11` and `8 of 50` |
| `unseen-3` | 31 | `3 of 32 kinds checked, 1 of them without enough history yet (starved_experts 31 of 50 records), 29 not configured` |
| `unseen-4` | 9 | as `seeded`, with `9 of 11` and `9 of 50` |
| `unseen-6` | 7 | as `seeded`, with `7 of 11` and `7 of 50` |
| `unseen-2`, `unseen-5` | 400 | `7 of 32 kinds checked, 25 not configured`, as before |

- **What did not change.** The incidents, the per-step samples and the run state of all seven fixtures are identical (the golden files hold them, and a script compared each file with the commit before). The line before this amendment read `3 of 32 kinds checked, 29 not configured` on the five MoE fixtures.
- **"Records needed" is what a kind's slowest rule needs (decided by the author Nov 16, 2026).** The field exists to stop an agent reading "nothing found" as healthy, which only works if "enough history" means every rule of the kind could have fired. With the fastest rule, `moe_imbalance` would say "enough" after one record while the rule that judges a Granite-style router still needs 11.
- **What a participant may make of it.** In four fixtures `moe_imbalance` is both reported and marked short of history. The incident was found by a rule of the kind that needs fewer records, the line says "but reported by a faster rule", and the incident stands. A participant who doubts the incident because of the clause is reading it wrongly. No question asks about the clause, and it is not scored.
- **Checked.** `test_gate1.py` (44 tests) and `experiments/moe-bakeoff/test_fixtures.py` (37 tests) pass unchanged. Not run: a trial with an agent or a person.

### Nov 17, 2026: the gate is judged on the evidence arm against stock Rerun, and every evidence file carries a `run` block

Entered by the author before any trial was run. It settles the choice left open under "How this differs from plan Step 8", and records one more change to what arms C and R are shown. It changes no question, no answer key and no scoring rule.

- **Why now.** The Rune viewer is postponed (plan decision "Rune viewer postponed", Nov 16, 2026), so arm B will have no answers and the plan's two lines, which are about arm B, cannot be checked. What the first release ships is the evidence with stock Rerun as the viewer. The question the gate can answer today is whether the evidence alone does as well as the evidence with stock Rerun beside it.
- **The comparison.** Arm **C** (`evidence.json` and its summary) is judged against arm **R** (stock Rerun 0.43.2 with the bake-off blueprint, the same `evidence.json` and summary, and `run.rrd`), on the six held-out fixtures, agent trials only.
- **The pass line.** Both parts must hold:
  1. *Accuracy.* C's accuracy minus R's is −5.0 percentage points or more. With three trials on each of the six held-out fixtures an arm has 96 scored items, so C may be at most 4 items behind R (4 of 96 is 4.2 points; 5 of 96 is 5.2 and fails). The 5 points are the plan's own, from Step 9.
  2. *Tool calls.* C's median agent tool calls to a correct answer is no more than R's. "To a correct answer" is this document's existing gate measure: the median over correct trials only. If C has no correct trial, the line fails. If R has no correct trial and C has one, the line is met.
- **A stated limit.** Arm R's material is arm C's plus one file, so the accuracy part is hard to fail: an agent in arm R can ignore the recording and answer as arm C does. The tool-call part is what can fail, in either direction. A pass says the recording is not needed to answer these six questions. It does not say the evidence is right, since both arms read the same evidence, and it says nothing about questions that need interaction (see the last paragraph of "How this differs from plan Step 8").
- **Fairness.** Every arm that is shown evidence is shown the same `run` block and the same "without enough history" clause. Arms C and R get byte-identical `evidence.json` and `summary.txt`; `test_arm_r_gets_what_c_gets_and_the_recording` holds this for every fixture. Arm A is shown neither file and is unchanged.
- **What the `run` block is.** Since the plan's decision "Projects, run names and comparing runs" (Nov 17, 2026) every evidence file carries `run`: the project, the run's name and its ID. In the fixtures it is `{"project": "uncategorized", "name": "<fixture>", "run_id": "<fixture>"}`, for example `unseen-2`. It repeats the fixture's name, which the folder already shows, and carries nothing from the key, the host or the user. Read from a fresh build of all seven fixtures on Nov 17, 2026. `summary.txt` is as the Nov 16 amendment left it.
- **What is fixed before the first trial, still to be entered here.** The model, and the time limit per trial. Both apply to every arm alike. A trial that passes the limit is scored on the answer file as it stands at the limit.
- **What no longer applies to this gate.** The author's own timing on two arms and the counterbalancing order were for arm R against arm B. This gate is judged on agent trials. They return with arm B.
- **What stays as written.** The plan's Step 9 lines stay quoted above and are checked when arm B has answers. Arms A and B are still built. The six questions, the key, the scoring rules and the fixtures are unchanged.
- **Not built yet.** `harness.py score` checks arm B against the other arms (`gate_checks`). It does not check C against R by the two parts above. Until it does, the results table gives each arm's accuracy and medians and the line is read from them by hand.
  - *Built Nov 17, 2026.* `harness.py score` now checks C against R by the two parts above (`evidence_gate_checks`), in its own section of the results table; the arm B lines are unchanged. Accuracy is compared exactly, not on rounded percentages. It also says whether the two arms' answer files name the same model, read from `participant`. The answer format has no field for the time limit, so the table says it is not recorded and it is compared with this amendment by hand. One case this amendment does not settle: both arms have correct agent trials and one of them has `tool_calls` `null`. The harness then leaves the tool-call part, and the verdict, undecided rather than guess; the author's to settle. The validation sets are human trials, so they leave this check "not checked" and still score 32 of 32 and 21 of 32. Not run: a trial with an agent.

### Nov 20, 2026: a speed ratio with a zero denominator is not measured

Entered on the author's word on Nov 20, 2026, before the scorer was changed. The only answer files in the repository are the two validation sets of "Validation of the harness"; no trial has been scored for the gate. It changes how one line of the results table is reported. It changes no question, no answer key, no fixture and no accuracy rule.

- **The rule, in the author's words.** "For a trial with a zero denominator, report the speed ratio as 'not measured.' Retain both arms' raw tool-call counts. The undefined ratio contributes neither a pass nor a failure." Missing counts remain distinct from observed zeros.
- **Why.** The speed line divides the other arm's median by arm B's. `harness.py` gave a zero median for arm B an infinite ratio, and an infinite ratio reads as meeting 2.0× (audit item X02). Zero tool calls would then pass the line with nothing measured.
- **How the scorer applies it.** This part is the application, not the author's text, and is the author's to correct. The author's text speaks of a trial, and the speed line is defined above over each arm's median. So a speed ratio whose denominator, arm B's median, is zero is reported as "not measured", with both arms' medians shown as they are. It is left out of "at least one of the two must be ≥ 2.0", as a measure with no correct trials in both arms already is. The same holds for the human-seconds ratio, which the same line of the scorer computes. A missing count (`tool_calls` `null`) is not a zero and is never turned into one.
- **What does not change.** The fixtures and `generate.py`. The accuracy line. The tool-call part of the Nov 17 amendment, which compares two medians and divides nothing. The case that amendment left open, one arm with `tool_calls` `null`, stays open.
- **Not built yet.** `harness.py score` still computes the infinite ratio. It is changed after this entry, with tests for zero over zero, a positive count over zero, and an ordinary ratio.
