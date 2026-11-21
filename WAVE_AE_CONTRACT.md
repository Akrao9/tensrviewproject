# Wave AE contract: five held items

**Approved by the author on Nov 21, 2026, all five for implementation, with amendments to items 1, 4 and 5, which are made here. Nothing on this page is built.** Amended the same day on the author's second word: this page's five readings stand, with one addition to section 1 (a run where no pair was compared), a condition on section 3's warning, and section 6 for late rows (L30), which the author chose then. It is on the branch `contracts/wave-ae`, cut from `265311c`, the base of wave AE. The Python detectors (`sdk/src/tensorquill/evidence.py`) and the server (`server/src/detect.rs`) build sections 1, 2 and 4 from this page and from the cases in `sdk/tests/fixtures/wave_ae/cases.json`, each from the cases alone, and must give each case's `expected`. The capture (`sdk/src/tensorquill/capture.py`) builds the record's side of sections 1, 3 and 4. Section 5 is the readers' and starts once lane AE1 has released their files.

Each case is a **pin** (behaviour the base has and this page keeps; `sdk/tests/test_wave_ae_contract.py` runs it on every tree) or **new** (not built on the base; its `expected` is the target, and its `today` is a record of what `265311c` answers, written once by `make_cases.py` and never recomputed). 36 cases: 21 pins, 15 new. A new case's `expected` is written by hand from this page; the generator shows only that the base does not meet it.

Three rules hold for every section:

- **A new key appears only when its condition holds.** A record, an evidence file, a server report and a status written for a run without the condition are byte for byte what they are on the base. The record keys travel in `extra_json`, as `step_skipped_by` and `moe_source` do, so the Arrow schema and the protocol number do not change.
- **Nothing here blocks, pauses, terminates or raises into training.** A wrong argument is refused at `init`, never inside a step.
- **Gate 1.** What its arms are shown is compared with the approved baseline at the trial merge, after everything here is built, and never read to choose a rule. A difference is brought to the author before the merge.

Where this page states something the author's words do not, it says "this page's reading", and that reading is the author's to change.

## 1. Separately clipped groups (`clip_calls`)

**The fault** (lane AD3, `lane-ad3/l20_two_clips.py`). `clip_grad_norm` on a record is the root of the summed squares of what every `clip_grad_norm_` call of the step returned, which is right as a total. `update_spike` is the one kind that splits that pre-clipping total by parameter (`_shared_totals`), and only for a pair of records that name different parameters. The split assumes one call scaled every gradient by one factor. With group `a` (norm 0.40) and group `b` (16.0) clipped by two calls it gives `a.weight` 5.94 where the truth is 0.40, and no record says there were two calls.

**The record.** A step's record carries `clip_calls`, a whole number above 1, when more than one `clip_grad_norm_` call covering this session's parameters ran in the step. With one call, or none, there is no key.

- **An absent key does not mean one call.** Under `torch.compile` the session knows that clipping happened and not how many times, and a capture written before this key never says. So absent means "one call, or the count is not known". The README says so.
- A value that is not a whole number above 1 (1, text, a fraction, a boolean) is read as no key (`clip_calls_that_is_one_is_read_as_no_key`, `clip_calls_that_is_text_is_read_as_no_key`, `clip_calls_that_is_a_fraction_is_read_as_no_key`, `clip_calls_that_is_true_is_read_as_no_key`, pins).

**The rule, both engines.** Only the comparison that needs a per-parameter split is suppressed, in the author's words: "Suppress only the update_spike comparison that needs a per-parameter split when either record shows multiple clip calls. Keep comparisons that can use the recorded values directly."

| The pair of records `update_spike` compares | Either has `clip_calls` above 1 | What happens |
|---|---|---|
| Names different parameters, so the totals would be split | Yes | Not compared: nothing is flagged, and its change joins no baseline (`a_split_pair_with_two_clip_calls_is_not_compared`, `a_split_pair_whose_earlier_record_had_two_clip_calls`, new) |
| Names different parameters | No | Split and compared, as on the base (`a_split_pair_with_one_clip_call_is_compared`, pin) |
| Names the same parameters, or cannot be split for another reason the base already has (a record without `grad_norms`, a norm that is not finite, no shared parameter, norms that are all 0) | Either | Compared with the totals as recorded, as on the base (`a_pair_naming_the_same_parameters_is_compared_whatever_the_clip_calls`, `a_pair_that_cannot_be_split_is_compared_whatever_the_clip_calls`, pins) |

- The later record of a pair that was not compared is still the record the next one is compared with (`after_a_pair_that_is_not_compared_the_next_is`, new).
- **What the evidence says.** When at least one pair was not compared, `not_checked` gains one entry: `{"check": "clip_groups", "kinds": ["update_spike"], "comparisons": N, "step": S, "reason": ...}`, with `N` the pairs not compared and `S` the step of the later record of the first such pair (`two_pairs_that_are_not_compared`, new). The entry also carries `compared`, the pairs that were compared. While `compared` is above zero `update_spike` stays `checked` in `kind_checks`. The author: "if every eligible comparison was suppressed, the reader must make clear that zero were judged". So with `compared` at zero the kind reads `unavailable`, with the entry's reason, and the reason says that none of the comparisons was made (`every_pair_needs_a_split_and_none_is_compared`, new). As any `not_checked` entry does, it makes the run read `degraded` (approved by the author: "A suppressed clip comparison may mark coverage degraded"). The reason's words: "clipped by more than one clip_grad_norm_ call: the pre-clipping total cannot be split by parameter, so N comparison(s) of update_spike were not made".
- A run whose records carry `clip_calls` and whose pairs never need a split has no entry and no other difference.
- **Where the entry sits** (added at the trial merge, where the two engines had each chosen a place): right after `update_size`, the other entry of `update_spike`, and before `gradients` (`clip_groups_beside_other_entries`, new).

**The record's side, in the capture lane's tests.**

| Test | Loop | Expected on the record |
|---|---|---|
| The regression | `lane-ad3/l20_two_clips.py`: group `a` (norm 0.40) and group `b` (16.0), one `clip_grad_norm_` call each | `clip_calls: 2`, and `clip_grad_norm` 16.0050 as on the base |
| One call | The same groups clipped by one call over both | No `clip_calls`; the record is what it is on the base |
| No clipping | A step that does not clip, after one that clipped twice | No `clip_calls` on it: the count is the step's own |
| Compiled | The clipping inside a compiled step | No `clip_calls`, whatever the number of calls: the count is not known there |

**Not built, stated:** an exact split from each call's own norm and parameters.

## 2. A stalled parameter and a resume

**The fault** (lane AD1, `lane-ad1-item4-history-across-resume.py`). A process's first weight check compares nothing, so its record has no `weights_unchanged`, and a row without the list holds every streak. A streak therefore runs through a resume: unchanged at 50 and 100, a resume at 101, an uncompared check at 150 and unchanged at 200 raises the stall at 200, on three comparisons with the interval from 100 to 150 never compared.

**The rule, both engines.** This amends contract 2 of `stalled_weights` (signed off Nov 19, 2026), in the author's words: "Reset the streak at the attempt boundary. The first post-resume check establishes the baseline."

- Over one rank's resolved records in step order, when a record's attempt differs from that of the record before it, every parameter's streak starts again at zero before that record is read. A record without `attempt` is of attempt 0.
- Only the streak starts again. Which parameters were seen changing, and which were already reported, are kept: a stall raised before the resume stands, with its step (`a_stall_raised_before_the_resume_stands`, pin).
- In the fixture the incident belongs at step 300 (`a_stall_does_not_carry_across_a_resume`, new). The uninterrupted control is raised at 150 as on the base (`a_stall_in_one_attempt`, pin), and a stall wholly inside the resumed attempt is raised where it was (`a_stall_wholly_inside_the_resumed_attempt`, pin).
- A row that carries `weights_unchanged` is a comparison the capture made; a process's first check makes none and carries no list. No case states a new attempt whose first check carries one: the capture does not write it.

**What it gives up:** a parameter stalled right through a resume is reported up to three checks later, 150 steps at the default interval.

**Files that are pinned.** A golden file that holds a stall across a resume changes its step; the lanes report each one. Gate 1's held-out copies are not read to find out.

## 3. An optimizer before the one that ends the step (`ends_step=`)

**The fault** (lane AD3, `lane-ad3/item5_a.py`, case 1; pinned by `test_step_accounting.py::test_an_optimizer_that_steps_before_the_lead_is_taken_for_the_step_before`). The hooks cannot tell an optimizer that steps after the step-ending one in the same iteration from one that steps before it in the next. So an optimizer that sits before the step-ending one in the loop has its update written one step early, and the last record misses its update and lists its parameter under `weights_unchanged` although it moved.

**The argument.** `tensorquill.init(..., ends_step=optimizer)`, opt-in, with the default unchanged. In the author's words it "names the optimizer whose step closes each logical iteration".

- With it, a step ends when that optimizer's step returns, and only then. Every other optimizer's step since the last ended step belongs to the step that ends next: its update, and what it changed, are written on that record. Nothing is read again as a follower of the step before.
- An iteration in which another optimizer does not step writes nothing of that optimizer on its record: its previous update is not carried forward.
- Without the argument everything is as on the base, the pinned test included, which stays as the default's stated limit.
- **Approved by the author, each refused at `init` and never inside a step:** `ends_step` that is not an optimizer holding at least one of the model's parameters; `ends_step` together with `scaler=` (the scaler's `update()` ends the step there) or with `auto_step=False` (`session.step()` ends it).
- **Approved by the author, with a condition:** when the named optimizer has not stepped while others stepped ten times, one warning says so, once; nothing is recorded until it steps. The condition: "provided it cannot raise into training under -W error". The warning is given so that a warnings filter that turns warnings into errors cannot raise out of the user's optimizer step, and a test runs the loop under that filter.

**Regressions, in the capture lane's tests (a training loop each, so no case in the file).**

| Test | Loop | Expected |
|---|---|---|
| The regression | `item5_a.py`, case 1: `other` starts at iteration 2 and sits before `first`; `ends_step=first`, `weights=1` | `4.weight`'s update is on the records of steps 2, 3 and 4, and the record of step 4 does not list it under `weights_unchanged`. On the base: steps 1, 2 and 3, and listed unchanged at 4 |
| The author's control | The same loop, but `other` does not step in one middle iteration | That iteration's record has no `update_rms` for `other`'s parameters and lists them unchanged; the records around it are as in the regression |
| A true follower | `first.step(); other.step()` with `ends_step=other` | The same records the default gives for that loop |
| Two halves | A GAN's two players, each with its own backward, `ends_step=` the second | Each player's update is on the record of the iteration it stepped in |
| The default | The pinned test, unchanged | As on the base |

## 4. Several optimizers through one scaler (`step_skipped_part`)

**The fault** (lane AD3, `lane-ad3/item5_b.py`). With two optimizers through one `GradScaler` and an Inf in only the second's gradients, PyTorch applies the first and skips the second. The record says `step_skipped: true`, `step_skipped_by: "scaler"` for the whole step. `loss_scale_trouble` then counts a step on which an update occurred toward its ten consecutive skipped updates, which can raise a false incident. The author: "Recording the partial skip while leaving that counter unchanged would fix the wording but preserve the bug."

**The record.** When the scaler skipped at least one optimizer and applied at least one in the same step, the record keeps `step_skipped: true` and `step_skipped_by` as today, and gains `step_skipped_part: {"skipped": n, "of": m}`, two whole numbers with 0 < n < m: the optimizers that went through the scaler since its last update, and how many of them it skipped. It is read from the scaler's own per-optimizer verdicts at the next flush, as `step_skipped` is, with no device sync added. Every optimizer skipped, none skipped, or one optimizer: no key.

**The counts of `loss_scale_trouble`, both engines.** This amends section 1 of `STEP_AND_EVIDENCE_CONTRACT.md`, in the author's words: "Count a confirmed mixed outcome once in recorded_steps, count it separately as partial, and reset the consecutive fully skipped streak because an update occurred."

| The record | `recorded_steps` | `skipped_steps` | `partial_steps` | The streak of fully skipped updates |
|---|---:|---:|---:|---|
| A skip the scaler confirmed, with a valid `step_skipped_part` (whoever else reported it) | +1 | +0 | +1 | Reset to zero |
| A skip the scaler confirmed, with no such key, or one that is not a mixed outcome | +1 | +1 | +0 | +1, as on the base |
| Every other record | As on the base | | +0 | As on the base |

- The paired regression: five full skips, one partial step and five full skips end at 11 recorded, 10 skipped, 1 partial and a streak of 5, and nothing is raised (`five_full_skips_one_partial_step_five_full_skips`, new). The same with the caller also reporting the partial step (`a_partial_step_the_caller_also_reported`, new).
- A partial step and then ten full skips raise the rule at the tenth of them (`a_partial_step_then_ten_full_skips`, new). An incident carries `partial_steps` beside `skipped_steps` and `recorded_steps` only when it is above zero.
- The controls stay: one optimizer (`eleven_full_skips_with_one_optimizer`, pin), and a key that does not state a mixed outcome, which is a full skip (`a_part_key_that_says_every_optimizer_skipped_is_a_full_skip`, `a_part_key_that_says_no_optimizer_skipped_is_a_full_skip`, `a_part_key_that_is_not_two_counts_is_a_full_skip`, pins).
- The key is the scaler's verdict on a skip it confirmed. On a skip only the caller reported, and on a record whose `step_skipped` is false, it says nothing, and the record is counted as on the base (`a_part_key_on_a_skip_only_the_caller_reported_says_nothing`, `a_part_key_on_an_applied_step_says_nothing`, pins). The cases of `STEP_AND_EVIDENCE_CONTRACT.md` are unchanged.
- A loss scale below the floor on a partial step raises the rule with reason `scale`, as on any step (`a_low_loss_scale_on_a_partial_step`, pin).

**The kinds judged parameter by parameter.** The record does not say which optimizer's update was applied, so, in the author's words, "Keep parameter-level checks unavailable where the capture cannot attribute the outcome." `grad_spike`, `update_spike`, what counts as update history, and `stalled_weights` read a partial step as they read a skipped one today: not judged, and a stall's streak neither grows nor ends (`a_stall_across_a_partial_step`, pin). That can miss evidence from the optimizer that applied its update, and never invents any.

**What the evidence keeps.** `skipped_updates` counts a partial step under `partial` and under no other name, with the key there only when its count is above zero (`skipped_updates_counts_a_partial_step_apart`, `skipped_updates_counts_a_partial_step_both_reported_apart`, new; `skipped_updates_without_a_partial_step`, pin). Where a reader says skipped updates in words, a partial step is "N step(s) on which the scaler skipped some optimizers and applied others". The all-ranks view and the MCP tools pass the block on as they do today.

**The record's side, in the capture lane's tests.**

| Test | Loop | Expected on the record |
|---|---|---|
| The regression | `lane-ad3/item5_b.py`: two optimizers through one scaler, an Inf only in the second's gradients at step 2 | `step_skipped: true`, `step_skipped_by: "scaler"`, `step_skipped_part: {"skipped": 1, "of": 2}`; the weight check on it as on the base (the second's parameters unchanged, the first's changed) |
| Every optimizer skipped | An Inf in both optimizers' gradients | `step_skipped: true`, no `step_skipped_part` |
| None skipped | Two optimizers, no overflow | `step_skipped: false`, no `step_skipped_part` |
| One optimizer | One optimizer through the scaler, with and without an overflow | Every record byte for byte what it is on the base |
| No sync | The regression's loop | No device sync added at the step: the verdicts are read at the flush, as `step_skipped` is |

**Not built, stated:** which parameters each verdict covers, so that each could be judged by its own optimizer's outcome.

## 5. An input that holds several run IDs

**The fault** (lane AD5, `lane-ad5-several-run-ids-repro.py`). `build_evidence`, `compare` on a named run, `list_incidents` and `get_series` refuse records of more than one run ID. `runs` and `list_runs` pick one run and never name the other. `first_divergence` called directly reads the rows of two run IDs given as one run as one run.

**The rule: refuse per input, and say so.** In the author's words: "first_divergence(A, B) must still compare a clean run A with a different clean run B. If either input itself contains multiple run IDs, return no_answer and identify the side and IDs."

- **`first_divergence(records_a, records_b)`.** Each input is checked on its own, over every row of the chosen rank, not its first and last. Two clean inputs with different run IDs are compared, as on the base (`two_clean_runs_with_different_ids_are_compared`, pin). An input whose rows hold more than one run ID gives `answer: "no_answer"`, a `reason` that names the side and the IDs, and `several_run_ids`, a map from the side (`"a"`, `"b"`) to its sorted run IDs, present only then (`an_input_with_another_run_id_only_in_the_middle`, `the_other_input_with_two_run_ids`, `both_inputs_with_two_run_ids`, new). A row with no `run_id` is of no named run and joins only rows that have none, as `check_one_run` reads it.
- A step written twice within one run ID is still one run (`a_step_written_twice_within_one_run`, pin). The base's `test_divergence.py::test_a_step_written_twice_keeps_the_later_record` makes its second row from the other run's; its input is given one run ID, and what it asserts stays.
- **`runs` and the MCP `list_runs`.** A records file whose rank holds more than one run ID is one row with `evidence.state: "several_run_ids"`, naming the IDs, with no run state and no incident claimed, whatever evidence file lies beside it. A file of one run ID is listed exactly as on the base.
- **`compare`.** When a named run is not found and a file in the folder holds that ID among several, the error names the file and the IDs it holds.
- **The author's test:** a file whose first and last rows say `r1`, with `r2` only in the middle, through each reader. It catches a reader that checks only the endpoints.
- **To settle in the lane, not here:** lane AD5 wrote that `runs` would have to read `run_id` from every line of a large file, while the notes above `_RECORD_HEAD` in `runs.py` say the scan reads the run ID of every line already. The lane reports which is so, and the cost if any, before building.

## 6. Rows not stored because their step was not above one already kept (L30)

The server keeps a row only when its step is above the last step it kept for the attempt. Lane AE1 showed that for a discarded row at or below the highest stored step the server cannot tell a step that arrived late from a row sent again. The author chose both counts, in these words: "Report the total number of rows discarded because their step was at or below the attempt's highest kept step. Separately report the lower bound on distinct late steps the server can prove within a part. Count a step in that lower bound only once, and only if it was above the watermark at the start of the part and had not already been kept. Leave the remaining discards unclassified. Keep store ordering unchanged."

- **`rows_not_newer`**, on the rank's entry in `watchdog.json`, only when above zero: every row received with its keys readable and not stored because its step was not above the highest step kept for its attempt. A row sent again is counted again.
- **`late_steps_confirmed`**, beside it, only when above zero: distinct steps that were above the watermark when their part began, were not kept, and arrived inside that part. A step is counted once. It is committed when the part is stored.
- The rest of `rows_not_newer` is not classified. No late row becomes stored.
- **The three tests, against the real server:** the 60-row part of `test_wave_ad_seams.py` gives 12 and 12; a stored part sent again adds zero to `late_steps_confirmed`; the same discarded step sent again adds to `rows_not_newer` and not to `late_steps_confirmed`.
- **What a reader says**, wherever dropped rows are said (one rank's evidence, the all-ranks view, `runs`, `compare`, the MCP tools), carrying both keys as written: "The server received N row(s) whose step was not above a step it had already kept for their attempt, and did not store them. A row sent again is counted again, so N is not a number of steps lost. M distinct step(s) among them are confirmed to have arrived out of order inside one stored part, and are not stored. M is a lower bound: a late step that crosses a part boundary is not in it, so the same rows can give another M when the parts are cut elsewhere. The other rows are not told apart from rows sent again." With M at zero the two sentences on M become "None is confirmed as a step that arrived out of order."
- **Who builds it:** lane AE1 the server (`ingest.rs`, `watchdog.rs`); lane AE8 the readers in `runs.py`, `compare.py` and `mcp_server.py`; lane AE6 the readers in `evidence.py`. One owner a file.

## RS10's transport, the author's word (Nov 21, 2026)

"I approve the sender supplying its batch start and the server using it to identify the refused range. A refusal confirms only the range rejected; it does not acknowledge those bytes as stored." The joined restart regression passed on the merged tree, and the author then approved the transport: "RS10 transport is approved." Its wording is part 8 of `DETECTOR_RULES_CONTRACT.md`.

## Who builds what

| Section | Capture (`capture.py`) | Python detectors (`evidence.py`) | Rust server (`detect.rs`) | Readers |
|---|---|---|---|---|
| 1 | `clip_calls` on the record | The suppressed pair and the `clip_groups` entry | The same, reading `clip_calls` from `extra_json` | |
| 2 | | The streak at an attempt boundary | The same | |
| 3 | `ends_step=` | | | |
| 4 | `step_skipped_part` on the record | The counts, `partial_steps`, `skipped_updates.partial` | The same, reading the key from `extra_json` | |
| 5 | | | | `divergence.py`, `runs.py`, `compare.py`, `mcp_server.py`, once lane AE1 has released them |

At the trial merge both engines are held to the same answer on every case of sections 1, 2 and 4 through `build_evidence`, the full suite is run with the tree's own rebuilt server, and Gate 1's shown files are compared with the approved baseline.
