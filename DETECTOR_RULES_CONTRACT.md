# Detector rules contract, wave AD

**Signed by the author on Nov 21, 2026, as written at `b195236`, but for the transport of part 8. Nothing here is built.** This commit adds to `b195236` only the record of that signature, the author's clarification of `layers_not_read`, and one case for it; no rule and no other case differs. It is on the branch `contracts/wave-ad`, cut from `eb5a71b`. Both engines (the Python detectors in `sdk/src/tensorquill/evidence.py` and the server in `server/src/detect.rs`) build from this page and from the cases in `sdk/tests/fixtures/rules/cases.json`, and must give each case's `expected`.

What was measured before any rule was proposed is in `experiments/rule-selection/RESULTS.md`. Every input to that is on `experiments/rule-selection/allowlist.json`. Gate 1's six held-out cases have not been read, and stay held out.

## Where each decision stands

| # | Decision | The author's word, Nov 21, 2026 |
|---|---|---|
| 1 | The `activation_drift` rule (part 2) | **Signed as written at `b195236`**, after the fresh run was reported |
| 2 | `moe_imbalance` on a router that has not been trained (part 3) | **Approved and kept closed**, with when the incident began kept apart from when it became reportable, and the first reportable router winning |
| 3 | At least 10 earlier finite values for a score (part 1) | **Approved** |
| 4 | What a reader shows when the MAD is zero (part 1) | **Approved with other words:** "no score: the earlier values had zero median absolute deviation" |
| 5 | The four measurements and their names (part 6) | **Approved.** They and the score stay descriptive |
| 6 | What a refused batch does to its stream (part 8) | **Approved in intent. The transport is not signed:** lanes AD2 and AD4 build and test the proposed one; final acceptance needs the real server to deliver the refusal and the production sender to resume at the right byte |
| 7 | Three keys on the wire (part 7) | **Approved**, with two clarifications that are now in part 7 |

Released on the author's word: this contract to lanes AD3, AD4 and AD5 for the wire keys, to AD2 and AD4 to build and test the proposed transport of part 8, and to the engine lanes AD7 and AD8 once the lanes they are cut from have reported.

## 1. The score beside a ratio (S6)

Wherever an incident records a `ratio` that was taken against a window of earlier values, it also records a `score`: how far the value is from the window's median, in units of the window's spread.

- **The arithmetic.** `score = (value - median) / (1.4826 * MAD)`. The median is the rules' own (`_median`: the mean of the two middle values of an even count). The MAD is the median of the absolute distances from that median. Both are taken over the same earlier values the ratio's baseline was taken over, and never over the value itself.
- **How many earlier values it needs.** At least 10 finite values in the window. With fewer the score is `null` and `score_note` is `"short_history"`. A value that is not a finite number does not count.
- **A zero MAD.** When more than half the window is the same value the MAD is 0. Nothing is divided by it: the score is `null` and `score_note` is `"no_dispersion"`.
- **What a reader shows.** A number to one decimal place ("score 31.2"). For `short_history`: "no score: fewer than 10 earlier values". For `no_dispersion`: "no score: the earlier values had zero median absolute deviation". A reader never shows a null score as 0 and never leaves the line out.
- **Why those words.** Six equal values and four different ones have a zero MAD, though the values vary. "The earlier values did not vary" would be false there. That example is a case (`score_six_alike_and_four_different`), and each case carries the words a reader shows as `display`, which a reader test in each lane pins.
- **It is a description.** By the author's word the score and the four measurements of part 6 stay descriptive: nothing is judged from them in this wave.
- **No trend is removed first.** The sibling repository scores residuals from a local trend. On TensorQuill's own captures that changed the largest healthy score from 75.9, 11.9 and 24.3 to 75.5, 8.8 and 13.7, so it is left out of this wave.
- **It raises nothing.** No rule reads the score. On these captures it separates activation faults from healthy runs by a wide margin and does not separate the total gradient norm (a healthy Granite step scores 76, two seeded faults 31 and 54).

Where the key appears, each beside its `ratio`:

| Incident | Entry | The window |
|---|---|---|
| `grad_spike` | The incident itself | The totals of the 50 records before the step |
| `grad_spike` | Each of `parameters` | That parameter's norms in its own window |
| `activation_spike` | Each of `modules` | The 50 records before, of that module and statistic |
| `activation_drift` | Each of `modules` | The 50 records the baseline is taken from (part 2) |
| `attention_collapse`, `attention_logit_growth` | Each of `modules` | The earlier probes of the baseline. There are at most 5, so the score is always `null`, `short_history` |

Cases of the arithmetic: `score_above`, `score_below_is_negative`, `score_ten_earlier_values`, `score_nine_earlier_values`, `score_values_that_are_not_finite_do_not_count`, `score_no_dispersion`, `score_all_equal`, `score_six_alike_and_four_different`.

Cases over records, where each expected incident is today's with the score added, and the generator checks each window against the baseline today's rule reports: `score_on_grad_spike_by_the_total`, `score_on_grad_spike_by_one_parameter` (the second parameter path, whose baseline is the parameter's own earlier norms outside an excursion), `score_on_grad_spike_with_twelve_earlier_records`, `score_on_activation_spike`.

## 2. `activation_drift`

**Today.** A module's last 10 records against its first 50, at 3 times. A model trained from scratch grows its activations for as long as it trains, so the ratio has no ceiling: it fired on 18 of 28 healthy record sets.

**Proposed.** The median of a module's last 10 records against the median of the 50 records before those 10, for `std` and for `abs_max / std`, in that order. It fires at 3 times, up or down. A module is judged only from its 110th record, so that the 50 records of the baseline hold none of its first 50.

- The kind needs 110 records, where it needed 60.
- The incident keeps its shape: `step`, `modules` (each with `module`, `stat`, `baseline`, `recent`, `ratio`, and now `score`), `affected`, `affected_steps`.
- `dead_activations`, which shares the function, is not changed.

**Measured** (the table is in `RESULTS.md`): no healthy record set fires, with a largest ratio of 1.68 against the factor of 3. The three recorded GPU faults fire at step 254, as today, with ratios near 100.

**How far that evidence goes.**

- **It is in-sample.** The plain trailing baseline was tried first and fired on one healthy run at step 59. "Never use a module's first 50 records" was added after seeing that. So the result on these runs is a fit to them, not a test.
- **22 of the 28 healthy record sets are judged at all.** The other six have no layers or fewer than 110 records. B4's healthy run, the one the plain trailing rule fired on, has 102 records: it is quiet here because it is never judged, not because it was judged and passed.
- **The 22 are about ten distinct workloads.** Granite in three capture modes is one job. B9, B10 and the hardware simulation share one model and give the same numbers. The two nanoMoE arms share a model and a seed.
- **Six development sets were never judged:** too short, or without layers. One of them is the healthy run that prompted the warm-up change.

**The fresh run** (Nov 21, 2026; not used to choose the rule; run once against it as written). nanoMoE from scratch in another shape: 4 layers, width 128, 4 experts at top-1 with a router in every block, dropout 0.1, 900 steps, 901 records.

| | |
|---|---|
| Modules with layer statistics | 4 |
| Modules that reached 110 records | 4, with 901 records each |
| Modules judged | 4, from step 109, on 792 records each |
| Did any fire | No. The largest ratio was 1.44, at step 122 |
| Today's rule on the same run | Fires at step 114, and reaches 5.28 |

The author's reading, recorded as given: one seed from a familiar model family. It exercised the rule and showed a difference from the current detector. B4's short healthy run remains unjudged.

**What it gives up, stated so it is signed knowingly.**

- A slow drift never fires, however far it goes (`drift_a_slow_drift_is_not_seen`).
- A shift inside a module's first records is not seen by this kind (`drift_a_change_in_the_first_records_is_not_seen`). `activation_spike` judges from the 11th record.
- After a shift the incident's `affected_steps` is about 30, not the rest of the run: the new level becomes the baseline.

Cases: `drift_growth_from_scratch_is_not_a_shift`, `drift_a_step_change_is_named`, `drift_a_fall_is_named_too`, `drift_just_under_the_factor`, `drift_a_slow_drift_is_not_seen`, `drift_a_change_in_the_first_records_is_not_seen`, `drift_not_judged_before_110_records`, `drift_the_outlier_ratio`, `drift_one_of_two_modules`.

## 3. `moe_imbalance` on a router that has not been trained

**Today.** The earliest sample with an overloaded or dead expert. nanoMoE with its own settings is flagged at steps 0 and 1 (one expert at 36.2% against a line of 35%, at random initialisation) and never again in 2,001 steps.

**Proposed.** The samples are unchanged: each still says which experts are overloaded or dead. What changes is which flagged sample is reported.

- A router whose first sample is at step 0 and is flagged is in the state it started in.
- That leading run of flagged samples is not reported while it is shorter than 10 samples.
- Once it reaches 10 samples it is reported. **When it began and when it became reportable are two facts, and the incident carries both:** `step` is the first flagged sample's step, `reportable_step` is the tenth's, and `since_start` is true. A reader says "collapsed since step 0, reportable at step 9", never that it was found at step 0.
- Among several routers the incident is the one that could be reported first (`a_start_and_an_earlier_reportable_failure`).
- Any flagged sample after the leading run has ended is reported as today.
- A router whose first sample is not at step 0 is judged as today.

**Measured.** Not judging step 0 alone moves nanoMoE's incident to step 1. A rule that asks for three flagged samples in a row misses the practice fixture's overload, which is on the last two of four sampled steps. This rule is quiet on nanoMoE with its balance loss, fires at step 7 on another router without it, and reports every seeded collapse at the step and router today's rule does. A limit of 5 or of 50 gives the same answers on these inputs.

**While a router is still in its starting state.** A short capture must not read as checked and clean. When the records end with a router flagged on every sample since step 0 and fewer than 10 of them, nothing is reported for it, and the `moe_imbalance` entry of `kind_checks` carries `starting_state`: a list of `{"router", "samples"}`. With that key present `enough_history` is false, so the kind reads as checked without enough history (`start_and_the_capture_ends`).

**The public function.** `first_incident()` follows this rule, since `build_evidence` and the Rerun export both take the incident from it. The per-step samples it reads are unchanged.

**A positive case, and what stays open.** On the fresh run of part 2 the routing collapsed in all four routers, and both today's rule and this one reported it. Healthy-router coverage on a model of another shape remains open.

**What it gives up.** A router collapsed from its first step is reported 10 samples later than today.

Cases: `start_two_samples_then_even`, `start_nine_samples_then_even`, `start_ten_samples_is_reported`, `start_then_a_later_overload`, `start_ends_and_comes_back`, `not_at_step_0_is_not_a_start`, `start_in_a_sparse_capture`, `one_router_starts_another_fails`, `a_healthy_router`, `start_and_the_capture_ends`, `a_start_and_an_earlier_reportable_failure`.

## 4. The learning rate on `update_spike`

The incident gains `lr`, the record's `lr` list at the flagged step, and `lr_before`, the same at the step it was compared with. Each is present only when that record carries `lr`. Nothing about when the kind fires changes: the rule does not read `lr`, which the generator checks.

Cases: `update_spike_carries_the_learning_rate`, `update_spike_without_a_learning_rate`.

## 5. `parameters_not_judged`

Today the Python detectors name the parameters whose gradient was not read on the `stalled_weights` entry of `kind_checks`, and the server writes no such key.

- Both engines write `parameters_not_judged`, a sorted list of names, on the `stalled_weights` entry of one rank's `kind_checks`, when a `gradients` entry of `not_checked` lists them and the kind is not itself `unavailable`.
- The all-ranks view's `stalled_weights` entry carries the sorted union under the same key, and `parameters_not_judged_by_rank` (rank to list) when the ranks' lists differ. "The run-level check" of the bug list is the run-level part of the all-ranks view (`combine_rank_evidence`).

Cases: `stalled_weights_names_what_it_did_not_judge`, `the_all_ranks_view_names_them_by_rank_when_ranks_differ`.

## 6. Four measurements, with no incident and no threshold

Each is a block of numbers in one rank's evidence, as `weight_trends` is. It is present when the records hold what it reads and absent otherwise. No rule reads it, it changes no run state, and no line is drawn on it in this wave.

The first two are the robust local trend of the sibling repository LLM-Training-Debugger (`robust_local_trend`), written again from its description: a line through the median of the first third of the series and the median of its last third, each placed at the median of its own step numbers. It uses the actual step numbers. A series under six points has no trend.

| Block | Gap | What it reads | What it records |
|---|---|---|---|
| `gradient_trend` | B4 | The total gradient norm (the pre-clipping norm when the run clips), skipped steps left out | `series`, `points`, `first_step`, `last_step`, `first`, `last`, `growth` (last over first), `per_100_steps` (the slope as a share of `first`) |
| `attention_trends` | Q1 | Each attention module's largest logit, probe by probe | `statistic`, `modules`, and `listed`: the ten with the largest `growth`, each with the fields above |
| `update_trends` | Q2 | Each parameter's `update_rms` against the median of its own 50 records before, from 10 earlier values on | `parameters`, `judged`, and `lowest`: the ten parameters whose update fell furthest, each with `ratio`, `step`, `baseline`, `update_rms` |
| `weight_jumps` | Q3 | Each tensor's norm at two consecutive weight checks | `checks`, `parameters`, and `largest`: the ten largest changes, each with `ratio`, `step`, `previous_step`, `before`, `after` |

**What these measurements do not do, in the author's words of Nov 21, 2026:** B4's re-created fault is still missed, and Q1 has only synthetic coverage. Neither gap is closed by this wave.

What the development captures say of the trend, so that it is not over-read:

- It shows a sustained rise where there is one (5.2 times over B1's buggy long run).
- A healthy run can rise more than a buggy one (B14: 2.32 healthy, 1.39 buggy).
- It does not show B4's re-creation: that gradient norm climbs in three steps and then sits at 38.45, so its trend is flat (`gradient_trend_a_plateau`).
- No capture on the allowlist has the six probes `attention_trends` needs. Its case is the review's own hand-written series.

Cases: `gradient_trend_a_steady_rise`, `gradient_trend_a_plateau`, `gradient_trend_reads_the_pre_clip_norm_and_real_steps`, `gradient_trend_under_six_points`, `attention_trends_a_slow_rise`, `attention_trends_under_six_probes`, `update_trends_one_parameter_collapses`, `update_trends_absent_without_updates`, `weight_jumps_one_tensor_jumps`, `weight_jumps_one_check`.

## 7. Three keys on the wire

None changes the Arrow schema or the protocol number: the two record keys travel in `extra_json`, as `moe_refused` and `grads_not_read` do.

- **`layers_not_read`** on a record, said once on the first record written after the capture knows: a map from module name to why it could not be read. In evidence it is a `not_checked` entry with `check: "layers"`, `modules`, `reason`, `step` and `coverage`. The two coverages are different things and read differently:
  - `"partial"`: some modules were read. No kind becomes `unavailable`: the activation kinds stay `checked` for the modules read and name the others under `modules_not_judged`, as `stalled_weights` does for parameters.
  - `"none"`: no module was readable on any record. The activation kinds are `unavailable` and the entry lists them under `kinds`. Nothing was looked at, so nothing reads as clean.
  - **It describes a persistent limitation of the capture, by the author's word.** The note is said once for the run, so it names a module the capture cannot read at all. A step on which a module could not be read, in a run where it otherwise is, must not make the run or the kind `unavailable`.
  - **A limit of v1, stated:** an intermittent gap is not represented. A module that is read on some records and not on others is not named in `layers_not_read`; its records simply lack it on those steps, and it is judged on the records that hold it (`a_module_missing_on_some_records_is_judged_on_the_rest`). A reader of v1 cannot tell such a gap from a module that was not watched on those steps.
- **`moe_source`** on a record, said once: for each router, `form` and `point`.
  - `form` is how the routing reached the capture: `"scores"` (TensorQuill took the top-k of the hooked module's output), `"indices"` (the model gave the chosen experts) or `"map"` (a routing map). It describes the representation and nothing else.
  - `point` is where in the model it was taken: `"requested"` (before any capacity limit, bias or noise that follows), `"dispatched"` (after the model has finished assigning), or `"unknown"`.
  - Indices and maps can be produced before or after capacity handling, so the form never tells a reader which. **A reader says "dispatched assignments" only when `point` is `"dispatched"`**, which the capture writes only where the capture point establishes it. Scores are always `"requested"`. With `"unknown"` a reader says the assignments are as the model gave them and that it is not known whether a capacity limit came first.
- **The step capture stopped at**, on the heartbeat. A beat that carries `capture_stopped` also carries `capture_stopped_step`: the last step the session committed before it stopped, or null when it committed none. Step 0 and null are different and stay different all the way to a reader. The server's status shows it as `step` inside the rank's `capture_stopped`, beside `reason`; a beat from a sender older than the key gives no `step` key.

Cases, each from what is written on the wire to what a reader holds: `layers_not_read_some_modules`, `layers_not_read_no_module`, `a_module_missing_on_some_records_is_judged_on_the_rest`, `moe_source_scores_are_what_the_router_asked_for`, `moe_source_indices_at_an_unknown_point`, `moe_source_a_map_after_dispatch`, `moe_source_absent`, `capture_stopped_at_step_0`, `capture_stopped_before_any_step`, `capture_stopped_from_an_older_sender`.

## 8. RS10: what a refused batch does to its stream

**Approved in intent by the author. The transport is not signed.** The intent: a batch refused for good is counted once, is not sent again, and never costs rows the server could have stored. How the server tells the sender is still open, and is settled by the tests below.

One rank sends a batch the server can store (A, 3 rows, ending at byte 300), a batch whose `step` column is of another type (B, 2 rows, to 500), and another it can store (C, 3 rows, to 800), then disconnects and returns.

Two facts the run established, which shape the choice:

- **A `DoPut` stream has one schema.** A batch refused for a key column is followed, in its stream, only by batches that would be refused the same way. So keeping the stream open cannot make a later batch of it storable, and B travels in a stream of its own.
- **A new stream's first answer names no offset.** The sender restarts from the offset it saved itself. So what happens after a refusal is decided by what the sender saved.

**The sender's rule.** Its saved offset moves for two reasons only: the server confirmed it stored the bytes, or the server confirmed, naming exactly the byte range the sender sent, that it refused them for good. A refusal that is lost, or that names another range, moves nothing.

| | Today (`eb5a71b`, pinned) | Intended |
|---|---|---|
| Batches the client sends | A, then B | A, B once, then C directly |
| Acknowledged after A | 300 | 300 |
| What the server says of B | The status names `step` and no byte range | A confirmation that names bytes 300 to 500 and the column, then the same status |
| Where the sender restarts | 300: B again | 500 |
| Rows stored | A's three | A's and C's six, and only those |
| `refused_batches` | 1 | 1 |

The tests, and what each one shows today:

| Test | What it exercises | Today |
|---|---|---|
| `test_rs10_acceptance.py::test_the_run_as_the_server_answers_today` | The real server; every batch the client sent is recorded | Passes: pins the left column |
| `...::test_the_run_as_the_contract_states_it` | The real server: after a confirmed refusal of B at 300 to 500, restart at 500, C sent directly, only A and C stored | A strict expected failure until the server confirms the range |
| `...::test_the_same_refused_bytes_sent_again_are_counted_once` | The original byte range replayed three times, apart from the run | Passes |
| `test_rs10_sender.py::test_a_confirmed_refusal_moves_the_offset_to_the_end_of_the_refused_bytes` | The production `FlightSender` against a stand-in that injects the refusal | A strict expected failure until the sender reads the confirmation |
| `test_rs10_sender.py::test_a_refusal_that_is_lost_or_names_another_range_keeps_the_last_confirmed_offset` | The production sender, with no answer, and with an answer for another range | Passes, and must go on passing |

**Still open before the transport is signed.**

- Whether a Flight client reads a last message on the acknowledgement channel before the error status ends the call, **with the real server**. With pyarrow's own server, which the stand-in is, a client read the confirmation before the status in 20 of 20 tries (Nov 21, 2026). The real server is another implementation, and that is not shown for it.
- If it cannot be relied on with the real server, the same three values (start, end, column) travel in the status text, and the sender applies the same rule to them.
- The real sender cannot be made to send a batch the real server refuses, since it builds every batch from one schema. So the two halves are joined by the shared values in the two test files, not by one process pair.

**Left as it is:** when every batch of a sender is refused (a sender newer than its server), each batch costs one stream and one backoff. No rule stops the sender after several refusals in a row.

## What has cases, and what is still owed

- **56 shared cases:** the score's arithmetic (8) and over records (4), `activation_drift` (9), `moe_imbalance` (11), the four measurements (10), the learning rate and `parameters_not_judged` (4), the wire keys (10).
- **Five RS10 tests** in two files.
- **On the wave's checklist, not done:** the spike, diversity, starvation and stall thresholds are still to be read on the development runs, as measurement only.

## 9. Held out of this page

- **The per-parameter split of a pre-clipping total when groups are clipped separately (L20).** Lane AD3 is reproducing what the capture records on such a step. The rule is written after that.
- **Thresholds for the four measurements, and for spike, diversity, starvation and stall.** They need a real job.

## What changes in files that are frozen or shown

- **Golden files.** `everything.json` holds one `activation_drift` incident at step 185; it stays at that step with the same ratio (3.62), and its `affected_steps` goes from 115 to 29. (Corrected Nov 21, 2026: this note first said the ratio changes too. Lane AD8 found it does not, and the reference rule agrees.) Every incident with a `ratio` gains `score`. Every evidence file whose records hold gradient norms, update sizes, weight checks or six probes gains its measurement block.
- **Gate 1.** Not read yet. Once this page is signed the frozen rules are run over Gate 1 once, and every change to what an arm is shown is reported for the author's amendment before the final trial.
