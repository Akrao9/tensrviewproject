# Detector rules contract, wave AD

**Draft for the author's signature, Nov 21, 2026. Nothing here is built.** It is on the branch `contracts/wave-ad`, cut from `eb5a71b`. Both engines (the Python detectors in `sdk/src/tensorquill/evidence.py` and the server in `server/src/detect.rs`) build from this page and from the cases in `sdk/tests/fixtures/rules/cases.json`, and must give each case's `expected`.

What was measured before any rule was proposed is in `experiments/rule-selection/RESULTS.md`. Every input to that is on `experiments/rule-selection/allowlist.json`. Gate 1's six held-out cases were not read, and are not read until this page is signed.

## What the author is asked to decide

| # | Decision | Proposed |
|---|---|---|
| 1 | The `activation_drift` rule | A trailing baseline that never uses a module's first 50 records (part 2) |
| 2 | The `moe_imbalance` rule for a router that has not been trained | The state a router starts in is excused for up to 10 samples (part 3) |
| 3 | The least number of earlier finite values for a score | 10 (part 1) |
| 4 | What a reader shows when the MAD is zero | No score, with the words "the earlier values did not vary" (part 1) |
| 5 | The four measurements and their names | `gradient_trend`, `attention_trends`, `update_trends`, `weight_jumps` (part 6) |
| 6 | What a refused batch does to its stream, by RS10's acceptance run | The server says which bytes it refused for good, then ends the stream; the sender goes on from their end (part 8) |
| 7 | Three keys on the wire | `layers_not_read`, `moe_source`, and the step on `capture_stopped` (part 7) |

## 1. The score beside a ratio (S6)

Wherever an incident records a `ratio` that was taken against a window of earlier values, it also records a `score`: how far the value is from the window's median, in units of the window's spread.

- **The arithmetic.** `score = (value - median) / (1.4826 * MAD)`. The median is the rules' own (`_median`: the mean of the two middle values of an even count). The MAD is the median of the absolute distances from that median. Both are taken over the same earlier values the ratio's baseline was taken over, and never over the value itself.
- **How many earlier values it needs.** At least 10 finite values in the window. With fewer the score is `null` and `score_note` is `"short_history"`. A value that is not a finite number does not count.
- **A zero MAD.** When more than half the window is the same value the MAD is 0. Nothing is divided by it: the score is `null` and `score_note` is `"no_dispersion"`.
- **What a reader shows.** A number to one decimal place ("score 31.2"). For `short_history`: "no score: fewer than 10 earlier values". For `no_dispersion`: "no score: the earlier values did not vary". A reader never shows a null score as 0 and never leaves the line out.
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

Cases: `score_above`, `score_below_is_negative`, `score_ten_earlier_values`, `score_nine_earlier_values`, `score_values_that_are_not_finite_do_not_count`, `score_no_dispersion`, `score_all_equal`.

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
- **One fresh run is the check.** A from-scratch MoE job of another shape, not used to choose the rule, is run once against this rule as written. Its result is in `RESULTS.md` under "Validation".

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
- Once it reaches 10 samples it is reported, dated at its first sample, with `since_start: true`.
- Any flagged sample after the leading run has ended is reported as today.
- A router whose first sample is not at step 0 is judged as today.

**Measured.** Not judging step 0 alone moves nanoMoE's incident to step 1. A rule that asks for three flagged samples in a row misses the practice fixture's overload, which is on the last two of four sampled steps. This rule is quiet on nanoMoE with its balance loss, fires at step 7 on another router without it, and reports every seeded collapse at the step and router today's rule does. A limit of 5 or of 50 gives the same answers on these inputs.

**While a router is still in its starting state.** A short capture must not read as checked and clean. When the records end with a router flagged on every sample since step 0 and fewer than 10 of them, nothing is reported for it, and the `moe_imbalance` entry of `kind_checks` carries `starting_state`: a list of `{"router", "samples"}`. With that key present `enough_history` is false, so the kind reads as checked without enough history (`start_and_the_capture_ends`).

**The public function.** `first_incident()` follows this rule, since `build_evidence` and the Rerun export both take the incident from it. The per-step samples it reads are unchanged.

**What it gives up.** A router collapsed from its first step is reported 10 samples later than today.

Cases: `start_two_samples_then_even`, `start_nine_samples_then_even`, `start_ten_samples_is_reported`, `start_then_a_later_overload`, `start_ends_and_comes_back`, `not_at_step_0_is_not_a_start`, `start_in_a_sparse_capture`, `one_router_starts_another_fails`, `a_healthy_router`, `start_and_the_capture_ends`.

## 4. The learning rate on `update_spike`

The incident gains `lr`, the record's `lr` list at the flagged step, and `lr_before`, the same at the step it was compared with. Each is present only when that record carries `lr`. Nothing about when the kind fires changes. The engine lanes hold both engines to the same two keys on the existing `update-size-a-jump` record set.

## 5. `parameters_not_judged`

Today the Python detectors name the parameters whose gradient was not read on the `stalled_weights` entry of `kind_checks`, and the server writes no such key.

- Both engines write `parameters_not_judged`, a sorted list of names, on the `stalled_weights` entry of one rank's `kind_checks`, when a `gradients` entry of `not_checked` lists them and the kind is not itself `unavailable`.
- The all-ranks view's `stalled_weights` entry carries the sorted union under the same key, and `parameters_not_judged_by_rank` (rank to list) when the ranks' lists differ.

## 6. Four measurements, with no incident and no threshold

Each is a block of numbers in one rank's evidence, as `weight_trends` is. It is present when the records hold what it reads and absent otherwise. No rule reads it, it changes no run state, and no line is drawn on it in this wave.

The first two are the robust local trend of the sibling repository LLM-Training-Debugger (`robust_local_trend`), written again from its description: a line through the median of the first third of the series and the median of its last third, each placed at the median of its own step numbers. It uses the actual step numbers. A series under six points has no trend.

| Block | Gap | What it reads | What it records |
|---|---|---|---|
| `gradient_trend` | B4 | The total gradient norm (the pre-clipping norm when the run clips), skipped steps left out | `series`, `points`, `first_step`, `last_step`, `first`, `last`, `growth` (last over first), `per_100_steps` (the slope as a share of `first`) |
| `attention_trends` | Q1 | Each attention module's largest logit, probe by probe | `statistic`, `modules`, and `listed`: the ten with the largest `growth`, each with the fields above |
| `update_trends` | Q2 | Each parameter's `update_rms` against the median of its own 50 records before, from 10 earlier values on | `parameters`, `judged`, and `lowest`: the ten parameters whose update fell furthest, each with `ratio`, `step`, `baseline`, `update_rms` |
| `weight_jumps` | Q3 | Each tensor's norm at two consecutive weight checks | `checks`, `parameters`, and `largest`: the ten largest changes, each with `ratio`, `step`, `previous_step`, `before`, `after` |

What the development captures say of the trend, so that it is not over-read:

- It shows a sustained rise where there is one (5.2 times over B1's buggy long run).
- A healthy run can rise more than a buggy one (B14: 2.32 healthy, 1.39 buggy).
- It does not show B4's re-creation: that gradient norm climbs in three steps and then sits at 38.45, so its trend is flat (`gradient_trend_a_plateau`).
- No capture on the allowlist has the six probes `attention_trends` needs. Its case is the review's own hand-written series.

Cases: `gradient_trend_a_steady_rise`, `gradient_trend_a_plateau`, `gradient_trend_reads_the_pre_clip_norm_and_real_steps`, `gradient_trend_under_six_points`, `attention_trends_a_slow_rise`, `attention_trends_under_six_probes`, `update_trends_one_parameter_collapses`, `update_trends_absent_without_updates`, `weight_jumps_one_tensor_jumps`, `weight_jumps_one_check`.

## 7. Three keys on the wire

None changes the Arrow schema or the protocol number: the two record keys travel in `extra_json`, as `moe_refused` and `grads_not_read` do.

- **`layers_not_read`** on a record, said once on the first record written after the capture knows: a map from module name to why it could not be read. In evidence it is a `not_checked` entry with `check: "layers"`, `modules`, `reason` and `step`. No kind becomes `unavailable` for it: the activation kinds stay `checked` for the modules read and name the others under `modules_not_judged`, as `stalled_weights` does for parameters.
- **`moe_source`** on a record, said once: a map from router name to `"scores"` (TensorQuill took the top-k of the hooked module's output), `"indices"` (the model gave the chosen experts) or `"map"` (a routing map). In evidence it is copied as recorded under `moe_source`, and a MoE incident on a `"scores"` router says the counts are what the router asked for, which a capacity limit, a bias or noise after the hooked module can change.
- **The step capture stopped at**, on the heartbeat. A beat that carries `capture_stopped` also carries `capture_stopped_step`, the last step the session committed before it stopped, or null when it committed none. The server's status shows it as `step` inside the rank's `capture_stopped`, beside `reason`.

## 8. RS10: what a refused batch does to its stream

The acceptance run is `sdk/tests/test_rs10_acceptance.py`, written against the real server before this part is signed. One rank sends a batch the server can store (A, 3 rows, ending at byte 300), a batch whose `step` column is of another type (B, 2 rows, to 500), and another it can store (C, 3 rows, to 800), then disconnects and returns.

Two facts the run established, which shape the choice:

- **A `DoPut` stream has one schema.** A batch refused for a key column is followed, in its stream, only by batches that would be refused the same way. So "keep the stream open" cannot make a later batch of that stream storable, and B travels in a stream of its own.
- **A new stream's first answer names no offset.** The sender restarts from the offset it saved itself. So what happens after a refusal is decided by what the sender saved.

| | Today (`eb5a71b`, pinned by the test) | Proposed |
|---|---|---|
| Acknowledged after A | 300 | 300 |
| What the server says of B | Ends the stream with `INVALID_ARGUMENT` naming `step`. No answer first | First an answer on the acknowledgement channel: `{"end_byte_offset": 300, "refused": {"end_byte_offset": 500, "column": "step"}}`. Then the same status ends the stream |
| The sender's restart byte before C | 300: it sends B again, and again | 500: it counts B as refused for good and saves 500 |
| Acknowledged after C | Nothing: C waits behind B for as long as B is refused | 800 |
| Rows stored | A's three | A's and C's six |
| `refused_batches` | 1, though B was sent twice | 1 |
| The sender's restart byte after the run | 300 | 800 |

- **Why the stream still ends:** by the first fact, nothing later in it could be stored.
- **Why the sender goes on:** the author's decision of Nov 20, 2026 is that such a batch "is refused and counted". Today it is counted and then sent again without end, and every valid batch behind it waits.
- **What is lost:** B's rows, which no reader could have read. Both sides count them: the server in `refused_batches`, the sender in a count of its own that `close()` and its last error report.
- **What must not happen:** a refusal skips rows the server could have stored. The sender moves its offset only to the end the server named, and only on an answer that carries `refused`.
- **Left as it is:** when every batch of a sender is refused (a sender newer than its server), each batch costs one stream and one backoff. This wave sets no rule to stop sending after several refusals in a row.

**What the acceptance run does not cover.**

- **It does not use the real sender.** TensorQuill's sender builds every batch from one schema, so it cannot send a batch the server refuses for a key column. The run's client is a raw Flight client that follows the sender's rules as this part states them. The real sender's handling of the answer is lane AD4's, tested against the fake server in `sdk/tests/flight_fake_server.py`.
- **"An answer, then the status" is not yet shown to arrive.** Whether a Flight client reads a last message on the acknowledgement channel before the error status ends the call was not tried. Lanes AD2 and AD4 show it first. If it does not arrive, the status text carries the same two values (the refused end offset and the column) and the sender reads them from there; the numbers in the table stay as they are.

The alternative, if the author prefers it: change nothing on the wire, and have the sender keep its offset and retry only at its backoff ceiling. Then C never arrives while B is refused, and nothing valid is ever skipped.

## What has cases and what does not

- **With shared cases (36):** the score's arithmetic (7), `activation_drift` (9), `moe_imbalance` (10), the four measurements (10).
- **With an acceptance test:** RS10.
- **Prose only, so far:** the learning rate on `update_spike` (part 4), `parameters_not_judged` (part 5, where "the run-level check" is the run-level part of the all-ranks view, `combine_rank_evidence`), and the three wire keys (part 7).
- **Still to add before the engine lanes start:** a records-level case of the score for `grad_spike`, whose two parameter paths take different baselines, and for `activation_spike`. Without them the two engines would first meet on the score's windows at the merge.

## 9. Held out of this page

- **The per-parameter split of a pre-clipping total when groups are clipped separately (L20).** Lane AD3 is reproducing what the capture records on such a step. The rule is written after that.
- **Thresholds for the four measurements, and for spike, diversity, starvation and stall.** They need a real job.

## What changes in files that are frozen or shown

- **Golden files.** `everything.json` holds one `activation_drift` incident at step 185; it stays at that step with another ratio and count. Every incident with a `ratio` gains `score`. Every evidence file whose records hold gradient norms, update sizes, weight checks or six probes gains its measurement block.
- **Gate 1.** Not read yet. Once this page is signed the frozen rules are run over Gate 1 once, and every change to what an arm is shown is reported for the author's amendment before the final trial.
