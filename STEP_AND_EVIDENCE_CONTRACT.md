# Skipped steps, due checks and wording: what both engines say

Written Nov 20, 2026, from the author's decisions of that day (v3 plan, "The author's decisions on the wave's calls"), and approved by the author the same day with two amendments, made here: who reported a skipped update (section 1) and the last step of a rank that was started again (section 3). Three smaller rules of section 1, this page's own reading of the first amendment, were signed off by the author the same day, and the second was then amended by the author: what a caller's skip does to the scaler's streak depends on whether the SDK saw the scaler apply the update. It covers four small things that the Python detectors and the Rust server must say alike, so each is written once here, with cases, before either engine is changed. The cases are in `sdk/tests/fixtures/behaviour/cases.json`, checked by `sdk/tests/test_step_and_evidence_contract.py`.

Each case is marked. A **pin** is behaviour today's Python detectors already have; the test runs it, so it cannot move unnoticed. A **new** case is not built: its expectation is the target, in the words used here, and the lane that builds it turns it into a test of its own engine. A new case over records also states what today's Python detector answers on them, which is not the target, and the test runs that: the case cannot pass before its rule is built.

Outside this page: the checkpoint-to-model check (B14), the median-and-MAD score (S6), the cross-rank activation compare, and replica attribution, which is section 4a of `REPLICA_OBSERVATION_CONTRACT.md`.

## 1. `session.step(skipped=...)`

`session.step()` marks the end of one optimizer step under `auto_step=False`, for a framework whose stepping the hooks cannot see. It gains one argument, `skipped`, for an update that was due and did not happen. Whether an update was skipped stays under the key a loss scaler already writes, `step_skipped`. Who reported it is recorded beside it under a new key, `step_skipped_by`, because a caller's skip and a scaler's are not read alike by every kind. The Python detectors' rules marked as pins below were run for this page; the Rust detectors read the same column (`server/src/detect.rs`), seen by reading and not run here.

**What is recorded.**

| The call | The scaler given to `init` | `step_skipped` | `step_skipped_by` | Case |
|---|---|---|---|---|
| `session.step(skipped=False)` | None, or one that applied the step | `false` | No key | `an_update_due_and_applied` (new) |
| `session.step(skipped=True)` | None | `true` | `"caller"` | `an_update_due_and_skipped` (new) |
| `session.step(skipped=True)` | One the SDK observed apply this update | `true` | `"caller"`, with `scaler_applied: true` | `a_skip_only_the_caller_reported` (new) |
| `session.step(skipped=True)` | One that did not act on this update, so its outcome is unknown | `true` | `"caller"` | `a_skip_the_caller_reported_with_no_scaler_outcome` (new) |
| `session.step(skipped=False)` | One that skipped the step | `true` | `"scaler"` | `a_skip_only_the_scaler_reported` (new) |
| `session.step(skipped=True)` | One that skipped the step | `true` | `"both"` | `a_skip_both_reported` (new) |
| `session.step()` | One that skipped the step | `true` | `"scaler"` | `nothing_said_with_a_scaler` (new) |
| `session.step()` | None | No key | No key | `nothing_said_without_a_scaler` (new, today's behaviour) |
| No call | Any | No record is ended: an accumulation microstep is not a step | | `an_accumulation_microstep` (new) |

The step is skipped when either the caller or the scaler says so. `step_skipped_by` is written only on a record whose `step_skipped` is true, and it is not a column of the Arrow schema: it travels in `extra_json`, as `grads_not_read` does, so no schema changes. `skipped` takes only `True`, `False` or nothing; anything else is refused at the call, as a wrong `lr` is, and never inside training's own step.

**What the scaler did on a caller's skip.** A record whose `step_skipped_by` is `"caller"` covers two cases that `loss_scale_trouble` reads differently, so the record keeps them apart. When the SDK observed the scaler apply the update, the record also carries `scaler_applied: true`. When the scaler's outcome is unknown (there is no scaler, or it did not act on this update), the record has no such key. The key is written on no other record, and it travels in `extra_json` too.

**A capture written before this key.** A record with `step_skipped: true` and no `step_skipped_by` was written by a scaler, the only thing that could write it then, and is read as `"scaler"`. Every kind reads such a capture as it did before (`ten_skips_in_a_row_with_no_loss_scale`, `a_gradient_not_finite_on_a_skipped_step`, `an_activation_not_finite_on_a_skipped_step`, pins). Its evidence has no `skipped_updates` (`a_capture_written_before_the_key`, new).

**What may judge a skipped step.**

| Kind | A skip only the caller reported | A skip the scaler reported, alone or with the caller | Cases |
|---|---|---|---|
| `grad_spike` | Not judged, and not part of the history later steps are measured against | The same | `a_spike_on_a_skipped_step` (pin) |
| `update_spike` | The same | The same | In the engines' own tests today |
| What counts as update history | Not counted | Not counted | In the engines' own tests today |
| `stalled_weights` | Does not count toward a stall: the weights were not meant to move | The same | `weights_unchanged_across_skipped_steps`, `weights_unchanged_across_skips_the_caller_reported`, `weights_unchanged_across_skips_the_scaler_reported`, `weights_unchanged_across_skips_both_reported` (new) |
| `loss_scale_trouble` | Not counted: ten in a row do not raise it | Counted: ten in a row raise it, with or without a recorded loss scale | `ten_skips_only_the_caller_reported` (new); `ten_skips_the_scaler_reported`, `ten_skips_both_reported` (pins) |
| `non_finite`, a layer statistic | Raised | Raised | `an_activation_not_finite_on_a_skip_the_caller_reported`, `an_activation_not_finite_on_a_skip_the_scaler_reported`, `an_activation_not_finite_on_a_skip_both_reported` (pins) |
| `non_finite`, a gradient | Not raised from the gradient alone. The evidence keeps it (below) | With the scaler alone: not raised, and nothing kept, as today. With both: not raised, and kept, since the caller reported it | `a_gradient_not_finite_on_a_skip_the_caller_reported`, `a_gradient_not_finite_on_a_skip_the_scaler_reported`, `a_gradient_not_finite_on_a_skip_both_reported` (new) |

**What the evidence keeps.** One rank's evidence gains a key, `skipped_updates`, present only when a record of the run carries `step_skipped_by`, so the evidence of a capture written before the key is what it was. It counts the skipped steps by who reported them, and holds `non_finite_gradients` only when a skip the caller reported (alone or with the scaler) had a gradient norm that was not finite:

```json
"skipped_updates": {
  "caller": 2, "scaler": 0, "both": 1,
  "non_finite_gradients": {"steps": 2, "first_step": 3, "gradients": ["blk.weight"]}
}
```

`steps` is how many such steps there were, and `gradients` names what was not finite at the first of them, as a `non_finite` incident names it (a parameter, `<total>` or `<total before clipping>`). The key raises nothing, and changes neither the value `non_finite` reads nor the run's state. When a layer statistic is not finite on the same step, `non_finite` is raised for the layer with an empty `gradients`, as on any skipped step, and the gradient is kept here (`an_activation_and_a_gradient_not_finite_on_a_skip_the_caller_reported`, new). The Python reader writes the key in `build_evidence`; the server writes the same object in the rank's report beside `not_checked`, and a reader of that report passes it on unchanged. How the all-ranks view shows it is the Python lane's call, and it must not drop it.

**Three rules the amendment does not state, signed off by the author on Nov 20, 2026.** The two engines would part on each, so each is fixed here with a case, and both engines use the same words and counters:

1. *A skipped step and a stall's streak.* A skipped update cannot advance a stall's streak, whoever reported it. A streak that exists carries across it, and the usual resets still apply, as for a row whose total gradient norm is not finite. So two qualifying unchanged checks, a skip and a third qualifying check raise the stall at the third check, not at the skip (`a_stall_across_a_skipped_step`, new).
2. *A caller's skip inside a run of the scaler's.* Amended by the author after the sign-off. A skip only the caller reported keeps `step_skipped: true` and never adds a scaler-confirmed skip. What it does to the scaler's streak depends on what the SDK saw of the scaler on that update:
   - *Outcome unknown* (no `scaler_applied`): the streak is left as it is. The step is not an observation this rule judges, and adds to neither `skipped_steps` nor `recorded_steps`.
   - *The SDK observed the scaler apply the update* (`scaler_applied: true`): the streak ends, as at any recorded applied update. It adds nothing to `skipped_steps`.

   The streak is described as "10 consecutive scaler-confirmed skips among observations judged by this rule", which says that it can span more training steps than it counts. The paired cases: five scaler-confirmed skips, one caller-only skip and five more. With the middle outcome unknown the streak reaches ten and the rule is raised (`a_caller_skip_of_unknown_scaler_outcome_inside_scaler_skips`, new). With the middle one known to be applied it ends at five, and nothing is raised (`a_caller_skip_the_scaler_applied_inside_scaler_skips`, new). Three caller-only skips of unknown outcome between two fives raise it the same way, over 13 training steps (`caller_skips_inside_a_run_of_scaler_skips`, new). One count is this page's reading and awaits the author: a caller's skip the scaler was seen to apply is an observation of the scaler, so it counts in `recorded_steps` (`ten_scaler_skips_after_a_caller_skip_the_scaler_applied`, new).
3. *The loss scale on a caller's skip.* A measured `grad_scale` below the floor on a skip only the caller reported raises `loss_scale_trouble` with reason `scale`: the low measurement is the evidence, whoever reported the skip. That step still adds nothing to the scaler's skip counts (`a_low_loss_scale_on_a_skip_only_the_caller_reported`, pin).

A step with `step_skipped: false`, or with no key, is judged by every kind as today (`a_spike_on_an_applied_step`, `a_spike_with_nothing_said`, pins).

**What an adapter must establish.** The argument is for a recorded optimizer-update boundary.

- Megatron-LM returns `update_successful` from its optimizer at that boundary: pass `skipped=not update_successful`. That is a caller's report, also when Megatron's own loss scaler caused the skip, so such skips never raise `loss_scale_trouble`; a scaler given to `init` is the only thing that does.
- DeepSpeed's `was_step_applied()` is false at an ordinary accumulation microstep as well as at a skipped update. An adapter decides first, from its own count of microsteps, whether an update was due, and calls `session.step` only then.
- accelerate's `optimizer_step_was_skipped` is true when any tracked optimizer skipped. With one optimizer it is the answer; with several it cannot say that every one skipped, and an adapter must not pass it as `skipped`.

## 2. A check that is due, unread gradients, and a rank with no peer

**Due.** A cross-rank comparison is due for a step once any rank has recorded its part for that step. Before that the kind reads as the ranks gave it, and a run is not degraded for a check that has not come round yet. Once a comparison is due and none was made, that must be visible: the kind reads `unavailable`, with the ranks that are missing. For replica observations this is built (lane Z2: no step entry leaves the kind as it is; a step entry with nothing compared is `unavailable` with `check: "replica_coverage"`). The cases that pin it are in `sdk/tests/test_lane_z2_reader.py`: `test_a_lone_rank_file_of_a_larger_group_does_not_read_as_checked`, `test_a_step_that_compared_nothing_does_not_read_as_checked` and `test_a_servers_result_with_a_rank_missing_does_not_read_as_checked`. No engine changes for this.

**Unread gradients.** A parameter whose gradient could not be read is named in `grads_not_read` on a record, and the evidence has a `not_checked` entry with `check: "gradients"` naming each one, with why and the first step. What that does to each kind is decided by what the kind is judged from, and a finding on what was read always stands (it already does: run on today's Python, each of the three cases below raises its incident, and all three kinds read `unavailable`):

| Kind | Judged from | With an unread gradient | Case |
|---|---|---|---|
| `non_finite` | Layer statistics and each gradient norm that was read | `checked`. A value that is not finite in anything that was read fires | `a_nan_layer_beside_an_unread_gradient` (new) |
| `grad_spike` | The total gradient norm | `unavailable`, since the total leaves the unread parameter out (by `record_grads`' own account of how it reads the rest of a group; not measured here). A spike in the total as read still fires | `a_spike_beside_an_unread_gradient` (new) |
| `stalled_weights` | Each parameter's own history | `checked` for the parameters read; an unread parameter is not judged and is named | `a_stall_beside_an_unread_gradient` (new) |

The run still reads `degraded` for the `not_checked` entry, as any such entry makes it. This replaces lane Z2's call, which made all three kinds `unavailable`.

**A rank with no peer.** One rank with batch hashes has no rank to be compared with. The server's `ranks_compared` lists a rank only when it was compared with at least one other, so for a lone rank it is empty, and the kind `duplicate_batches` reads `not_configured` with the words "one rank observed, zero peers compared". The same words are used for replica fingerprints (`REPLICA_OBSERVATION_CONTRACT.md`, section 4a). Case: `one_rank_with_batch_hashes` (new).

## 3. A rank that went silent after its capture had stopped

Contract 3 (v3 plan, Nov 19, 2026) has a rank whose capture stopped while training went on read `capture_stopped`, and the author kept the rule that such a rank raises `rank_silent` or `job_stalled` if its heartbeats later cease. What was missing is the link between the two. The incident keeps its kind and gains two keys for such a rank: `capture_stopped`, the value of that key in the rank's status entry, unchanged (today a `reason` such as `sink_error` and `seconds`, how long ago), and `capture_last_step`, the last step stored by the rank's current attempt, or null when that attempt stored none. Its text then says that capture had stopped after that step, for that reason, and that TensorQuill cannot tell from this rank whether training itself stalled. A rank whose capture never stopped gets neither key.

**After a restart.** A step an earlier attempt stored is not the current attempt's. When the job was started again and the new attempt's capture stopped before its first record, `capture_last_step` is null, and the text says capture had stopped before this attempt stored a step, for that reason. It names no step, and above all not the earlier attempt's last one. The server's status already works this way, by reading and not run here: a new attempt clears the rank's `last_step` and its `capture_stopped` (`take_attempt`, `server/src/watchdog.rs`), shows `attempt_without_steps`, and a heartbeat of the new attempt can then say its capture stopped. The case pins a previous attempt that reached step 100 (`silent_after_a_restart_that_stored_nothing`, new); its incident is whichever of the two kinds the silence raises.

The status does not carry the step at which capture stopped apart from the last step stored; if the two can differ, saying so needs the heartbeat to carry it, which is the SDK's and is not asked for here. Cases: `silent_after_capture_stopped`, `silent_with_capture_running` (new).

## 4. A checkpoint whose model state cannot be identified

`resume_incomplete` lists, for each component of a checkpoint, a `finding`: `missing` reads "not in the checkpoint". Which entry of a checkpoint is the model's state is found by a rule over names and keys, and the rule can fail to find one that is there. So for the model component alone, when no entry is identified as the model's state, the finding is `not_identified`, and it reads "model state could not be identified". `missing` is kept for a comparison that knows what the checkpoint should hold, which is B14's and is not built. The other components keep `missing`. Cases: `a_checkpoint_with_no_identified_model_state`, `a_checkpoint_with_no_optimizer_state` (new).

## 5. Who builds what

| Section | SDK | Python detectors | Rust server |
|---|---|---|---|
| 1 | `capture.py`: the argument, and `step_skipped_by` and `scaler_applied` on the record | `stalled_weights`, `loss_scale_trouble`, the evidence's `skipped_updates`, and the labels that say a skip is the scaler's | The same, reading `step_skipped_by` and `scaler_applied` from `extra_json` |
| 2, unread gradients | Nothing | The three kinds' entries | Reading `grads_not_read`, which it does not do today, and the same entries |
| 2, a rank with no peer | Nothing | The words | `ranks_compared` |
| 3 | Nothing | The incident's two keys and text | The same keys in its incident |
| 4 | Nothing | The finding and its words | The same finding |

## 6. Not settled here

- How the all-ranks view shows `skipped_updates`.
- Whether a caller's skip the scaler was seen to apply counts in `recorded_steps` (section 1, rule 2).
- Whether the first step of a comparison, while its last rank's record is still on its way, should be given a short grace before the kind reads `unavailable`. Today it is visible at once and clears when the record arrives.
- Whether the heartbeat should carry the step at which capture stopped, where it can differ from the last step stored.
