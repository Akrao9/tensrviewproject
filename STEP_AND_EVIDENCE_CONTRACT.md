# Skipped steps, due checks and wording: what both engines say

Written Nov 20, 2026, from the author's decisions of that day (v3 plan, "The author's decisions on the wave's calls"). It covers four small things that the Python detectors and the Rust server must say alike, so each is written once here, with cases, before either engine is changed. The cases are in `sdk/tests/fixtures/behaviour/cases.json`, checked by `sdk/tests/test_step_and_evidence_contract.py`.

Each case is marked. A **pin** is behaviour today's Python detectors already have; the test runs it, so it cannot move unnoticed. A **new** case is not built: its expectation is the target, in the words used here, and the lane that builds it turns it into a test of its own engine.

Outside this page: the checkpoint-to-model check (B14), the median-and-MAD score (S6), the cross-rank activation compare, and replica attribution, which is section 4a of `REPLICA_OBSERVATION_CONTRACT.md`.

## 1. `session.step(skipped=...)`

`session.step()` marks the end of one optimizer step under `auto_step=False`, for a framework whose stepping the hooks cannot see. It gains one argument, `skipped`, for an update that was due and did not happen. The record key is the one a loss scaler already writes, `step_skipped`, so the detectors read a caller's skip as they read a scaler's. The Python detectors' rules below were run for this page; the Rust detectors read the same column (`server/src/detect.rs`), seen by reading and not run here.

**What is recorded.**

| The call | What it means | `step_skipped` on the step's record |
|---|---|---|
| `session.step(skipped=False)` | An update was due and was applied | `false` (`an_update_due_and_applied`, new) |
| `session.step(skipped=True)` | An update was due and was skipped | `true` (`an_update_due_and_skipped`, new) |
| No call | No update was due: an accumulation microstep is not a step | No record is ended (`an_accumulation_microstep`, new) |
| `session.step()` | Not said | As today: the scaler's verdict when `init` was given `scaler=`, and no key otherwise (`nothing_said_with_a_scaler`, `nothing_said_without_a_scaler`, new as cases, today's behaviour) |

With both a scaler and the argument, the step is skipped when either says so. `skipped` takes only `True`, `False` or nothing; anything else is refused at the call, as a wrong `lr` is, and never inside training's own step.

**What may judge a skipped step.**

| Kind | A step with `step_skipped: true` | Case |
|---|---|---|
| `grad_spike` | Not judged, and not part of the history later steps are measured against | `a_spike_on_a_skipped_step` (pin) |
| `update_spike` | The same | In the engines' own tests today |
| What counts as update history | Not counted | In the engines' own tests today |
| `non_finite` | Only layer statistics are read; a gradient that is not finite is what a skip is for | `a_gradient_not_finite_on_a_skipped_step`, `an_activation_not_finite_on_a_skipped_step` (pins) |
| `loss_scale_trouble` | Counted: ten skips in a row raise it, with or without a recorded loss scale | `ten_skips_in_a_row_with_no_loss_scale` (pin) |
| `stalled_weights` | Does not count toward a stall: the weights were not meant to move | `weights_unchanged_across_skipped_steps` (new) |

A step with `step_skipped: false`, or with no key, is judged by every kind as today (`a_spike_on_an_applied_step`, `a_spike_with_nothing_said`, pins).

**What an adapter must establish.** The argument is for a recorded optimizer-update boundary.

- Megatron-LM returns `update_successful` from its optimizer at that boundary: pass `skipped=not update_successful`.
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

Contract 3 (v3 plan, Nov 19, 2026) has a rank whose capture stopped while training went on read `capture_stopped`, and the author kept the rule that such a rank raises `rank_silent` or `job_stalled` if its heartbeats later cease. What was missing is the link between the two. The incident keeps its kind and gains two keys for such a rank: `capture_stopped`, the value of that key in the rank's status entry, unchanged (today a `reason` such as `sink_error` and `seconds`, how long ago), and `capture_last_step`, the rank's `last_step`, which is the last step stored before capture stopped. Its text then says that capture had stopped after that step, for that reason, and that TensorQuill cannot tell from this rank whether training itself stalled. A rank whose capture never stopped gets neither key. The status does not carry the step at which capture stopped apart from the last step stored; if the two can differ, saying so needs the heartbeat to carry it, which is the SDK's and is not asked for here. Cases: `silent_after_capture_stopped`, `silent_with_capture_running` (new).

## 4. A checkpoint whose model state cannot be identified

`resume_incomplete` lists, for each component of a checkpoint, a `finding`: `missing` reads "not in the checkpoint". Which entry of a checkpoint is the model's state is found by a rule over names and keys, and the rule can fail to find one that is there. So for the model component alone, when no entry is identified as the model's state, the finding is `not_identified`, and it reads "model state could not be identified". `missing` is kept for a comparison that knows what the checkpoint should hold, which is B14's and is not built. The other components keep `missing`. Cases: `a_checkpoint_with_no_identified_model_state`, `a_checkpoint_with_no_optimizer_state` (new).

## 5. Who builds what

| Section | SDK | Python detectors | Rust server |
|---|---|---|---|
| 1 | `capture.py`: the argument and the record | `stalled_weights` | Its twin of `stalled_weights` |
| 2, unread gradients | Nothing | The three kinds' entries | Reading `grads_not_read`, which it does not do today, and the same entries |
| 2, a rank with no peer | Nothing | The words | `ranks_compared` |
| 3 | Nothing | The incident's key and text | The same key in its incident |
| 4 | Nothing | The finding and its words | The same finding |

## 6. Not settled here

- Whether `session.step(skipped=True)` on a record that also holds a gradient norm that is not finite should say so anywhere but in the skip count. Today nothing does, by the scaler's rule.
- Whether the first step of a comparison, while its last rank's record is still on its way, should be given a short grace before the kind reads `unavailable`. Today it is visible at once and clears when the record arrives.
- Whether the heartbeat should carry the step at which capture stopped, where it can differ from the last step stored.
