# A checkpoint's tensors against the model's: what the SDK records and what both engines say

Written Nov 20, 2026, for the author's sign-off. **Nothing on this page is built.** It is TrainCheck's B14: a checkpoint that left two LSTM tensors out, a resume that loaded it with `strict=False`, and no finding, because `resume_incomplete` looks only at which components a checkpoint holds (`experiments/traincheck-bugs/README.md`). The page is written once, with cases, before either engine changes, as the other two contract pages were. The cases are in `sdk/tests/fixtures/checkpoint/cases.json`, checked by `sdk/tests/test_checkpoint_tensor_contract.py`.

Each case is marked. A **pin** is what the code already answers, and the test runs it. A **new** case is not built: its expectation is the target, and the case also states what the code answers today, which the test runs, so the case cannot pass before its rule is built. The lane that builds a rule turns the case into a test of its own engine.

## Awaiting the author

Each of these is written into the page as proposed, with cases. A different answer changes the cases named beside it.

1. **Which missing tensors raise.** Proposed: only trainable parameters. A frozen parameter or a buffer that is not in the checkpoint is counted and named in the evidence and raises nothing, because an adapter-only checkpoint leaves the frozen base out on purpose and the record cannot tell that from a loss. The cost: BatchNorm statistics lost at a resume are named but not raised. (`an_adapter_only_checkpoint`, `buffers_left_out`.)
2. **Shapes.** Proposed: a name both hold with another shape is raised as `shape_differs`. PyTorch's own `load_state_dict` refuses such a tensor even with `strict=False`, so it reaches a running job only through a loader that drops what does not fit; then it was not restored. The data type is not compared. (`a_shape_that_differs`.)
3. **Unexpected tensors.** Proposed: a tensor the checkpoint holds and the model does not is named in the evidence and never raised. (`tensors_the_model_does_not_have`.)
4. **Which entry is the model's, at a load.** Proposed: the entry whose names are the model's, in place of today's rule over names and keys (section 2). It closes both of L23's known limits and reads a bare state dict, which today is not taken for the model at all. It needs one number, chosen here and not measured: at least half of the entry's names must be the model's.
5. **A save.** Proposed: not compared in this wave. A save has the model too, and comparing there would name the left-out tensors before any resume, but nothing is raised at a save today and that would be a new kind.
6. **Lightning.** `lightning.py` builds its load manifest before the session exists, so it must pass the module it is given. Proposed: add `lightning.py` to lane AC1's files. Without that, a Lightning load carries no `tensors` and nothing is said of its tensors.
7. **The class.** `resume_incomplete` is an observation ("a component left out or reset on purpose looks the same"). Proposed: it stays one in this wave, also when a trainable tensor is verified missing.

## 1. Where the comparison is made

Only the training process has the live model, so the SDK compares and records, and the two engines only read the record.

- **When.** Inside `session.checkpoint(state, event="load")`, once, before the first step of the resumed attempt. Nothing is added to a training step. A save is not compared (`a_save_is_not_compared`, a pin).
- **What is read.** The names and shapes of the model's `state_dict()`, which leaves out buffers that are not saved; which parameters are trainable (`requires_grad`) at the moment of the call; and the names and shapes of the checkpoint entry's tensors. No tensor's values are read and nothing waits on a device.
- **The model** is the one given to `tensorquill.init`.
- **Nothing is raised into training.** If reading the model's names raises, the load is recorded as it is today with `"tensors": {"state": "unavailable", "why": "error"}` (`the_comparison_raises`).
- **Public API only**, so the same on PyTorch 2.6.
- **Not measured:** the time of the call on a model of many thousands of tensors. The lane that builds it measures it and reports the number.

## 2. Which entry is compared

The candidates are every mapping in the checkpoint that holds tensors under string keys, at the top level or one level down (as `checkpoint_manifest` already looks), and the checkpoint itself when tensors sit directly in it (`a_bare_state_dict`). An entry already recognised as an optimizer's, a scheduler's or a scaler's state is not a candidate.

1. **Wrappers.** A leading `module.` (DDP) or `_orig_mod.` (`torch.compile`), repeated or not, is removed from every name on both sides (`saved_from_a_ddp_wrapped_model`, `saved_from_a_compiled_model`).
2. **One prefix.** If fewer than half of an entry's names are then the model's, one leading dotted path is tried: each leading path of the entry's names (`net.`, `net.lstm.`) removed from the entry's names that start with it, and each leading path of the model's names removed from the model's. The removal that leaves the most names in common is kept when it brings the share to at least half. The path removed is recorded as `prefix` (`the_session_watches_an_inner_module`, `the_checkpoint_was_saved_from_an_inner_module`).
3. **The entry** is the candidate with the most names in common with the model. A tie goes to the entry today's name rule takes for the model, then to the first in the checkpoint's order (`an_ema_copy_beside_the_model`, `a_cache_beside_the_model`).
4. **Identified or not.** The entry is the model's state when at least half of its names are the model's (`exactly_half_the_names_are_the_models`). Otherwise no entry is identified: the names follow another scheme, or it is another model's state, and naming every tensor of the model as missing would be a guess (`a_checkpoint_of_another_model`, `names_that_mostly_differ`, `a_cache_with_parameter_like_names`).

At a load, this decides whether the manifest's `components` holds `model`. So two answers change from today's, both L23's known limits: a model with unusual names under a plain key is now found (`unusual_names_under_a_plain_key`), and a cache with parameter-like names and no model beside it is no longer taken for the model. `checkpoint_manifest` called without a model, as for a save, keeps today's rule and today's tests.

## 3. What the load's manifest carries

One new key, `tensors`, on the manifest of a load. It rides in `checkpoint_json` with the rest of the manifest; the record's schema does not change.

Compared:

```json
"tensors": {"state": "compared", "entry": "model", "model": 7, "trainable": 7, "checkpoint": 5, "matched": 5,
            "missing": {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}}
```

| Key | Meaning |
|---|---|
| `entry` | The checkpoint key compared; null for a bare state dict |
| `model`, `trainable` | How many tensors the model's `state_dict()` has, and how many of them are trainable parameters |
| `checkpoint`, `matched` | How many tensors the entry holds, and how many names both hold |
| `missing` | Trainable parameters not in the entry |
| `missing_frozen`, `missing_buffers` | Frozen parameters and buffers not in the entry |
| `unexpected` | Tensors in the entry that the model does not have |
| `shape_differs` | Names both hold with different shapes, and under `shapes` both shapes of each |
| `prefix` | The one prefix removed, and from which side |

A list is present only when it is not empty. Each has `count`, the whole number, and `names`, sorted, at most 50 (`more_missing_than_are_named`). Names are written as the model has them, wrappers removed.

Not compared:

| `tensors` | When | Case |
|---|---|---|
| `{"state": "unavailable", "why": "not_identified"}` | No entry is identified (section 2). `closest` names the best candidate and its counts when one shares any name. `components` holds no `model` | `a_checkpoint_of_another_model` |
| `{"state": "unavailable", "why": "folder"}` | `session.checkpoint_files`: file names only, no file opened. Hugging Face's `Trainer` and Accelerate record their checkpoints this way | `a_checkpoint_folder` |
| `{"state": "unavailable", "why": "error"}` | The comparison raised | `the_comparison_raises` |
| no key | A capture written before this page, a save, or a load recorded by an adapter that passes no model | `a_resume_recorded_before_this_check` |

## 4. What both engines say at a resume

`resume_incomplete` is judged where it is today: at the first step of a later attempt that starts after the run's first step. The engines read `tensors` from the load's manifest only. A resume judged from a save, because no load was recorded, reads no `tensors` even if the save carries one (`a_resume_judged_from_a_save`, a pin).

**Findings.** Today's findings are computed as today, from `components` and the rest, in today's order. Then, from `tensors` with `state` `compared`:

| List | Finding | Case |
|---|---|---|
| `missing` | `{"component": "model", "finding": "missing_tensors", "count": 2, "tensors": [...]}` | `a_resume_with_two_tensors_verified_missing` |
| `shape_differs` | `{"component": "model", "finding": "shape_differs", "count": 1, "tensors": [...]}` | `a_resume_with_a_shape_that_differs` |

`count` and `tensors` are the list's `count` and `names`, unchanged. A list is read only when its `count` is a positive whole number (`a_missing_list_with_no_count`); a `tensors` that is not an object is ignored (`a_tensors_entry_that_is_not_an_object`). `missing_frozen`, `missing_buffers` and `unexpected` add no finding. Beside another finding, the tensors come last (`missing_tensors_beside_a_missing_component`).

**The three readings the author asked to keep apart**, and the two beside them:

| What the load recorded | `resume_incomplete` | The evidence's `checkpoint_tensors` | Case |
|---|---|---|---|
| Compared, a trainable tensor missing | `missing_tensors`, naming each | `compared`, with the lists | `a_resume_with_two_tensors_verified_missing` |
| No entry identified as the model's | `not_identified`, as today, once | `unavailable`, `not_identified` | `a_resume_whose_model_was_not_identified` |
| A folder | Nothing from tensors; the folder's own findings as today | `unavailable`, `folder` | `a_resume_from_a_folder`, `a_resume_from_a_folder_with_no_optimizer_file` |
| Compared, every trainable tensor there | Nothing | `compared`, with what was counted | `a_resume_with_every_tensor_verified` |
| No `tensors` | Nothing from tensors | No entry | `a_resume_recorded_before_this_check` |

So silence from `resume_incomplete` means every tensor was verified only where `checkpoint_tensors` says `compared` for that resume. For a folder, and for a load with no `tensors`, the evidence does not say the tensors are there.

**The evidence block.** `checkpoint_tensors` is a top-level key of a rank's evidence, present only when at least one judged resume has a `tensors` object with `state` `compared` or `unavailable`:

```json
"checkpoint_tensors": {"resumes": 1, "entries": [{"step": 10, "rank": 0, "attempt": 2000, "state": "compared",
    "entry": "model", "model": 7, "trainable": 7, "checkpoint": 5, "matched": 5,
    "missing": {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}}]}
```

An entry carries the step, rank and attempt of the resume, then `state`, `why`, `entry`, `model`, `trainable`, `checkpoint` and `matched` where the record has them, and each of the five lists as its `count` and `names`. `resumes` counts them all; `entries` lists the first five by step, as `resume_incomplete` lists its changes. The all-ranks view carries each judged rank's block with its rank, as it does `skipped_updates`. Cases for a list that raises nothing: `a_resume_with_only_frozen_tensors_missing`, `a_resume_with_only_buffers_missing`, `a_resume_with_only_unexpected_tensors`; for an error: `a_resume_whose_comparison_raised`; for the cap: `a_resume_with_more_missing_than_named`.

**Words.** In the incident's sentence, after the component's name: `missing_tensors` reads "2 trainable tensors not in the checkpoint: lstm.bias_hh_l0, lstm.weight_hh_l0", and `shape_differs` reads "1 tensor with another shape in the checkpoint: head.weight". A sentence names at most five and then says "and 55 more"; the finding keeps every name it was given.

**What does not change.** No existing finding, sentence, kind label (`needs`) or note changes. `resume_incomplete` stays `checked` on what it checks today, and the run state is not changed by a `tensors` that is `unavailable`. Every new key is present only when the record carries `tensors`, so the evidence of a capture written before this page is byte for byte what it is today, Gate 1's shown files among them.

## 5. What this does not see

- **A tensor that is in the checkpoint and was never loaded into the model.** Names are compared, not values.
- **A checkpoint folder.** Hugging Face's `Trainer` and Accelerate resume from one, and its tensors are not read. Reading the names from a `safetensors` header would open a file, which `checkpoint_files` promises not to do; not proposed here.
- **A load that is not recorded,** and a first load at the run's first step, which is not a resume and is not judged today either.
- **A parameter frozen after the call.** It is counted as trainable, so it is raised if missing.
- **An entry with fewer than half its names the model's.** Nothing is named as missing; the finding is `not_identified`.

## 6. Who builds what

| Part | Lane AC1, Python | Lane AC2, Rust |
|---|---|---|
| Sections 1 to 3 | `data.py`: the comparison and the entry rule. `capture.py`: `checkpoint()` passes the model on a load, `checkpoint_files()` marks a folder load. `lightning.py`, if point 6 is signed | Nothing |
| Section 4, findings | `data_checks.py` | `detect/data.rs` |
| Section 4, the block and the words | `evidence.py`, with the all-ranks view | The block in its report, from `detect.rs` or `detect/data.rs` |

`accelerator.py` and `huggingface.py` need no change: both record a folder through `checkpoint_files`.

At the trial merge both engines are run on every `records` case and held to the same findings and the same block, in the seam test that already holds them together, and `experiments/traincheck-bugs/b14_tf29903.py` is run again: the buggy variant must raise `resume_incomplete` naming `lstm.weight_hh_l0` and `lstm.bias_hh_l0`, and the healthy one must not.

## 7. The coverage cases, by what they show

| Shows | Cases |
|---|---|
| Complete and missing | `every_tensor_is_in_the_checkpoint`, `two_lstm_tensors_left_out`, `more_missing_than_are_named` |
| Names made comparable | `saved_from_a_ddp_wrapped_model`, `saved_from_a_compiled_model`, `the_session_watches_an_inner_module`, `the_checkpoint_was_saved_from_an_inner_module` |
| Which entry | `a_bare_state_dict`, `unusual_names_under_a_plain_key`, `a_cache_beside_the_model`, `an_ema_copy_beside_the_model`, `exactly_half_the_names_are_the_models` |
| No entry | `a_cache_with_parameter_like_names`, `a_checkpoint_of_another_model`, `names_that_mostly_differ` |
| Lists that raise nothing | `an_adapter_only_checkpoint`, `buffers_left_out`, `tensors_the_model_does_not_have` |
| A shape | `a_shape_that_differs` |
| Not compared | `a_checkpoint_folder`, `the_comparison_raises`, `a_save_is_not_compared` |
