# A checkpoint's tensors against the model's: what the SDK records and what both engines say

Written Nov 20, 2026, and signed by the author on Nov 21, 2026 with the amendments below. **Nothing on this page is built yet;** lanes AC1 and AC2 build it. It is TrainCheck's B14: a checkpoint that left two LSTM tensors out, a resume that loaded it with `strict=False`, and no finding, because `resume_incomplete` looks only at which components a checkpoint holds (`experiments/traincheck-bugs/README.md`). The page is written once, with cases, before either engine changes, as the other two contract pages were. The cases are in `sdk/tests/fixtures/checkpoint/cases.json`, checked by `sdk/tests/test_checkpoint_tensor_contract.py`.

Each case is marked. A **pin** is what the code already answers, and the test runs it. A **new** case is not built: its expectation is the target, and the case also states what the code answers today, which the test runs, so the case cannot pass before its rule is built. The lane that builds a rule turns the case into a test of its own engine.

## The author's decisions

Approved as proposed (Nov 21, 2026):

1. **Which missing tensors raise:** only trainable parameters. A frozen parameter or a buffer that is not in the checkpoint is counted and named in the evidence and raises nothing, because an adapter-only checkpoint leaves the frozen base out on purpose and the record cannot tell that from a loss. The cost: BatchNorm statistics lost at a resume are named but not raised.
2. **Shapes:** a name both hold with another shape is raised as `shape_differs`. PyTorch's own `load_state_dict` raises on such a tensor whatever `strict` is (read in `torch/nn/modules/module.py`), so it reaches a running job only through a loader that drops what does not fit; then it was not restored. The data type is not compared.
3. **Unexpected tensors:** named in the evidence, never raised.
4. **A save** is not compared in this wave.
5. **Lightning:** `lightning.py` joins lane AC1's files, so a Lightning load passes its module and the entry it loads.
6. **Tied weights:** a name of the model that is the same parameter as a name the entry holds is not missing. safetensors' `save_model` writes a tied weight once (read in its source).
7. **The class:** `resume_incomplete` stays an observation, also when a trainable tensor is verified missing.

Amended by the author (Nov 21, 2026), and written into sections 1 and 2:

- **Which entry is compared.** The entry the loader identifies is used when that is known. A conventional checkpoint with a credible `model` entry and a fuller `ema` copy compares `model`. Where the source is genuinely ambiguous, tensor coverage is unavailable. The half-overlap rule stays as a filter on candidates, not as the decision: a cache that matches one common name, or a tie settled by the order of a dictionary, is too little evidence to call tensors missing.
- **No device waits, kept by construction.** `model.state_dict()` can run custom hooks or a distributed wrapper's state export, so it is not called. The model is read by a path known to touch local metadata only, and coverage is unavailable where that cannot be guaranteed.
- **Regressions,** now cases: an incomplete `model` beside a complete `ema` names both missing LSTM tensors whichever comes first; two equally plausible unnamed entries give unavailable coverage in either order; and no gathering export is called.

How the loader says it: the Lightning adapter knows the entry (`state_dict`), and a hand-written loop uses an optional keyword, `session.checkpoint(state, event="load", model_state=state["model"])`, given the very mapping it passes to `load_state_dict`. The author kept the keyword (Nov 21, 2026). Without it the rules of section 2 decide.

Amended again by the author the same day, on what an inferred entry may establish:

- **The loader's word is evidence; a key named `model` is not.** When the loader names the mapping, one matching tensor can establish that sixty others are missing. When TensorQuill only inferred the source from a key named `model`, one generic matching name is too little for that finding: the outcome is `unavailable`, `weak_match`.
- **Regressions,** now cases: the same one-of-61 mapping once inferred and once named by the loader; a genuine one-tensor model, which keeps its coverage; and two conventional entries that both qualify with neither named by the loader, `unavailable`, `ambiguous`, in either order.

This page's reading of "too little", for the author to overrule: an inferred entry, whether found by its key or by its names, is compared only when it holds at least half of the model's names. So the conventional key decides which entry is looked at before the others, and never by itself that tensors are missing. One case moved with it: `exactly_half_the_names_are_the_models` holds two of the model's seven names under the key `model`, and now reads `weak_match` where the first signed page compared it.

## 1. Where the comparison is made

Only the training process has the live model, so the SDK compares and records, and the two engines only read the record.

- **When.** Inside `session.checkpoint(state, event="load")`, once, before the first step of the resumed attempt. Nothing is added to a training step. A save is not compared (`a_save_is_not_compared`, a pin).
- **How the model is read.** By walking its modules and reading each one's own parameter and buffer tables, as `torch.nn.Module.named_parameters` and `named_buffers` do: names, shapes, `requires_grad` at the moment of the call, and which names are one parameter. `model.state_dict()` is never called and no state-dict hook runs (`a_model_whose_state_export_gathers`). No tensor's values are read.
- **Where that cannot be guaranteed.** A module that replaces `named_parameters`, `named_buffers` or `named_modules` with its own could do anything in that call, so the call is not made and coverage is `unavailable`, why `model_not_readable` (`a_model_that_replaces_its_parameter_walk`).
- **Buffers that are not saved.** A buffer registered with `persistent=False` is left out, as `state_dict()` leaves it out. That is read from the module's `_non_persistent_buffers_set`, the one attribute here that is not public; where a module does not have it, its buffers are not listed. Buffers raise nothing either way.
- **What is read of the checkpoint.** The names and shapes of the tensors in the mapping it was given, which is already in memory.
- **The model** is the one given to `tensorquill.init`.
- **Nothing is raised into training.** If anything in the comparison raises, the load is recorded as it is today with `"tensors": {"state": "unavailable", "why": "error"}` (`the_comparison_raises`).
- **Not measured:** the time of the call on a model of many thousands of tensors. The lane that builds it measures it and reports the number.

## 2. Which entry is compared

**Making names comparable.** Before any counting:

1. **Wrappers.** A leading `module.` (DDP) or `_orig_mod.` (`torch.compile`), repeated or not, is removed from every name on both sides (`saved_from_a_ddp_wrapped_model`, `saved_from_a_compiled_model`).
2. **One prefix.** If an entry then fails the filter below, one leading dotted path is tried: each leading path of the entry's names (`net.`, `net.lstm.`) removed from the entry's names that start with it, and each leading path of the model's names removed from the model's. The removal that leaves the most names in common is kept when the entry then passes the filter. The path removed is recorded as `prefix` (`the_session_watches_an_inner_module`, `the_checkpoint_was_saved_from_an_inner_module`, `one_part_of_a_larger_model`).

**The filter.** An entry is a candidate when at least half of its names are the model's, or at least half of the model's names are in it (`half_the_entrys_names_and_the_loader_named_it`, `one_part_of_a_larger_model`). The filter only rules entries out. Passing it does not make an entry the model's.

**The entries looked at** are every mapping in the checkpoint that holds tensors under string keys, at the top level or one level down (as `checkpoint_manifest` already looks), and the checkpoint itself when tensors sit directly in it. An entry recognised as an optimizer's, a scheduler's or a scaler's state is not looked at.

**The decision,** in this order. The first rule that applies decides, and `by` records which:

| | Rule | Outcome | Cases |
|---|---|---|---|
| 1 | **The loader named the entry** (`model_state=`, or the Lightning adapter) | Compared, `by: "loader"`, when it passes the filter, however few of the model's names it holds. When it does not pass, `unavailable`, why `names_differ`: it is the model's entry by the loader's word, but its names follow another scheme | `an_adapter_only_checkpoint`, `the_loader_names_the_ema_copy`, `more_missing_than_are_named`, `half_the_entrys_names_and_the_loader_named_it`, `the_loader_names_an_entry_of_another_model` |
| 2 | **Exactly one conventional entry passes the filter.** Conventional: its key names a model by today's name rule (`model`, `state_dict`, `net`, `module`, `network`, `weights`) and is not an averaged copy (a key with the word `ema`, `swa`, `shadow`, `avg`, `average` or `averaged`) | Compared, `by: "name"`, whatever other entries hold, when it holds at least half of the model's names. Otherwise `unavailable`, why `weak_match` | `an_ema_copy_beside_the_model`, `an_incomplete_model_beside_a_complete_ema`, `a_complete_ema_before_an_incomplete_model`, `a_cache_beside_the_model`, `a_model_of_one_tensor`; `one_matching_name_under_a_key_named_model`, `exactly_half_the_names_are_the_models` |
| 3 | **More than one conventional entry passes** | `unavailable`, why `ambiguous`, with `candidates`, sorted. Neither their order nor which holds more names decides | `two_conventional_entries`, `two_conventional_entries_in_the_other_order` |
| 4 | **No conventional entry passes, and exactly one other entry does** | Compared, `by: "names"`, when it holds at least half of the model's names. Otherwise `unavailable`, why `weak_match` | `a_bare_state_dict`, `unusual_names_under_a_plain_key`; `a_cache_sharing_one_name`, `an_adapter_only_checkpoint_the_loader_did_not_name` |
| 5 | **No conventional entry passes, and more than one other entry does** | `unavailable`, why `ambiguous`, with `candidates`, sorted. Neither their order nor which holds more names decides | `two_unnamed_entries_alike`, `two_unnamed_entries_alike_in_the_other_order` |
| 6 | **No entry passes** | `unavailable`, why `not_identified` | `a_cache_with_parameter_like_names`, `a_checkpoint_of_another_model`, `names_that_mostly_differ` |

**Half of the model's names,** in rules 2 and 4, counts a tied name as held, and is a share, not a count: a model of one tensor whose one name is in the entry is compared (`a_model_of_one_tensor`).

**Tied weights.** In a compared entry, a name of the model that is the same parameter as a name the entry holds counts as held, and is listed under `tied` (`a_tied_weight_saved_once`).

**What `components` then says of the model, at a load.**

| Outcome | `components.model` |
|---|---|
| Compared, or `names_differ` | Present: the entry compared, or the one the loader named |
| `not_identified` | Absent, so `resume_incomplete` says `not_identified` as it does today |
| `ambiguous`, `weak_match`, `model_not_readable`, `error` | Whatever today's rule gives. Weak evidence changes nothing |

Five cases answer differently from today, each stated in the case's `today`. Three of them newly raise `not_identified`:

| Case | `components.model` today | With this page | At a resume |
|---|---|---|---|
| `a_bare_state_dict` | absent | present | `not_identified` no longer raised |
| `unusual_names_under_a_plain_key` (L23) | absent | present | `not_identified` no longer raised |
| `a_cache_with_parameter_like_names` (L23) | present | absent | `not_identified` newly raised |
| `a_checkpoint_of_another_model` | present | absent | `not_identified` newly raised |
| `names_that_mostly_differ` | present | absent | `not_identified` newly raised |

`checkpoint_manifest` called without a model, as for a save, keeps today's rule and today's tests.

## 3. What the load's manifest carries

One new key, `tensors`, on the manifest of a load. It rides in `checkpoint_json` with the rest of the manifest; the record's schema does not change.

Compared:

```json
"tensors": {"state": "compared", "entry": "model", "by": "name", "model": 7, "trainable": 7, "checkpoint": 5, "matched": 5,
            "missing": {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}}
```

| Key | Meaning |
|---|---|
| `entry`, `by` | The checkpoint key compared (null for a bare state dict), and which rule of section 2 chose it: `loader`, `name` or `names` |
| `model`, `trainable` | How many tensors the model has (parameters and saved buffers), and how many of them are trainable parameters |
| `checkpoint`, `matched` | How many tensors the entry holds, and how many names both hold |
| `missing` | Trainable parameters not in the entry |
| `missing_frozen`, `missing_buffers` | Frozen parameters and buffers not in the entry |
| `unexpected` | Tensors in the entry that the model does not have |
| `shape_differs` | Names both hold with different shapes, and under `shapes` both shapes of each name listed |
| `tied` | Names not in the entry that are the same parameter as a name the entry holds; not missing |
| `prefix` | The one prefix removed, and from which side |

A list is present only when it is not empty. Each has `count`, the whole number, and `names`, sorted, at most 50 (`more_missing_than_are_named`, where the loader named the entry). Names are written as the model has them, wrappers removed; an unexpected tensor's name is written as the entry has it.

Not compared, `{"state": "unavailable", "why": ...}`:

| `why` | When | Case |
|---|---|---|
| `not_identified` | No entry passes the filter. `closest` names the best one and its counts when one shares any name | `a_checkpoint_of_another_model` |
| `ambiguous` | More than one entry could be the model's and nothing says which. `candidates` names them | `two_unnamed_entries_alike` |
| `weak_match` | The entry was inferred, by its key or by its names, and holds under half of the model's names. `closest` names it and its counts | `one_matching_name_under_a_key_named_model`, `a_cache_sharing_one_name` |
| `names_differ` | The loader named an entry that fails the filter | `the_loader_names_an_entry_of_another_model` |
| `model_not_readable` | The model cannot be read by the local path (section 1) | `a_model_that_replaces_its_parameter_walk` |
| `folder` | `session.checkpoint_files`: file names only, no file opened. Hugging Face's `Trainer` and Accelerate record their checkpoints this way | `a_checkpoint_folder` |
| `error` | The comparison raised | `the_comparison_raises` |

No `tensors` key at all: a capture written before this page, or a save (`a_resume_recorded_before_this_check`).

## 4. What both engines say at a resume

`resume_incomplete` is judged where it is today: at the first step of a later attempt that starts after the run's first step. The engines read `tensors` from the load's manifest only, and read `why` as a word to pass on, not one to act on. A resume judged from a save, because no load was recorded, reads no `tensors` even if the save carries one (`a_resume_judged_from_a_save`, a pin).

**Findings.** Today's findings are computed as today, from `components` and the rest, in today's order. Then, from `tensors` with `state` `compared`:

| List | Finding | Case |
|---|---|---|
| `missing` | `{"component": "model", "finding": "missing_tensors", "count": 2, "tensors": [...]}` | `a_resume_with_two_tensors_verified_missing` |
| `shape_differs` | `{"component": "model", "finding": "shape_differs", "count": 1, "tensors": [...]}` | `a_resume_with_a_shape_that_differs` |

`count` and `tensors` are the list's `count` and `names`, unchanged. A list is read only when its `count` is a positive whole number (`a_missing_list_with_no_count`); a `tensors` that is not an object is ignored (`a_tensors_entry_that_is_not_an_object`). `missing_frozen`, `missing_buffers` and `unexpected` add no finding. Beside another finding, the tensors come last (`missing_tensors_beside_a_missing_component`).

**Records no case states.** Lane AC2 met these while building the server and gave each a test; they are written here so the Python reader answers the same. They are this page's rules from Nov 21, 2026, not the author's words.

- The scalar keys of an entry (`state`, `why`, `entry`, `by`, `model`, `trainable`, `checkpoint`, `matched`) are copied as the record has them, whatever their type, null included.
- A `tensors` whose `state` is `unavailable` and that carries lists has them in its entry. They raise nothing.
- A `state` that is neither `compared` nor `unavailable`, or no `state`, gives no entry and no finding. `resumes` counts the entries.
- A `count` is read as `_int` in `data_checks.py` reads a whole number: an integer, never a float, a boolean or a string, and here also not above 9,223,372,036,854,775,807, which the server cannot hold. A list that is not an object is not read.
- `names` that is not a list reads as an empty list; the finding is still raised with its `count`.
- Of a list of `names`, only the strings are kept; `count` is unchanged. Found at the wave's merge: the server dropped the others and the Python reader kept them, and no case or lane test held a name that is not a string.
- The tensor findings come after every finding of today's, the optimizer snapshot's and the scaler's included.

**The three readings the author asked to keep apart**, and those beside them:

| What the load recorded | `resume_incomplete` | The evidence's `checkpoint_tensors` | Case |
|---|---|---|---|
| Compared, a trainable tensor missing | `missing_tensors`, naming each | `compared`, with the lists | `a_resume_with_two_tensors_verified_missing` |
| No entry identified as the model's | `not_identified`, as today, once | `unavailable`, `not_identified` | `a_resume_whose_model_was_not_identified` |
| A folder | Nothing from tensors; the folder's own findings as today | `unavailable`, `folder` | `a_resume_from_a_folder`, `a_resume_from_a_folder_with_no_optimizer_file` |
| An ambiguous source | Nothing from tensors | `unavailable`, `ambiguous` | `a_resume_from_an_ambiguous_checkpoint` |
| Compared, every trainable tensor there | Nothing | `compared`, with what was counted | `a_resume_with_every_tensor_verified` |
| No `tensors` | Nothing from tensors | No entry | `a_resume_recorded_before_this_check` |

So silence from `resume_incomplete` means every trainable tensor was verified only where `checkpoint_tensors` says `compared` for that resume. For a folder, an ambiguous source, and a load with no `tensors`, the evidence does not say the tensors are there.

**The evidence block.** `checkpoint_tensors` is a top-level key of a rank's evidence, present only when at least one judged resume has a `tensors` object with `state` `compared` or `unavailable`:

```json
"checkpoint_tensors": {"resumes": 1, "entries": [{"step": 10, "rank": 0, "attempt": 2000, "state": "compared",
    "entry": "model", "by": "name", "model": 7, "trainable": 7, "checkpoint": 5, "matched": 5,
    "missing": {"count": 2, "names": ["lstm.bias_hh_l0", "lstm.weight_hh_l0"]}}]}
```

An entry carries the step, rank and attempt of the resume, then `state`, `why`, `entry`, `by`, `model`, `trainable`, `checkpoint` and `matched` where the record has them, and each of five lists (`missing`, `missing_frozen`, `missing_buffers`, `unexpected`, `shape_differs`) as its `count` and `names`. `tied`, `prefix`, `shapes`, `closest` and `candidates` stay in the record. `resumes` counts them all; `entries` lists the first five by step, as `resume_incomplete` lists its changes. The all-ranks view carries each judged rank's block with its rank, as it does `skipped_updates`. Cases for a list that raises nothing: `a_resume_with_only_frozen_tensors_missing`, `a_resume_with_only_buffers_missing`, `a_resume_with_only_unexpected_tensors`; for an error: `a_resume_whose_comparison_raised`; for the cap: `a_resume_with_more_missing_than_named`.

**Words.** In the incident's sentence, after the component's name: `missing_tensors` reads "2 trainable tensors not in the checkpoint: lstm.bias_hh_l0, lstm.weight_hh_l0", and `shape_differs` reads "1 tensor with another shape in the checkpoint: head.weight". A sentence names at most five and then says "and 55 more"; the finding keeps every name it was given.

**What does not change, and what does.** No existing finding's name, sentence, kind label (`needs`) or note changes. `resume_incomplete` stays `checked` on what it checks today, and the run state is not changed by a `tensors` that is `unavailable`. Every new key is present only when the record carries `tensors`, so the evidence of a capture written before this page is byte for byte what it is today, Gate 1's shown files and the golden files among them.

A capture written after it is different, and that is expected, not a regression. A load recorded through the session carries `tensors`, so its evidence gains `checkpoint_tensors`; a Hugging Face or Accelerate resume gains the `folder` entry; and where section 2 identifies the model differently, `not_identified` appears or goes. A test that drives a session through a resume and compares a whole manifest or a whole evidence file will see these. The lane lists each such test it had to touch, with the before and after.

## 5. What this does not see

- **A tensor that is in the checkpoint and was never loaded into the model.** Names are compared, not values.
- **A checkpoint folder.** Hugging Face's `Trainer` and Accelerate resume from one, and its tensors are not read. Reading the names from a `safetensors` header would open a file, which `checkpoint_files` promises not to do; not proposed here.
- **A load that is not recorded,** and a first load at the run's first step, which is not a resume and is not judged today either.
- **A parameter frozen after the call.** It is counted as trainable, so it is raised if missing.
- **A source that is ambiguous or a weak match.** Nothing is named as missing, and the evidence says coverage is unavailable. Naming the entry with `model_state=` resolves it. A checkpoint that holds under half of the model, such as a head saved alone under `model`, is a weak match until the loader names it.
- **A model whose modules replace PyTorch's own walk of parameters.** Coverage is unavailable. In the clones read on Nov 21, 2026, PyTorch's own FSDP wrapper (`FullyShardedDataParallel`), its activation-checkpoint wrapper (`CheckpointWrapper`) and `RemoteModule` do; nothing in Transformers, Lightning, Accelerate, DeepSpeed or Megatron-LM does.

## 6. Who builds what

| Part | Lane AC1, Python | Lane AC2, Rust |
|---|---|---|
| Sections 1 to 3 | `data.py`: reading the model, the entry rules and the comparison. `capture.py`: `checkpoint()` takes `model_state=`, passes the model on a load, and `checkpoint_files()` marks a folder load. `lightning.py`: passes the module and its `state_dict` entry | Nothing |
| Section 4, findings | `data_checks.py` | `detect/data.rs` |
| Section 4, the block and the words | `evidence.py`, with the all-ranks view | The block in its report, from `detect.rs` or `detect/data.rs` |

`accelerator.py` and `huggingface.py` need no change: both record a folder through `checkpoint_files`.

At the trial merge both engines are run on every `records` case and held to the same findings and the same block, in the seam test that already holds them together. The real B14 workloads are run again beside that, `experiments/traincheck-bugs/b14_tf29903.py` in both variants: the healthy run must have no missing-tensor finding, and the buggy run must name `lstm.weight_hh_l0` and `lstm.bias_hh_l0`.

## 7. The coverage cases, by what they show

| Shows | Cases |
|---|---|
| Complete and missing | `every_tensor_is_in_the_checkpoint`, `two_lstm_tensors_left_out`, `a_model_of_one_tensor` |
| Names made comparable | `saved_from_a_ddp_wrapped_model`, `saved_from_a_compiled_model`, `the_session_watches_an_inner_module`, `the_checkpoint_was_saved_from_an_inner_module`, `one_part_of_a_larger_model` |
| The loader's word | `an_adapter_only_checkpoint`, `the_loader_names_the_ema_copy`, `more_missing_than_are_named`, `half_the_entrys_names_and_the_loader_named_it`, `the_loader_names_an_entry_of_another_model` |
| A conventional entry beside others | `an_ema_copy_beside_the_model`, `an_incomplete_model_beside_a_complete_ema`, `a_complete_ema_before_an_incomplete_model`, `a_cache_beside_the_model` |
| One entry, by its names | `a_bare_state_dict`, `unusual_names_under_a_plain_key` |
| Too little evidence | `one_matching_name_under_a_key_named_model`, `exactly_half_the_names_are_the_models`, `a_cache_sharing_one_name`, `an_adapter_only_checkpoint_the_loader_did_not_name`, `two_conventional_entries`, `two_conventional_entries_in_the_other_order`, `two_unnamed_entries_alike`, `two_unnamed_entries_alike_in_the_other_order` |
| No entry | `a_cache_with_parameter_like_names`, `a_checkpoint_of_another_model`, `names_that_mostly_differ` |
| Lists that raise nothing | `buffers_left_out`, `tensors_the_model_does_not_have`, `a_tied_weight_saved_once` |
| A shape | `a_shape_that_differs` |
| Reading the model | `a_model_whose_state_export_gathers`, `a_model_that_replaces_its_parameter_walk`, `the_comparison_raises` |
| Not compared | `a_checkpoint_folder`, `a_save_is_not_compared` |

## 8. Proposed, awaiting the author's sign-off: a checkpoint folder's tensors (lane AK-2, Nov 24, 2026)

**Not signed.** This section is lane AK-2's proposal, built and tested on CPU in wave AK (the plan's Phase 1, item 3). Nothing above it is rewritten. Where it would change a signed statement, the statement is named at the end, and stays as written until the author signs.

**What is read.** At `session.checkpoint_files(folder, event="load")` the model's weights in the folder are listed by name and shape from `.safetensors` headers only (`safetensors_listing` in `data.py`). A `.safetensors` file starts with N, an unsigned little-endian 64-bit integer, then N bytes of UTF-8 JSON naming each tensor's `dtype`, `shape` and `data_offsets`, with `__metadata__`, which is not a tensor. From each file those 8 + N bytes are read with plain reads, and nothing after them: no tensor byte, so a file whose data region is cut off or overwritten lists the same. `data_offsets` are not read. The `safetensors` package is not imported. A file is checked with `stat` before it is opened, so a named pipe or a device is never opened. Pickled weights (`.bin`, `.pt`, `.pth`, `.pkl`) are never opened.

**The entries.** Read in the layouts that Transformers, Accelerate and DeepSpeed write (read in their source on Nov 24, 2026):

| In the folder | Entry | Written by |
|---|---|---|
| `X.safetensors.index.json` and the shards its `weight_map` names | One entry, `X.safetensors.index.json`: the shards' tensors together | `save_pretrained` with `max_shard_size` (`model-0000k-of-0000n.safetensors`), so a `Trainer` checkpoint of a large model |
| Any other `X.safetensors` | One entry each, `X.safetensors` | `save_pretrained` unsharded (`model.safetensors`); Accelerate's `save_state` (`model.safetensors` for the first prepared model, `model_1.safetensors` for the second, and so on: separate models, never one) |
| No `.safetensors` file | None: `unavailable`, `folder`, exactly as before this section | FSDP through Accelerate or the `Trainer` (`pytorch_model_fsdp.bin`, `pytorch_model_fsdp_0/`), DeepSpeed (`*_model_states.pt`), Accelerate's `safe_serialization=False` |

**Which entry is compared.** The entries go through section 2's rules 2 to 6, unchanged: names made comparable (wrappers, one prefix), the half filter, and tied weights. An entry is read by its file's stem as a state dict's entry is read by its key: `model.safetensors` and `model.safetensors.index.json` are conventional (`model`), `model_1.safetensors` and an unindexed shard are not. Rule 1 does not apply: the loader names a folder, not a file, so `by` is `name` or `names`, never `loader`. `entry` is the file compared. So one Accelerate folder of two different models compares the session's model with the file whose names are its own, and two shards whose index is gone read `ambiguous`. Two models of the same architecture in one Accelerate folder both pass the filter, and rule 2 compares `model.safetensors`, the first prepared model's, whichever the session watches.

**What `components` says.** The folder's components are read from its file names as before (`files_manifest`), whatever the comparison finds. Section 2's table, which takes `components.model` away at `not_identified`, is not applied to a folder: a file named `model.safetensors` is the model's by the framework that wrote it.

**What is refused.** Never raised. The load's `tensors` is then `{"state": "unavailable", "why": "folder", "refused": ..., "file": ...}`, with `tensor` where one tensor is at fault. `why` stays `folder`, the word both engines already pass on, so both read such a load exactly as they read a folder before this section; `refused`, `file` and `tensor` stay in the record, as `closest` does.

| `refused` | When |
|---|---|
| `short_file` | Fewer than 8 bytes |
| `header_too_large` | N over 100,000,000, the format's own cap (the `safetensors` package refuses it as "header too large") |
| `header_past_end` | N larger than the file after its first 8 bytes |
| `not_utf8`, `not_json`, `not_an_object` | The header, or an index, is not UTF-8, not JSON, or not a JSON object |
| `bad_entry` | A tensor entry that is not an object, or has no non-empty string `dtype`, or no `shape` that is a list of whole numbers from 0 up (`true` is not a number) |
| `bad_index` | An index without a `weight_map` of names to strings, or naming a shard by a path rather than a file name in the folder |
| `shard_absent` | An index names a shard that is not in the folder |
| `duplicate_tensor` | The same tensor name in two shards of one index |
| `unreadable` | A file that is not a regular file, or cannot be opened or read |
| `too_many_files` | More than 1,024 index and weight files |
| `too_many_header_bytes` | More than 32 MiB of headers and indexes in all |
| `no_tensors` | Headers that name no tensor |

One file refused refuses the folder: the other entries are not compared, since the rules choose among all of them. A comparison that raises reads `{"state": "unavailable", "why": "error"}`, as section 1 says.

**A save** opens no file and carries no `tensors`, as decision 4 says.

**The bounds, for the author to confirm.** 1,024 files and 32 MiB of headers at one load, chosen, not derived: the largest published models are sharded into fewer than 200 files (not measured here), and at the 110 bytes a tensor's entry took in the measured headers, 32 MiB is about 300,000 tensors. Read at the cap, the JSON would take about half a second, once, at the load (extrapolated from the 1 MB measurement below, not measured). Measured on an 8-core laptop, warm page cache, median of 50: listing the headers takes 0.07 ms (one file, 12 tensors), 0.4 ms (one file, 291 tensors, a 32-layer Llama), 0.55 ms (the same in 3 shards), 1.4 ms (14 shards, 723 tensors, an 80-layer Llama) and 17 ms (11 shards, 9,555 tensors, 1 MB of headers); the whole `checkpoint_files` call, with the model read and compared, 0.18, 2.4, 2.8, 6.9 and 23 ms. The plan's "under 1 ms" holds for reading the headers of a model of a few hundred tensors, not for the whole call.

**Cases** are tests in `sdk/tests/test_lane_ak2_safetensors.py`, not in `cases.json`: a real file read as the `safetensors` package reads it, its data region cut, overwritten and removed; a sharded folder by hand and by `save_pretrained`; a tied weight `save_pretrained` writes once; a real Accelerate folder of two models; `module.` names; two unindexed shards; each refusal, a named pipe among them; a load naming two missing tensors, and one a shape that differs, through both engines; a refused folder through both; a real `Trainer` resume from a folder whose `model.safetensors` lost a tensor; and nothing raised under `python -W error`.

**What it does not see.** A folder of pickled weights, FSDP's or DeepSpeed's included. A tensor whose data is missing or damaged in the file. A file in a sub-folder (a diffusers pipeline's `unet/`, for one). Which file the framework actually loaded: the comparison says what the folder holds, as section 1 says what a state dict holds.

**Signed statements this would change, if signed:**

- Section 3, the `folder` row of `why`: "file names only, no file opened" would read "no `.safetensors` weights in the folder, or their headers refused (`refused` says why)".
- Section 4, the row "A folder" of the readings kept apart: a folder with `.safetensors` weights is read as any compared load.
- Section 5, the bullet "A checkpoint folder", which proposed nothing: replaced by this section.
- Section 6, "`checkpoint_files()` marks a folder load": it now compares one.
- Two expectations in `sdk/tests/test_lane_ac1_checkpoint.py` that wrote an empty `model.safetensors` as a stand-in for its name and pinned `{"state": "unavailable", "why": "folder"}` exactly: the case `a_checkpoint_folder` through a session, and `test_a_folder_is_marked_at_a_load_and_not_at_a_save`. The empty file is now opened and refused, `refused: "short_file"`, `file: "model.safetensors"`, and both tests say so. `cases.json` is unchanged: the case's `tensors` is still `unavailable`, `folder`.
