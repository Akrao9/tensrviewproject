# Stage 5: Initial Zoom Out Review

**Target:** `TensorLens v3 — Product and Build Plan.md` (lines 1–397, the whole file)
**Date:** Nov 14, 2026
**Standard:** `zoom-out-standard.md` (EGM 0.1.1)
**Context given to reviewers:** `TensorLens failure list.md`, `experiments/moe-bakeoff/README.md`, `sdk/README.md`, `sdk/RUN_ON_YOUR_JOB.md`. No web pages.

## Reviewer status

| Reviewer | Lens | Status | Verdict |
| --- | --- | --- | --- |
| `egm-zoom-out` | Simplicity and scope | completed | Needs revision: 0 Blockers, 6 Major, 9 Nits |
| `egm-inventive-catalyst` | False compromises and how to remove them | completed | 0 Blockers (advisory only in review mode), 6 Major, 4 Nits |

Both reviewers say the plan's objective and reasons are clear (the standard's context check passes). Neither could see the two embedded diagrams (line 26, "How it works", and line 226, the roadmap) or any linked web page. Line 228 already says the roadmap diagram is stale; the zoom-out reviewer expects the "How it works" diagram still shows the Arrow Flight ingest path, which is not built.

After merging overlapping findings: **0 Blockers, 10 Major, 10 Nits.**

## Major findings

### M1. The Rust ingest server and Arrow Flight are scheduled before anything needs them
- **Reviewers:** zoom-out (1), inventive catalyst (5)
- **Where:** lines 37–38, 70, 282, 285–286 (step 14), 358
- **Problem:** Everything built or planned for users works from local files: Python detectors on JSON Lines produce `evidence.json` and `.rrd`. That covers the agent evidence, the Rerun export and the free release. The step 14 design now ends with "the local file is the record of truth; streaming is opt-in", so a server is needed only for central or hosted storage, which is the paid tier the plan defers until after Gate 4 (lines 305, 323). Yet step 14 is in Weeks 4–7 and brings a second language, a transport, a storage format, and the open question of moving detectors to Rust (line 358). Lines 70 and 282 also still say the worker sends Arrow batches over Flight, which contradicts line 286.
- **Recommendation:** Move step 14 and the Arrow Flight part of step 11 to "after Gate 4, if a hosted tier is pursued". Fix lines 70 and 282 to match line 286. Drop the "move detectors to Rust" clause (line 37) and the open decision at line 358 until a server exists. The inventive catalyst offers a cheaper interim path if central storage is wanted sooner: the opt-in sender uploads closed, rotated Arrow IPC files to an object store the customer owns, and detectors read them in Python. Its costs: detection waits for each file to close, many small files, and a later step to combine them.

### M2. Design, as-built notes and progress log contradict each other
- **Reviewer:** zoom-out (2), with overlap from inventive catalyst (8, 9)
- **Where:** tier table (80–87) against the as-built list (89–95); lines 36, 39, 70, 75, 83, 116, 122, 127; roadmap steps 11–18 against the progress table (234–247); line 232
- **Problem:** A new reader cannot tell which design is current. For example:
  - the tier table says MoE counts use `scatter_add_` and are summed across data-parallel ranks, but the as-built list says compare-and-sum, never summed, one file per rank;
  - lines 84 and 122 say the cross-GPU check runs every N steps, but line 94 says `replica_every=50`;
  - line 70 says data goes over Arrow Flight, but line 239 says JSON Lines;
  - step 15 says "the first two detectors", but nine are built;
  - step 18 asks for an exporter that is already done;
  - line 232 pins the state to `d78fa78`, but line 91 describes `session.log` from `0de9b4f`;
  - lines 116 and 127 promise the drift check fires "on the first step" replicas differ, but with `replica_every=50` it fires on the first *checked* step, up to 50 steps late.

  Each new commit adds another layer of difference.
- **Recommendation:** Rewrite the tier table and capture rules to describe what is built, and mark unbuilt parts once, inline. Then delete the "As built … differs from this table" block. Keep a short progress table, or point to `sdk/README.md` as the source of truth for status. Collapse roadmap steps that are already done. Reword the drift check to "first checked step".

### M3. Too much implementation detail for a product and build plan
- **Reviewer:** zoom-out (3)
- **Where:** capture rules (59–76), the cross-GPU code listing and notes (129–190), lines 54–55, 74, 110, 218–220, 286
- **Problem:** PyTorch and Rerun internals (`skip_nnmodule_hook_guards`, `mark_static_address`, `histc` dtype rules, `_foreach_norm` grouping, `run_native_app` closure types, chunk sizes, a 57-line Python function) duplicate the SDK README and the code, drift from them, and bury the decisions a plan reader needs to weigh.
- **Recommendation:** Replace the rules list with five invariants:
  - hooks have fixed shapes and are registered before compile;
  - no host sync inside the step;
  - one device-to-host copy every N steps;
  - the probe runs in eager mode on a fixed batch;
  - the CI gate blocks graph breaks, recompiles and CUDA-graph skips.

  Replace the code block with the data flow (fingerprint → `all_gather` on the replication group → relative spread → evidence table) and link to `sdk/src/tensorlens/distributed.py` and `sdk/examples/replica_check.py`. Keep the "where it applies" and "limits" tables.

### M4. The Tier 2 probe pass and Tier 3 replay are on the critical path without evidence they are needed
- **Reviewers:** zoom-out (4, Nit 12), inventive catalyst (3, 10a)
- **Where:** step 16 (line 288); lines 72–74, 85–87, 97
- **Problem:**
  - The probe pass is the most complex part of the capture design: switching compile mode, saving and restoring each module's `training` flag, `histc` workarounds, and interposing SDPA, plus the GQA and FlashAttention-2 cases. Yet it serves failure-list entry 7, which is Medium priority and last in the list's own value-for-effort order. Gate 0 has not shown that attention maps are needed.
  - The `model._orig_mod` fallback for "older PyTorch" (line 72) is dead: the SDK already requires PyTorch 2.6, where `set_stance` exists.
  - Replay inside the training job cannot reproduce the fault. By the time a detector fires, the weights and optimizer state have moved on. The plan's own failure list cites PaLM, where replaying the spiking batches from an earlier checkpoint did not spike. Replay inside a sharded job also needs every rank in lockstep, or the collectives deadlock.
  - Line 87 says replay output is "written to Parquet", which conflicts with the step 14 rule that local files are not Parquet.
- **Recommendation:**
  - Put the probe and replay behind Gate 0. Build them only if your own failure list has a case that block statistics miss. Until then, use the cheap stand-in the failure list names: the absolute max of the query and key projection outputs.
  - If the probe is built, switch Hugging Face models to `attn_implementation="eager"` first. Treat other frameworks as out of scope for attention maps in v1, rather than patching each new attention kernel.
  - Delete the `_orig_mod` fallback.
  - If replay is kept, make it a separate job: it loads the nearest earlier checkpoint and re-runs to the flagged step with full capture. State its limit plainly: good for lasting faults such as collapse or drift, unreliable for one-step spikes.

### M5. The Gate 1 custom-view study does not fit a plan with one user
- **Reviewers:** zoom-out (5), inventive catalyst (10b)
- **Where:** steps 6–9 (lines 274–277), step 22 (line 297), lines 40, 213–220
- **Problem:**
  - Gate 1 is a three-arm study (stock Rerun, a Rune prototype, JSON only) with counterbalanced engineers and pre-registered questions. But line 228 says the author is the only user and no interviews are planned. With one engineer, "2× median time" means nothing; only the agent arms give a signal.
  - Keeping Rune alive as a third viewer stack ("Rune remains an option", line 220) keeps a speculative branch open.
  - Stock Rerun has already been checked on the Granite recordings.
- **Recommendation:** Stay on stock Rerun for release. Keep only the agent comparison: JSON alone against JSON plus stock Rerun screenshots, which also answers Gate 3's question. Reopen the custom view only when outside users report that sorting or layer selection blocks them. The inventive catalyst adds that before any such comparison, the exporter could precompute sorted per-layer expert tables and a "biggest changes across layers" table for stock Rerun, so the stock arm is not weaker than it needs to be.

### M6. The roadmap puts scope ahead of the step the plan says matters most
- **Reviewer:** zoom-out (6)
- **Where:** roadmap (224–305), the "Gate order" open gap (262), "Who it is for" (11)
- **Problem:** The root risk is unproven demand (lines 18, 319, 347). The shortest path to evidence is: run on your own job (Gate 0), run the two-GPU drift check, then release. M1, M4 and M5 put a Rust server, a probe pass and a viewer study ahead of that. Line 11 also names RL and robotics teams, but every built signal and test is specific to MoE language models.
- **Recommendation:** Reorder to:
  1. Gate 0 now: your own job and your own failures.
  2. The two-GPU NCCL drift run.
  3. Gate 2 on the dogfood run, with router-only capture and the existing detectors.
  4. Gate 3: the agent comparison.
  5. Release.

  List the ingest server, probe pass, replay, custom view and MLflow importer under "after release, if evidence asks for it". Narrow "Who it is for" to MoE training teams until another domain shows a signal.

### M7. The product promises catching failures during the run, but detectors run after it
- **Reviewer:** inventive catalyst (4)
- **Where:** lines 5, 24, 37, 286, 312
- **Problem:** The pitch is "caught … before it wastes days of GPU time", but detectors run on the file after training ends (line 286; the guide runs `python -m tensorlens` after the run). The free tier detects nothing while the run is going, and nothing exists to trigger replay or extra capture.
- **Recommendation:** Run the detectors incrementally in a separate process that reads the growing local file (the same mechanism as the opt-in sender), rewriting `evidence.json` as it goes. Optionally add a small control file the SDK checks at each N-step flush, which needs no GPU sync, so that process can switch on full capture or raise an alert.
- **Costs:**
  - one more process per node to supervise;
  - detectors must become incremental and keep their window state across restarts;
  - a control loop needs a margin between its on and off thresholds so it doesn't flip back and forth.

### M8. Full-capture cost is treated as fixed
- **Reviewer:** inventive catalyst (1)
- **Where:** lines 63, 67, 245, 257, 327, 341, 357; CI gate (76, 284)
- **Problem:** The plan accepts +4.15% for full capture and says per-step gradient norms (1.5 ms) cannot be sampled without hiding spikes. Line 257 also says activation hooks "cannot be sampled without recompiles", which is stronger than the SDK README's "without extra compiled variants". Dynamo would compile one graph per flag value and switch between them, not recompile every step. That option was never costed.
- **Recommendation:**
  - **(a) Use the clipping call.** Jobs that clip already read every gradient once for the total norm, and the SDK already wraps `clip_grad_norm_`. The wrapper could compute per-parameter norms with `_foreach_norm`, form the total from them, and clip with that total (PyTorch's `get_total_norm` and `clip_grads_with_norm_` allow this split). For jobs that clip, that makes per-parameter norms nearly free. It also records them before clipping, fixing the gap at line 67.
  - **(b) Two compiled variants.** Compile the activation hooks as an on and an off variant behind a flag. Change the CI gate to "no recompiles after both variants are warm".
- **Costs:**
  - (a) Only helps jobs that clip, and it must match `clip_grad_norm_` exactly (`foreach`, `error_if_nonfinite`, non-L2 norms). Under FSDP2 each shard's norm needs a reduction. Megatron computes its norm elsewhere.
  - (b) Doubles compile time, and the CUDA-graph memory pool under `reduce-overhead`. It uses part of the recompile limit, and relies on Dynamo behaviour that can change between releases.

### M9. Float64 fingerprints trade precision against cost
- **Reviewer:** inventive catalyst (2)
- **Where:** lines 93–94, 122, 188, 190, 252, 344; failure-list entry 4
- **Problem:**
  - Float64 costs about 3.7 ms per check on the RTX PRO 6000. That forced 50-step sampling, and sampling weakened the stalled-weight detector: 28 RMSNorm weights reported instead of 49.
  - The 1e-6 relative tolerance balances sensitivity against false alarms. One flipped low bit (silent data corruption) changes a float64 norm by far less than 1e-6, so it goes unseen.
- **Recommendation:**
  - Fingerprint the raw bits instead. View bf16 or fp16 storage as `int16` (fp32 as `int32`), and take an int64 sum plus a position-weighted int64 sum. This is exact, needs no float64, and catches sign flips and swapped values (the gap at line 203).
  - For data-parallel replicas, which should be bit-identical, compare for exact equality. Keep the float norm only to show how large the drift is once the bit check fails.
  - For stalled weights, take the fingerprint before and after the optimizer step on each sampled step, so the check again means "this step changed nothing".
- **Costs:**
  - Replicas that legitimately differ in their bits (mixed GPU models, nondeterministic kernels, some tensor-parallel layouts) would raise false alarms, which means keeping a float fallback.
  - FP8 and packed types need their own views.
  - The full read of the weights (about 1.5 ms) remains, so some sampling stays.

### M10. The writer queue has no size limit, the pattern the plan criticises in MLflow
- **Reviewer:** inventive catalyst (6)
- **Where:** line 260 against line 286
- **Problem:** Line 286 criticises MLflow's unbounded in-memory queue. Line 260 accepts the same in TensorLens: slow writes grow an in-memory queue. A hung NFS or Lustre mount, at about 26 KB per step in full capture, grows host memory until the machine runs out and the job dies.
- **Recommendation:** Put a limit on the queue. When it fills, write the overflow to node-local disk. Only if that also fails, drop records and write an explicit gap record (step range and reason), so detectors treat those steps as known-missing rather than as missing or repeated steps.
- **Costs:** detectors must handle gap records, and the overflow path needs a rule for a full disk.

## Nits

| # | Reviewer | Where | Finding | Recommendation |
| --- | --- | --- | --- | --- |
| N1 | zoom-out | 41, step 21, 352 | The MLflow importer is listed as "Build" but depends on three or more engineers, and there is one. Plotting tracker values in Rerun also repeats what trackers show, against the "Against trackers" position (line 13). | Remove it from the build table or mark it deferred. |
| N2 | zoom-out | 99–112 | Only route 1 of the W&B section (read `wandb.run.id`) is built or needed. Wrapping `wandb.log` is less needed now that the SDK has its own step and `session.log`. Routes 3–4 are hosted-tier concerns. | Shorten it to the built route, "use the SDK's own step and record the W&B step as metadata", and one line on terms and legal review. |
| N3 | zoom-out | 54, step 10, 238 | The Lightning callback with a rank-0 broadcast: the guide's manual setup already covers `MLFlowLogger` through `run_id=self.logger.run_id`. | Defer the callback until a user asks. |
| N4 | zoom-out | 312 | The paid tier lists "approved fixes delivered as pull requests", which is the agent loop postponed from v2 (lines 46, 353). | Remove it, or label it future work. |
| N5 | zoom-out, inventive catalyst | 75, 83, 92, 355 | Multi-rank reporting appears as a rule (75), as "summed across ranks" (83) and as an open decision (355). The SDK writes one file per rank. | Keep per-rank files, combine them when analysing, and close the open decision. This keeps per-rank skew visible and adds no collective. |
| N6 | zoom-out | 212, 259, step 20 | The Linux software-rendering check appears three times. | Keep it once, as the open gap. |
| N7 | zoom-out | 324–325 | Two risk rows (competitors, and trackers adding per-layer views) overlap and have the same mitigation. | Merge them. |
| N8 | zoom-out | "Why it is in the plan" (cross-GPU) | Megatron's `check_weight_hash_across_dp_replicas_interval` (cited at line 94) and TrainCheck already cover data-parallel replica checks. | Add one sentence on what TensorLens adds beyond them: replicated tensor-parallel weights and the rank-by-layer evidence table. |
| N9 | inventive catalyst | 12, 52–55, 66; guide §4 | "One line" no longer matches correct setup: `init`, `session.log`, `close`, router specs, and `auto_step=False` plus `session.step` under GradScaler or Megatron. | Let the frameworks that own the step call TensorLens (Lightning callback, Hugging Face `TrainerCallback`, torchtitan and Megatron hooks). Keep optimizer hooks for plain PyTorch, and soften the "one line" pitch. |
| N10 | inventive catalyst | overhead gate | If the 1% gate compares median step times, the median hides work that runs only every N or 50 steps (the `.cpu()` flush, the weight check, the replica `all_gather`). | Measure total wall time over windows that include those steps. **Checked: the benchmark already does this (see below).** |

## What both reviewers say is already the right size

- The local append-only file is the record of truth, with opt-in streaming (step 14). Both reviewers name it as the minimal design.
- Router-only is the default capture for MoE models.
- Pinned buffers and a side stream are added only if the overhead gate shows a stall (line 71).
- The plan lists explicitly what is postponed (line 46).
- It rejects the MLflow tracking-store plugin (line 57).
- Scalars stay on the team's own tracker (line 28).
- The evidence is viewer-neutral JSON.

## Checked against the code by the main session

The reviewers could read only the plan and the four context files. I checked the claims I could against the SDK and the benchmark:

- **N10 does not apply to the benchmark.** `sdk/examples/moe_dogfood.py` times each whole round of 200 steps and divides by 200, then takes the median of the three round averages. Each round therefore includes 4 weight checks and 4 transfers to the CPU (`every=50`). The measured overheads already count the periodic work. The point still applies to anyone measuring overhead from per-step times.
- **M4, the dead fallback, is confirmed.** `sdk/RUN_ON_YOUR_JOB.md` requires PyTorch 2.6 or later, and `torch.compiler.set_stance` exists from 2.6, so the `_orig_mod` path for older PyTorch is never used.
- **M10 is confirmed.** `Session` creates `queue.Queue()` with no `maxsize` (`sdk/src/tensorlens/capture.py`).
- **M2, the stale pin, is confirmed.** The progress table says "as of `d78fa78`", and `main` is at `0de9b4f` with `session.log`, the 50-step drift check and the doc corrections since.
- **M8(a) is partly confirmed.** The SDK records per-parameter gradient norms just before `optimizer.step()`, after clipping (README, "Clipping"). Whether reusing the clipping pass saves the full 1.5 ms is not measured.
