# Stage 9: Implementation Readiness Test

**Target:** `TensorLens v3 — Product and Build Plan.md` (lines 1–397, the whole file, as you chose)
**Date:** Nov 14, 2026, round 1
**Standard:** `implementation-readiness-standard.md` (readiness and code lenses); the style lens used its persona's rules, weighted toward the Roadmap
**Context given to reviewers:** the four linked project files; the code-accuracy reviewer also read `sdk/src`, `sdk/examples`, `sdk/tests` and `experiments/moe-bakeoff/generate.py`.

## Reviewer status

| Reviewer | Status | Raw count |
| --- | --- | --- |
| `egm-implementation-readiness` | completed | **NOT READY**: 3 Blockers, 12 Major, 7 Nits |
| `egm-codebase-details` | completed | 0 Blockers, 11 Major, 9 Nits |
| `egm-style` (roadmap focus) | completed | 1 Blocker, 12 Major, 9 Nits |

After merging: **4 Blockers, 15 Major, 14 Nits.** IDs: R = readiness, K = code accuracy, W = style.

The code-accuracy reviewer confirms that the plan's high-level claims match the code: module paths, `python -m tensorlens`, the `capture=` modes, `replica_every=50`, `probe_every=1000`, 64-bin histograms, 64×64 maps, the threshold constants, the `rerun-sdk==0.43.2` pin, `profile_overhead.py`, `replica_check.py` and `JsonlSink`. The failures are in the remaining work, which is described as goals rather than files, interfaces and tests, and in rules and progress rows that no longer match the code.

## Blockers

### IB1. No file boundary for any remaining step
- **Lenses:** R-B1, K-M11a
- **Where:** lines 34–44, 264–305
- **Problem:** The standard requires exact repository paths to create, modify or delete. None are given for:
  - the Rust ingest server (crate location, `Cargo.toml`);
  - the opt-in sender;
  - the Lightning callback;
  - recording of skipped steps;
  - Tier 3 replay;
  - the CI workflow;
  - the MLflow importer;
  - the custom-view spike;
  - release packaging.
- **Recommendation:** Add a table for each remaining step: path, create or modify, and a one-line purpose. For the sender, plug into the existing `Sink` protocol (`sdk/src/tensorlens/sink.py`).

### IB2. The ingest and transport design is unspecified and contradicts itself
- **Lenses:** R-B2, R-M1, W-M5, W-M6 (repeats Stage 5 M1, Stage 6 #1, Stage 7 C1)
- **Where:** lines 24, 36–38, 70, 282, 285–286, 358
- **Problem:** Three designs coexist:
  - an in-process Flight sender (line 70, step 11);
  - local file first with an opt-in sender (line 286);
  - "JSON Lines until the ingest server exists" (line 36).

  Where detectors run is undecided (line 24 against 37 against 358). If the server is built, these are open:
  - the Arrow schema for nested fields;
  - the Flight `DoPut` contract;
  - where the sender's saved position is stored;
  - how the server removes duplicate batches after a resume;
  - the Parquet layout;
  - authentication and TLS;
  - retention.
- **Recommendation:** Pick one design and delete the others. For v1, the recommended path (Stages 5 and 8) is to defer the server: "detectors run offline on the local file via `python -m tensorlens`; ingest is deferred". If the server stays in scope, specify the items above.

### IB3. No test plan for the remaining work
- **Lens:** R-B3
- **Where:** whole document
- **Problem:** No remaining step lists test cases. Missing examples:
  - the sender stalls and resumes with no growth in memory and no dropped steps;
  - the server receives a duplicate batch;
  - a killed process leaves a partly written JSONL line;
  - the Lightning callback broadcasts the run ID from rank 0;
  - a step skipped by GradScaler is recorded;
  - replay reproduces activations within a stated tolerance;
  - the MLflow importer gets a run with no TensorLens data.
- **Recommendation:** Give each remaining step happy-path and edge-case tests with assertions.

### IB4. Gate 2, and the other gates, cannot be judged
- **Lenses:** W-B1, R-M7, W-M3, W-M4, W-M11 (repeats Stage 7 C9)
- **Where:** lines 266–270, 275–277, 289, 298, 303–305
- **Problem:**
  - Gate 2 does not name the capture mode, the GPU, the measuring method or what "caught" means. Router-only passes on one GPU and fails on another; full capture is +4.15%. "Every seeded failure" refers to an empty list, because your own failures aren't written yet.
  - Gate 0 has no observable checks: "can be instrumented", "covers", "narrow or rethink".
  - Gate 1 needs counterbalanced engineers but has one user. "2×" names no metric or comparison arm, and "matched accuracy" has no tolerance.
  - Gates 3 and 4 have no numbers. Step 25's "track outside use" has no data source.
- **Recommendation:** Write each gate as a checklist. For example, Gate 2: "≤1% ms/step against capture off, `capture="routers"`, on each named GPU, median of 3 paired 200-step rounds as in `moe_dogfood.py`; caught = the detector fires at the seeded step ±K on the seeded layer, for each case in a named failure file". Add counts and margins to Gates 3 and 4.

## Major

### IM1. The code records two steps per iteration when a model has two optimizers *(new; confirmed in code)*
- **Lens:** K-M3
- **Where:** line 65; `capture.py` `_owns` and `_after_optimizer_step`
- **Problem:** Every optimizer that holds one of the model's parameters ends a step. A model trained with two optimizers (Muon plus AdamW, or a split of parameter groups) therefore commits two TensorLens steps per training iteration. Step numbers drift from the trainer's, and gradients are recorded twice.
- **Recommendation:** Pick a rule, such as ending the step only after the last owned optimizer, or requiring `auto_step=False` with two or more optimizers. Document it in the plan and the guide, and add a test.

### IM2. Steps skipped by mixed precision are reported done but are not built
- **Lenses:** K-M2, R-M5, D-N2 from Stage 7
- **Where:** lines 55, 66, 238, 278
- **Problem:** Progress row 10 says "Done except the Lightning callback", but nothing reads GradScaler's `found_inf`. `auto_step=False` counts iterations but does not record which steps were skipped. The failure list (entry 10) says "Not captured".
- **Recommendation:** Mark it not built. If it stays in scope, define the record field and how it is read (the scaler's state is private).

### IM3. The capture rules contradict the code
- **Lenses:** K-M1, K-M4, K-M5, K-N1–N3
- **Where:** lines 62, 67, 69, 70, 72, 74
- **Problem:**
  - Line 62 says the last forward pass is kept. The code keeps the largest `abs_max` and sums router counts across microbatches; only mean, std and zero fraction are last-pass values.
  - Line 72 says SDPA is replaced "instead of calling the fused kernel". The probe calls the original kernel and recomputes `softmax(QKᵀ)` alongside it.
  - Line 74 suggests switching to `eager`; the code switches FlashAttention 2 and 3 to `sdpa`. Line 74 also still says "check both before building step 16", though step 16 is built.
  - Line 72 describes an `_orig_mod` fallback that the code cannot reach, because PyTorch ≥ 2.6 is required. It also leaves out two things the probe does: it turns off deterministic mode and restores the random number generator state. And `probe.py` syncs once per histogram despite line 72's advice to avoid it.
  - Line 67 names `tensorlens.record_grads()`, which does not exist; the call is `session.record_grads()`.
  - Line 69's ring slot is not `step % N`.
  - Line 70 lists fewer host transfers than a flush makes (clip, spread and logged-scalar rings), and never states the default `every=50`.
- **Recommendation:** Replace lines 59–76 with the five invariants proposed in Stage 5 M3 plus a link to the SDK README, or correct each rule to match the code.

### IM4. Tensor-parallel replica checks are motivated in the plan but cannot be expressed
- **Lens:** K-M8
- **Where:** lines 118, 197; `capture.py` `record_replica_spread`
- **Problem:** The plan says that under tensor parallelism "the user or an integration has to say" which weights are replicated. The built API always fingerprints every trainable parameter, so the BLOOM case used to justify the detector cannot be set up through `init(replica_group=...)`.
- **Recommendation:** Specify a parameter filter (for example `replica_params=`) or say tensor parallelism is not supported in v1.

### IM5. Multi-rank behaviour is described wrongly
- **Lenses:** K-M6, K-M7, R-M2 (repeats Stage 7 C6)
- **Where:** lines 75, 84, 123, 355
- **Problem:**
  - The plan says rank 0 (or one rank) computes and reports the spread. In the code, every rank computes it and writes it to its own file.
  - The plan says rank 0 reports Tier 0. In the code, every rank captures everything and writes `<run_id>.rank<rank>.jsonl`, with the rank read from `torch.distributed` or the `RANK` / `LOCAL_RANK` environment variables.
  - Line 355 lists this as undecided.
- **Recommendation:** Record the built default and close the decision. Note that the server would see every rank's copy of the spread if ingest is ever built.

### IM6. The cross-GPU code listing duplicates `distributed.py` and has drifted from it
- **Lens:** K-M9 (repeats Stage 5 M3)
- **Where:** lines 129–186
- **Problem:** The listing lacks what the shipped module handles: empty tensors (where the listing's `torch.stack([])` would raise), Apple MPS, and a missing `_foreach_norm`. Its comment implies this snippet is what ran, but `replica_check.py` runs the shipped `init(..., replica_group=...)`.
- **Recommendation:** Replace the listing with links to `sdk/src/tensorlens/distributed.py` and `sdk/examples/replica_check.py`.

### IM7. The Gate 1 fixtures come from a stale copy of the exporter
- **Lens:** K-M10
- **Where:** lines 32, 274, 276; `experiments/moe-bakeoff/generate.py`
- **Problem:** `generate.py` has its own `routing_colors`, which still paints every zero-count expert red as "Dead" (the SDK fixed this with Idle and Starved colors). It also has only the fixed 0.35 overload rule. A Gate 1 trial built from it would test a stock-Rerun arm that differs from what ships.
- **Recommendation:** Make Gate 1 fixtures by writing synthetic JSONL and running it through `tensorlens.build_evidence` and `export_rerun`. Retire the duplicated functions in `generate.py`.

### IM8. The CI gate has no CI design, and the plan names a different mechanism from the one built
- **Lenses:** R-M6, K-M11b
- **Where:** lines 63, 76, 241, 284
- **Problem:**
  - The plan does not name a CI system, a GPU runner, the script, the model or a PyTorch version matrix. Line 63 asks for each supported release, but the set is never listed; the guide says 2.6+, tested on 2.10 and 2.14.
  - The plan says the gate scrapes `TORCH_LOGS`. The built checks use Dynamo counters: `error_on_recompile`, `counters["graph_break"]` and `cudagraph_skips` (`gpu_check.py`, `test_compile.py`, `test_probe.py`).
- **Recommendation:** Name `gpu_check.py` and `test_compile.py` as the gate. Specify the workflow file, the runner, the command, the version matrix and the assertions.

### IM9. Tier 3 replay has no design
- **Lens:** R-M3 (repeats Stage 5 M4, Stage 7 C7)
- **Where:** lines 87, 97, 288
- **Problem:**
  - There is no batch ID: the SDK has no data-loader hook, and failure list entry 14 says the data position is not recorded.
  - Replay needs weights from a checkpoint.
  - There is no size cap, output location or schema.
  - "If time allows" is not a decision.
- **Recommendation:** Move Tier 3 to postponed work, or specify it fully.

### IM10. The Lightning callback is only partly specified
- **Lens:** R-M4
- **Where:** lines 54, 67, 278
- **Problem:** Missing:
  - the class and module path;
  - the broadcast mechanism;
  - which hook calls `init`;
  - whether the step comes from `trainer.global_step` or the optimizer count;
  - how it interacts with the `clip_grad_norm_` wrapper;
  - a test matrix.

  `torch.compile` under Lightning is untested. The only Lightning test covers `Precision.clip_grad_by_norm`, not a `Trainer` run (K-N7), so line 238's "tested on CPU" overstates.
- **Recommendation:** Specify the signature, the hooks and the step source; or defer the callback (Stage 5 N3).

### IM11. The evidence and record schemas are undefined and unversioned
- **Lenses:** R-M8, R-M9 (repeats Stage 6 #5)
- **Where:** lines 36, 210–212, 286, 294
- **Problem:**
  - The evidence JSON has `schema_version: 1` in code, but the plan neither gives the schema nor points to it.
  - The JSONL records have no version field, and readers already handle older captures ad hoc ("captures made before `moe_top_k` existed").
  - Step 19's "actual sample step next to each PNG" has no file naming or layout.
- **Recommendation:** Point to `evidence.py` for the evidence schema. Add a `schema_version` to each record with a rule for reading older files. Specify the screenshot layout.

### IM12. The progress table mixes status, findings and design, and is stale
- **Lenses:** W-M8, W-M9, W-M12, K-N4, K-N6, R-M12 (repeats Stage 7 C11, C16)
- **Where:** lines 230–262, 287–288, 335
- **Problem:**
  - The status wording varies ("Partly", "Done except…", "Done early", "Tier 2 built…").
  - Cells carry design notes.
  - Row 13's "one expected recompile" conflicts with step 13's zero-recompile gate.
  - "On both GPUs" (line 243) reads as a two-GPU run.
  - Step 15 says "the first two detectors" while nine kinds ship, and line 335 says seven.
  - Line 238's "each step records the learning rate" is false for learning rates held as tensors on the GPU.
  - The failure list's summary table, which line 232 points to, still says entries 4–6 are not handled.
- **Recommendation:**
  - Use a fixed status set (Not started / In progress / Done / Blocked), with one sentence per row naming the unmet done criterion.
  - Say whether the recompile fails the gate.
  - Write "on both single-GPU runs (RTX PRO 6000 and A100)".
  - Give one detector count.
  - Update the failure list's summary table.

### IM13. Next actions and failure handling are unclear
- **Lenses:** R-M12, W-M1, W-M2, W-M10
- **Where:** lines 224–228, 255–262, 270
- **Problem:**
  - "Only once the gate passes" is contradicted twice.
  - What to cut when a gate fails, and who decides, is not named.
  - "Targets, not measurements" can read as "gates are soft".
  - Open gaps don't say which step or gate they block.
  - An implementer cannot tell whether to resume at step 1 or continue at steps 11–16.
- **Recommendation:**
  - State the rule actually followed.
  - Give a cut order for each gate.
  - Reword "targets" as "thresholds set before measuring, changed only at a gate review".
  - Add "Blocks: …; closed when …" to each gap.
  - Add an ordered list of next actions.

### IM14. The MLflow importer has no mapping
- **Lens:** R-M10
- **Where:** line 41, step 21
- **Problem:** The exporter takes TensorLens records, but MLflow runs hold scalars, params and artifacts. There is no metric-to-view mapping, no answer on whether detectors run on it, no CLI, and no plan for paging through about 5,000 metrics.
- **Recommendation:** Specify the mapping, or defer the importer (Stage 5 N1).

### IM15. Release cannot be carried out
- **Lens:** R-M11
- **Where:** lines 302–303, 311, 351
- **Problem:**
  - The package name is undecided, and `tensorlens` is taken on PyPI.
  - "Apache 2.0 / MIT" is ambiguous.
  - There is no publishing target or versioning scheme.
  - The bake-off demo depends on a Rune checkout outside the repository.
  - Step 25's "track installs" leaves telemetry undecided, which clashes with the risk row on data leaving the cluster.
- **Recommendation:** Decide the name, the licence and the publishing target, and state that there is no telemetry (or design it).

## Nits

| # | Lens | Where | Finding |
| --- | --- | --- | --- |
| N1 | R | 125 | The 1e-6 tolerance applies to "either spread", but the norm and sum spreads have different scales. Say whether one tolerance covers both. |
| N2 | R | 216, step 22 | Give the location of the pinned Rerun app and the commit of the upstream `custom_view` example. |
| N3 | R, W-N8 | 304, step 26 | "Model-call cost" is undefined when the team's own agent does the diagnosis. Step 22's "upgrade cost" has no unit. |
| N4 | R, W-N8 | 259, step 20 | The software-Vulkan check has no pass criterion. The bake-off's byte-identical repeat check could be adopted. |
| N5 | R | 236, 269 | Row 1–5 says to add TrainCheck to the competitor re-check, but step 4 doesn't list it. |
| N6 | K-N5 | 105, 110 | Reading `wandb.run.id` is built, after `run_id=`, `MLFLOW_RUN_ID` and MLflow. Recording the W&B step as metadata is not. |
| N7 | K-N8 | 260, 286 | `JsonlSink` appends. With no run ID found, separate runs share `tensorlens-out/run.rank0.jsonl` and show up as repeated steps; the planned sender must handle this. |
| N8 | K-N9 | 122, 130, 190 | "Every N steps" should be `replica_every`. Line 190 says the cost is unmeasured, while 3.7 ms is a proxy from the float64 weight-norm pass, not a measurement of this check. |
| N9 | K-N9 | 61 | The `_orig_mod` check rejects `torch.compile(model)` wrappers but not in-place `model.compile()` or blocks compiled one by one. Hooks stay attached after `close()`. |
| N10 | W-N1 | 226–228 | Remove the stale roadmap diagram. |
| N11 | W-N2 | 224 | "Twelve weeks" has no start date. |
| N12 | W-N3 | 228, 266–270, 349 | The voice switches between "the author" and "you". |
| N13 | W-N4, N5, N7 | 206, 228, 252, 257, 282 | "Seeded failures" is used before it is defined. "Restate criteria from elsewhere" points nowhere. N has no default. 3.7 ms is per check, not per step. "Stage 3" is undefined. |
| N14 | W-N6, W-N9 | 236, 296 | "The guide is ready" is subjective. "Three or more engineers" depends on the open team-size decision (line 352). |

## Checked against the code by the main session

- **IM1 is confirmed.** `_after_optimizer_step` calls `_commit` for every optimizer `_owns` accepts (`capture.py:457–460`). It is a code bug as well as a plan gap.
- **IM3's GQA point is partly wrong.** `probe.py` repeats only the key heads for grouped-query attention, as the reviewer says. That is correct: the probe computes attention weights, `softmax(QKᵀ)`, which need only the query and key, not the value. The plan's line 74 is what is inaccurate, since it says to repeat both.
- **IM3's other points are consistent with what this session saw:** `abs_max` kept across microbatches (commit `4df57f1`) and `torch>=2.6`.
- Stages 6–9 reviewed the plan as committed at `fa1c8e3`. The other session's uncommitted edits to `capture.py` and `evidence.py` were not part of this review.
