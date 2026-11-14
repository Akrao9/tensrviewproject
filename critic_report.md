# Stage 7: Critic Review

**Target:** `TensorLens v3 — Product and Build Plan.md` (lines 1–397, the whole file)
**Date:** Nov 14, 2026, round 1
**Standards:** `feature-spec-standard.md` (design and structure lenses); the style lens used its persona's own rules
**Context given to reviewers:** `TensorLens failure list.md`, `experiments/moe-bakeoff/README.md`, `sdk/README.md`, `sdk/RUN_ON_YOUR_JOB.md`. No web pages.

## Reviewer status

| Reviewer | Status | Raw count |
| --- | --- | --- |
| `egm-design` (architecture) | completed | 0 Blockers, 11 Major, 5 Nits |
| `egm-structural` (document structure) | completed | 1 Blocker, 7 Major, 8 Nits |
| `egm-style` (technical writing) | completed | 0 Blockers, 16 Major, 17+ Nits |

After merging duplicates: **1 Blocker, 17 Major, 20 Nits.** IDs below are `<lens>-<n>`: D = design, S = structural, W = style.

All three reviewers say the plan does not yet meet the feature-spec standard. The design lens notes the capture rules are "unusually rigorous". The gaps are in what happens after capture, in the alternatives considered, and in risks without a mitigation.

## Blocker

### B1. No "Alternatives considered" section, and the largest unbuilt component has no rejected options
- **Lenses:** S-B1, D-M2
- **Where:** whole document; key choices at lines 34–46, 38, 208–220, 285–286, 358
- **Problem:** The standard requires viable alternatives with concrete reasons for rejecting them. Some smaller decisions have them, buried in prose:
  - the MLflow tracking-store plugin (line 57);
  - patching `Tensor.backward` (line 91);
  - streaming from memory, and Parquet as the local file (line 286);
  - sampling gradient norms (line 257);
  - the W&B routes table (lines 103–108).

  The decisions that shape the system have none. The biggest new component, a Rust Arrow Flight ingest server, is never weighed against simpler options:
  - local files plus uploading finished files to object storage, combining them into Parquet offline;
  - reusing the tracker's artifact store;
  - a Python ingest service next to the Python detectors that already exist.

  Choosing Rust is also what creates the open question at line 358 (detectors in Python or Rust). The v2 items postponed at line 46 are listed with no reason.
- **Recommendation:** Add an H2 "Alternatives considered" after "Build and borrow". Give one row per major decision (viewer, ingest, transport, storage, detector language, and the postponed v2 items): what was chosen, what was rejected, and why. Move the scattered rejections into it.

## Major

### C1. The data path is described three incompatible ways
- **Lenses:** D-M1, W-M4, S-M4 (repeats Stage 5 M1 and Stage 6 finding 1)
- **Where:** lines 24, 28, 36–38, 70, 282, 286
- **Problem:**
  - Line 24 says failures are detected on the ingest server.
  - Line 70 and step 11 say a worker sends Arrow batches over Flight.
  - Line 286's agreed design says a local file comes first, with an opt-in sender.
  - What is built runs the detectors offline, on JSON Lines files, after the run.

  The ~300-word transport design sits inside roadmap step 14, with an unattributed "from a reviewed proposal".
- **Recommendation:** Write one current data-flow section: hooks → GPU ring buffer → writer → local file → optional sender → ingest → detectors → evidence → Rerun or the agent. Promote the transport design there and leave a one-line pointer in step 14. Rewrite lines 24, 70 and 282 to match. Say where detectors run in v1 and later.

### C2. The cross-GPU check can hang the training job
- **Lens:** D-M5
- **Where:** lines 94, 122–124, 168–185
- **Problem:** `replica_spread` calls `dist.all_gather` every `replica_every` steps. Each rank decides when to call it from its own step counter. If ranks disagree, the ranks still in the collective block until the NCCL timeout and the job dies. Ranks can disagree when:
  - the automatic counter misses a step skipped by GradScaler;
  - one rank resumes with a different `start_step`;
  - an exception skips the call on one rank.

  A monitor that can hang the job breaks the "leave it on" premise.
- **Recommendation:**
  - Trigger on a step all ranks agree on: the trainer's step, or an all-reduce of the step every K steps.
  - Or run the check on a dedicated process group with a short timeout, or as an asynchronous collective with a watchdog.
  - Add the failure mode to Risks.

### C3. The writer queue has no size limit
- **Lens:** D-M6 (repeats Stage 5 M10)
- **Where:** line 260 against line 286
- **Problem:** Line 286 chooses local-file-first so that memory stays bounded. Line 260 accepts an in-memory queue that grows while writes are slow. At about 26 KB per step per rank, a stalled NFS or Lustre mount over hours leaks host memory on every rank, and a host out-of-memory is worse than a stall.
- **Recommendation:** Give the queue a maximum size and a policy for when it fills: block, spill to node-local disk, or drop with a gap record the detectors understand. Add it to Risks.

### C4. Nothing in the design catches failures during the run, although the pitch and paid tier promise it
- **Lens:** D-M3 (repeats Stage 5 M7)
- **Where:** lines 5, 24, 286, 312
- **Problem:** The plan promises to catch failures "before it wastes days of GPU time" and sells "round-the-clock monitoring, alerts". But every detector runs on a complete file after the run. The detectors keep state over windows (50-step medians, 3 checks in a row, starved-expert windows), so running them incrementally needs a design: how that state is saved, how late or out-of-order batches are handled, and a target time from a failing step to an alert.
- **Recommendation:** State a target detection delay and an incremental-detector design. Add an alert-latency criterion to Gate 3 or Gate 4.

### C5. The default capture mode does not deliver the pitch, and capture cannot be raised during a run
- **Lens:** D-M7; related wording issues W-M1, W-M2
- **Where:** lines 5, 12, 15, 36, 52, 61, 93, 245, 327, 357
- **Problem:**
  - The pitch promises activations, attention and routing from "one line", "cheaply under `torch.compile` and FSDP".
  - The MoE default is router-only, attention (Tier 2) is not built, and full capture costs +4.15%.
  - Hooks cannot change after compile, so a run that starts failing cannot be switched to full capture.
  - FSDP paths are untested (line 198), and no real failure has been caught yet (line 335).
  - "One line" also ignores the router discovery, the Megatron and GradScaler step calls, and the PyPI name problem (line 351). Line 52 still says `pip install`.
- **Recommendation:**
  - Either reword the pitch: routing always on, internals on demand, FSDP planned.
  - Or design a way to raise capture during a run: full hooks registered at init behind an on-device flag (cost to be measured), or a documented "restart from checkpoint with `capture="full"`" workflow tested in Gate 2.
  - Replace the `pip install` line with installing from a clone.

### C6. Multi-rank aggregation is not designed, and three statements disagree
- **Lenses:** D-M4, W-M14, S-N6 (repeats Stage 5 N5, Stage 6 finding 13)
- **Where:** lines 75, 84, 92, 260, 355
- **Problem:**
  - Line 75 says rank 0 reports; the build writes one file per rank; line 355 lists it as undecided.
  - Router counts are not summed across ranks, so imbalance is judged on one rank's tokens. That is noisier, and with expert parallelism one rank may not see the whole router.
  - Nothing defines how per-rank files are merged or aligned by step.
- **Recommendation:**
  - Specify where files are merged (sender, ingest or exporter) and how steps are aligned.
  - Decide whether router counts are summed before detection.
  - Add a multi-rank case to Gate 2.
  - Close the open decision at line 355.

### C7. Tier 3 replay assumes the failing state can be reproduced
- **Lens:** D-M8 (repeats Stage 5 M4)
- **Where:** lines 87, 97, 288
- **Problem:**
  - Replay needs the model and optimizer state at the flagged step. Since detection happens after the fact, only a checkpoint can restore them.
  - It also needs deterministic data loading.
  - The failure list's PaLM case says replay from an earlier checkpoint did not reproduce the spike.
  - The output ("MB to GB, written to Parquet") has no storage target, and Parquet conflicts with step 14.
- **Recommendation:** Specify the checkpoint dependency, how reliably replay reproduces a failure, and where the output goes. Otherwise drop Tier 3 from v1.

### C8. The probe pass can hang or run a job out of memory
- **Lens:** D-M9 (Stage 5 M4 covered its priority; this covers its safety)
- **Where:** lines 72–74, 288
- **Problem:**
  - Under FSDP, tensor parallelism or pipeline parallelism, an eager forward pass issues collectives, so the probe must run on every rank at the same step. That is the same hang risk as C2.
  - An explicit `softmax(QKᵀ)` uses memory that grows with the square of the sequence length. At 8k–32k context on a job near its memory limit, it can run out of memory and kill training.
  - Interposing `F.scaled_dot_product_attention` affects the whole process, and must be undone even if an exception is raised.
- **Recommendation:** Run the probe on all ranks at a coordinated step. Compute attention for one head over a bounded window. Restore SDPA in `try/finally`. Make an out-of-memory guard part of step 16's acceptance criteria.

### C9. Gate criteria cannot be judged as written
- **Lenses:** D-M10, W-M15
- **Where:** lines 245, 289, 298, 305, 327
- **Problem:**
  - Gate 2's "overhead under 1% on a dogfood run" names no hardware, model size, compile mode or capture mode. Router-only passes on the RTX PRO 6000 and fails on the A100.
  - All measurements are on a 1B model on one GPU with CUDA graphs off, while the target users train large models on clusters.
  - Gates 3 and 4 have no threshold: "finds the right cause more often", "report problems it caught".
- **Recommendation:**
  - Fix Gate 2's terms: capture mode, minimum step time or model size, compile mode, and a GPU set that includes a cluster-class configuration. Say whether the A100 result fails it.
  - Add a margin and sample size to Gate 3, and a minimum number of teams to Gate 4.

### C10. Security and capacity are missing from Risks
- **Lenses:** D-M11, S-M6
- **Where:** lines 20, 38, 52, 82, 286, 312–331, 351
- **Problem:**
  - The opt-in network sender and the Flight endpoint have no authentication, encryption or tenant-isolation design.
  - "Approved fixes delivered as pull requests" implies write access to customer repositories.
  - The Rerun MCP control port and `--web-viewer` forwarding expose local services.
  - The PyPI name squatting appears only in the decisions list.
  - There is no throughput or storage estimate. The Tier 0 estimate of about 1.3 KB per step (line 82) is about 20 times smaller than the measured 26 KB per step of JSON, and per-parameter norms are missing from the tier table.
  - Cost per run is not modelled, while pricing is per seat.
- **Recommendation:** Add risk rows for endpoint authentication, the repository-write path, local ports and the package name. Add a capacity estimate from measured sizes, and say whether per-seat pricing covers per-run storage.

### C11. Plan, as-built notes and progress disagree throughout
- **Lenses:** S-M3, W-M5, W-M6, W-M7, W-M8, W-M10, W-M12, W-M13, D-N1 (repeats Stage 5 M2)
- **Where:** many; the clearest:

| Line | Says | Actually |
| --- | --- | --- |
| 83–84, 122, 130 | `scatter_add_`, summed across ranks, every N steps | compare-and-sum, not summed, every 50 steps (line 94) |
| 37, 287 | two or three detectors | eight to ten built (line 243) |
| 62 vs 239 | last microbatch kept | largest value kept across microbatches |
| 190, 206 | cost and code untested | about 3.7 ms measured; passes on two CPU processes |
| 243, 253 | all 49 RMSNorm weights "never change" | at the 50-step default, 28 reported; some change in a few elements |
| 341 | "whether full capture can get under 1%" is unproven | that configuration measured +4.15% |
| 232 | state as of `d78fa78` | `main` is at `24f9a1b` |

- **Recommendation:** Decide whether the spec states the target design or the current one, and fix the table and text to match. Keep measurements in one place (the progress table or the SDK README) and link to them; +4.15% currently appears in six places.

### C12. Required sections are missing: problem statement, goals and non-goals
- **Lenses:** S-M1, S-M7, D-N4
- **Where:** lines 5–46
- **Problem:** The problem sits in bold paragraphs after the positioning bullets. Goals (overhead under 1%, the agent beating tracker-only data) and non-goals (postponed v2 items, hardware hangs, data problems, fully sharded setups) are only implied.
- **Recommendation:** Add "Problem" and "Goals and non-goals" sections under Product, with measurable goals taken from the gates.

### C13. The architecture diagram does not render
- **Lenses:** S-M2, D-N5, W-N5 (repeats Stage 6 finding 2)
- **Where:** lines 26, 226
- **Problem:** `&#91;embedded content: How it works · 7 parts, 8 flows\]` is a leftover from exporting the document. The roadmap embed is stale (line 228).
- **Recommendation:** Replace it with a Mermaid diagram or a numbered list of parts and flows. Delete or regenerate the roadmap embed.

### C14. Risk material is spread across four places, and several items have no mitigation
- **Lens:** S-M5
- **Where:** lines 201–206, 255–262, 317–331, 337–347
- **Problem:** Scalability limits appear only outside the Risks table: NFS and Lustre, A100 overhead, Linux headless rendering, untuned thresholds. "Open gaps" and "Unproven" have no mitigation column.
- **Recommendation:** Merge them into one Risks table (risk, effect, mitigation). Keep "Unproven" for hypotheses that a named gate will test.

### C15. The W&B state disagrees in three places
- **Lens:** W-M3
- **Where:** lines 36, 105, 354
- **Problem:** "W&B as a later target" (36), described as current (105), and listed as an open decision (354). The SDK README says W&B active-run detection is built and tested.
- **Recommendation:** Make the three agree.

### C16. The failure list's summary table contradicts the plan
- **Lenses:** W-M11, D-N3
- **Where:** line 232; the failure list's summary table
- **Problem:** The plan points to the failure list as the record of detector coverage. Its summary table still says entries 4–6 are "Not captured", "no spike detector" or "step counter restarts at 0", while its detail sections and the plan say they are covered. Gate 0's answer key depends on that table.
- **Recommendation:** Update the failure list's summary table.

### C17. The paid tier lists a feature nothing builds
- **Lenses:** W-M16 (repeats Stage 5 N4, Stage 6 finding 8)
- **Where:** line 312
- **Problem:** "Approved fixes delivered as pull requests", while the agent loop and fixes are postponed (lines 46, 353).
- **Recommendation:** Drop it, or label it post-v1.

## Nits

| # | Lens | Where | Finding |
| --- | --- | --- | --- |
| N1 | S-N1 | 59, 78, 120, 192, 201, 337, 349 | Bold text used as headings instead of H3s, so the sections are missing from the outline. The bold "How it works" at line 120 repeats the H2 at line 22; rename it "Detection steps". |
| N2 | S-N2 | 18, 20, 52, 311–313 | Bold lead-ins punctuated inconsistently. |
| N3 | S-N3, W-N17 | 360–397 | Sources use three bullet markers, so they render as three lists. Some claims cite mirrors or summaries (mcpservers.org, upd.dev, liner.com, apxml.com) where the failure list cites primary sources. |
| N4 | S-N4 | 82–86, 389, 393 | Unneeded escapes from export: `\~`, `all\_gather`, `set\_stance`. |
| N5 | S-N7, W-N4 | 46, 228, 266, 295 | Mixes "you" and "the author". |
| N6 | S-N8, W-N6 | 3, 5 | Header date Nov 13 with Nov 14 content; no "last updated"; v2 is not linked. |
| N7 | W-N1 | 40, 214, 277, 339 | One Gate 1 criterion, three wordings. |
| N8 | W-N2 | 72 | One bullet of about 200 words covering five topics. |
| N9 | W-N3 | 76 | Unclear sentence about static buffers and CUDA graphs. |
| N10 | W-N7 | 69–72, 82–87, 122 | N and M have no defaults. |
| N11 | W-N8 | 220, 275 | "Rune" is undefined (also Stage 6 finding 3). |
| N12 | W-N9 | 245 | Lists two RTX PRO 6000 router-only runs, not three (the Stage 3 run measured +0.54%). |
| N13 | W-N11 | 259 | "Headless screenshots only on macOS" should say Rerun screenshots: the bake-off rendered Rune under lavapipe on Linux arm64. lavapipe is the Vulkan driver built on llvmpipe. |
| N14 | W-N12 | 262 | "Gate order" is a recommendation, not a gap. |
| N15 | W-N13 | 327 | The mitigation "CI with compile on" reads as in place; line 241 says it is not wired. |
| N16 | W-N14, Stage 5 N7 | 324–325 | Overlapping risk rows. "Wedge" is jargon. "torch.compile support that they lack" is unsupported. |
| N17 | W-N15 | 347 vs 18 | Near-duplicate sentences. |
| N18 | W-N16 | 302 | "Apache 2.0 / MIT" could mean a dual licence or an undecided choice. |
| N19 | D-N2 | 238, 278 | Progress row 10 says "Done except the Lightning callback", but steps skipped by mixed precision are only handled through `auto_step=False`. |
| N20 | W minor | 97, 236, 270, 315, 112 | Vague or unsupported wording: "compares cleanly"; "guide is ready" though Megatron and Lightning with compile are untested; "narrow or rethink"; "an SDK alone is hard to charge for"; the terms caveat sits apart from the legal-review line. |

## Checked against the code by the main session

- **C2 is confirmed.** `Session._cadence` uses each process's own `_steps_committed`, and `replica_spread` calls `dist.all_gather` with no timeout or asynchronous option (`sdk/src/tensorlens/distributed.py`). A rank whose count is off by one step waits at the collective while the others move on. This is a code bug as well as a plan gap.
- **C3 is confirmed** (`queue.Queue()` with no `maxsize`, `sdk/src/tensorlens/capture.py`).
- **C11's commit is confirmed:** `main` is at `24f9a1b` (a profiling script added by another session). There are also uncommitted changes in `capture.py` and `evidence.py` from that session; this review did not touch them.
- **C10's size gap** follows from the measured 17 MB for 650 steps of full capture: about 26 KB per step against the plan's 1.3 KB estimate. The estimate leaves out per-parameter gradient and weight norms, and JSON is larger than binary.
