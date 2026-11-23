# Stage 8: Intermediate Zoom Out Review

**Target:** `TensorLens v3 — Product and Build Plan.md` (lines 1–397)
**Date:** Nov 14, 2026
**Reviewer:** `egm-zoom-out`, completed. It read the three earlier reports, the four context files and, read-only, `capture.py`, `evidence.py` and `probe.py`.
**Verdict:** needs revision. 0 Blockers, 7 Major, 5 Nits.

## The plan changed during the review

Another session committed `9a5a39f` (the Tier 2 probe pass is now built) and `fa1c8e3` (the progress table moved to `9a5a39f`) while Stages 6 and 7 were running. Their line numbers refer to the earlier text. So Stage 5's advice to put the probe behind Gate 0 has been overtaken: the probe exists. The reviewer reads this as more evidence for Stage 5 M6, that work keeps running ahead of the gates.

## Summary

About half of what Stages 6 and 7 recommend is essential, and most of that is cheap: fixing contradictions, defining terms, and two real bugs in shipped code. The rest would design parts the plan should defer. Adding everything would grow the plan from about 400 lines to well over 500. Two rules:

1. **Delete first.** Apply Stage 5's deferrals (M1 ingest server, M4 replay, M5 custom-view study, N4 paid pull requests) before anything else. That removes about a third of the later findings at no cost.
2. **Pay for each addition with a deletion.** Stage 5 M3's cuts (the capture rules, the code listing, the as-built block, the step 14 transport notes) free about 100 lines.

## Major

### Z2-1. Apply the deferrals first
- **Concerns:** Stage 6 findings 1, 3, 8; Stage 7 B1, C1, C7, C10 (the endpoint and repository-write parts), C17
- **Point:** Several later findings ask for design work on parts Stage 5 said to defer:
  - an alternatives study for the ingest server;
  - authentication and encryption for the Flight endpoint;
  - a threat model for "fixes as pull requests";
  - the checkpoint dependency of replay;
  - a definition of Rune for the three-arm Gate 1.

  Deferring step 14, removing the paid pull-request feature, dropping Tier 3 from v1 and shrinking Gate 1 to the agent comparison makes those findings small or moot.
- **Recommendation:** Do these deletions and deferrals before anything else.

### Z2-2. Stage 7 B1 is not a Blocker: move existing text into a short decisions table
- **Concerns:** Stage 7 B1, C12
- **Point:** The decisions that matter already have their reasons, scattered through the plan:
  - the MLflow tracking-store plugin (line 57);
  - patching `Tensor.backward` (line 91);
  - streaming from memory, and Parquet as the local file (line 286);
  - sampling gradient norms (line 257);
  - the W&B routes (lines 103–108).

  The ingest gap should be closed by deferring the component, not by studying options for it.
- **Recommendation:** Downgrade to Major. Add one compact "Decisions" table that *moves* the existing rejections, deleting them where they sit now. Add a "Non-goals" list that absorbs line 46 and the post-release deferrals: ingest, hosted sender, replay, custom view, MLflow importer, a built-in explainer. Skip a separate "Problem" section; lines 5 and 18–20 already serve.

### Z2-3. Catching failures during the run: fix the wording, skip the incremental design
- **Concerns:** Stage 7 C4, Stage 5 M7; lines 5, 24, 286, 312
- **Point:** Incremental detectors with saved state and an alert-latency gate are monitoring infrastructure for the paid tier, which is decided after Gate 4. In v1, re-running `python -m tensorlens` on the growing per-rank file already detects during the run without saved state. It re-reads the whole file, which is fine at about 26 KB per step.
- **Recommendation:** Reword the pitch to "detects from the local file, during or after the run". Add one line saying the CLI can be re-run on a file that is still being written, and check that the reader tolerates a partly written last line. List incremental detectors and alert latency as non-goals for v1.

### Z2-4. The cross-GPU hang is essential to fix; use the smallest fix
- **Concerns:** Stage 7 C2; lines 94, 122–124; `capture.py` `_cadence` and `replica_spread`
- **Point:** It is a real bug in shipped code and breaks "leave it on". But a dedicated process group or an asynchronous collective with a watchdog is more machinery than the risk needs. Under DDP:
  - every rank runs the same `step()` calls;
  - GradScaler decides to skip a step from all-reduced gradients, so every rank decides the same way.

  The realistic ways ranks disagree are a mismatched `start_step` and misuse of manual `session.step`.
- **Recommendation:**
  - When `replica_group` is set, schedule the check from the step the caller passes.
  - In auto-step mode, document that the step counts must stay in lockstep.
  - Add one Risks row.
  - Defer watchdogs until a hang is actually seen.

### Z2-5. The writer queue limit is essential; pick one policy, without a second disk
- **Concerns:** Stage 7 C3, Stage 5 M10; line 260 against 286; `capture.py` `queue.Queue()` with no `maxsize`
- **Point:** Spilling to node-local disk adds a second sink, a disk-full policy and a merge step. `sink=JsonlSink(path)` can already point the file at node-local disk.
- **Recommendation:** Give the queue a maximum size. When it is full, drop records, and write one gap record (step range and count) once the queue drains. Teach the repeated-step and spike detectors to skip gap records.

### Z2-6. The probe is built: freeze it rather than harden it
- **Concerns:** Stage 5 M4, Stage 7 C8; lines 72–74, 95, 244, 288
- **Point:** Two of C8's four asks are already in the code. `probe.py` restores what it patches in `try/finally` blocks, and it computes attention in blocks of query rows of at most 2^26 scores. The other two asks, coordinating ranks and bounding to one head, are new work on a feature that serves a Medium-priority failure with no detector.
- **Recommendation:**
  - Keep the probe opt-in, with no detector.
  - State that v1 supports it only on non-sharded models.
  - Replace lines 72–74 with one sentence and a link to the SDK README.
  - Delete the `_orig_mod` fallback from the plan text. (`capture.py` still checks `_orig_mod` for an unrelated purpose; leave that.)
  - Add no work on probe safety until Gate 0 shows that attention matters.

### Z2-7. Keep the new definitions and tables short, and give each fact one home
- **Concerns:** Stage 6 findings 1, 2, 4–7 and 9; Stage 7 C6, C9, C11, C13, C14
- **Point and sizes:**
  - **Architecture:** about five lines of prose, ending "ingest: deferred":
    - hooks feed a GPU ring buffer;
    - every 50 steps the data moves to the CPU;
    - a writer thread appends to a per-rank JSONL file;
    - `python -m tensorlens` produces the evidence JSON and the `.rrd`;
    - the engineer opens Rerun, and the agent reads the JSON and uses the viewer MCP.
  - **Evidence format:** a short field outline (detector, step, layer or expert, values, logged scalars) and a link to `evidence.py`. No schema in the plan.
  - **Detector table:** make the failure list's summary table its single home. It is stale (Stage 7 C16) and the Gate 0 answer key depends on it. In the plan, list detector names only.
  - **N and M:** the defaults are `every=50`, `replica_every=50` and `probe_every=1000`. Rename the expert count in `exp(H)/N`.
  - **Multi-rank:** close the decision as "per-rank files, analysed one rank at a time in v1". No merging design. The two-GPU NCCL run is Gate 2's multi-rank case.
  - **Gates:**
    - Gate 2: name the capture mode and the hardware (router-only, on the dogfood job's GPU), and say whether the A100 result fails.
    - Gates 3 and 4: one number each.
    - Drop the "cluster-class configuration" requirement.
  - **Risks:** merge by deleting duplicates, not by moving every open gap into a mitigation table.
- **Recommendation:** Accept these at the sizes given, and decline the larger versions.

## Nits

| # | Concerns | Recommendation |
| --- | --- | --- |
| N-a | Stage 7 C5 | Reword the pitch, and replace `pip install` with installing from a clone. Accept "restart from a checkpoint with `capture="full"`" as one line. Reject full hooks behind an on-device flag, and Stage 5 M8(b)'s two compiled variants: they add compile time, CUDA-graph memory and a dependency on Dynamo behaviour. |
| N-b | Stage 5 M8(a), M9 | Defer per-parameter norms from the clipping call until full capture is needed always on. Defer bit-level fingerprints until the two-GPU run shows the 1e-6 tolerance fails. |
| N-c | Stage 7 C10 (capacity) | Replace the 1.3 KB Tier 0 estimate with the measured 26 KB per step: one table cell, not a capacity section. The PyPI name is already an open decision and needs no risk row. |
| N-d | Stage 6 Nits 9, 16, 18–20 | Most of the glossary need goes away once Stage 5 M3's cuts are made. Define Rerun, blueprint, Granite and dogfood inline. Skip the baseline step times and TrainCheck background, which are in the SDK README. |
| N-e | Stage 7 C15–C17 and style Nits | Cheap consistency fixes. Accept them as written. |

## Agree, disagree, reprioritise

| Stance | Findings |
| --- | --- |
| Agree, essential | Stage 6 findings 1–4, 7, 8; Stage 7 C1, C2, C3, C11, C13, C15, C16, C17 |
| Agree, but smaller than proposed | Stage 6 findings 5, 6; Stage 7 B1, C5, C6, C8, C9, C10, C12, C14 |
| Disagree: not needed for v1 | C4's incremental design and alert-latency gate; C2's watchdog and separate process group; C3's spill to disk; C8's probe coordination and per-head bounding; C10's endpoint security design; Stage 5 M8(b) |
| Reprioritised | Stage 5 M6's reorder is still the most important change: Gate 0 on your own job, then the two-GPU run, then release. C2 and C3 move up, because they are the only later findings that are bugs in shipped code. |

## Checked against the code by the main session

- **The plan changed during the review.** Commits `9a5a39f` (probe pass) and `fa1c8e3` (plan progress table) are on `main` and pushed; `origin/main` is at `fa1c8e3`.
- **Z2-4's claim about GradScaler holds for DDP.** `GradScaler` checks for inf or NaN in the gradients after the all-reduce, so every rank sees the same result and skips the same steps. It does not hold where gradients are not all-reduced before the check, or where `start_step` differs between ranks.
