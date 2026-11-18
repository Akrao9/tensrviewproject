# Research papers and SageMaker Debugger review

Nov 17, 2026. A review of the 22 papers in `../researchpapers/`, of AWS's `smdebug` library in `../sagemaker-debugger/` and of its rule configurations in `../sagemaker-debugger-rulesconfig/` (cloned and added later the same day), read against TensorQuill as it is on `main` today. Nothing in the repository was changed. This file is new and untracked.

**State on Nov 18, 2026.** The folder now holds 50 papers and posts (51 files; one paper is there in two versions), and the file has grown in six passes. Sections 1 to 6 are the first pass, of Nov 17: first pages and keyword search over the first 22 papers. Section 7 covers twelve more papers the same way. Sections 8 and 9 are page-by-page reads, of the twelve and of the first 22, and they correct the earlier sections. Section 10 reads eleven more papers and posts added on Nov 18, two of them in part, and corrects section 9 in one place that matters. Where sections disagree, the later one is right; the corrected rows are marked in place. Section 11 reads the three papers section 10 listed as missing and the pages section 10 skipped, and corrects section 10 in eight places. Section 12, a sixth pass, reads the DeepSeekMoE paper and reports a web search for a source for plan row K7; its last part reads the one paper that search turned up and the user then added (Omi, Sen, Farhadi), which makes 50 papers and posts in 51 files. It overturns nothing in sections 1 to 11. Sections 1 to 11 are committed (`2fe4a91`, `d2f67b5`). Section 12 and its marks are not committed. The code was last read at `b11b2c6`.

**Where to start.** Section 1 for the ten proposals. Section 2 for what TensorQuill claims that the papers do not support, with F1 and F2 from section 8. Then the two corrections tables: C1 to C7 in section 8 and D1 to D9 in section 9. Two of the D rows change plan rows K5 and K7. **Section 10 takes one of those back in part: hidden expert collapse (K7) now has a source, row E1.** Section 10 also has three gaps in built kinds, H1 to H3, two of them run on CPU. **Section 11 corrects section 10 (rows J1 to J8): the BF16 attention finding H8 has four limits, and Laguna gives a second reported expert collapse, the first said to end in divergence.** It adds a fourth gap, M1. **Section 12 is the place for plan row K7: two more papers read, no failure case and no threshold found, and two limits on any future rule (rows P4 and P6).**

**How to read it.** Page numbers are PDF pages. "Measured" means the paper reports a number from an experiment. "Claims" means the paper says it without a number, or from one run. What I did not read is listed per paper at the end. TensorQuill facts come from the focused plan, the v3 plan, `sdk/README.md`, the failure list and the code under `sdk/src/tensorquill/`.

**The short version.**

- The papers support the product's premise. Five of them measure a run that fails while its loss looks healthy.
- They do not support four things TensorQuill says about itself. The largest is that silent data corruption is covered by the replica-drift check. It mostly is not.
- Reading `smdebug` turned up one defect in TensorQuill's capture: forward passes made during evaluation are added to the next training step's record. Reproduced on a toy model on CPU.
- No paper shows a failure that calls for breaking the freeze today. Three candidates for later come out stronger, and two new ones appear.
- **Added Nov 18, 2026.** Twelve more papers are covered in sections 7 and 8. Section 8 is the one to read: it has a second capture finding (router load is counted from the hooked scores, which is not the real routing where a model adds a balancing bias first; reproduced on CPU, finding F1), a measurement for finding W9 (F2), and seven corrections to section 7.
- **Added later on Nov 18, 2026.** The first 22 papers have now been read page by page (section 9); one of them, the INT8 inference paper, only in part. Section 9 makes nine corrections to sections 1 to 8. Three overturn a statement, and the plan copied two of those: the one source for hidden expert collapse does not show it, and output-logit growth also happens in stable runs. It finds no new capture defect.
- **Added last on Nov 18, 2026 (section 10).** Eleven more papers and posts, nine read through their main body and two in part (DeepSeek-V3, Motif 3). One earlier correction is taken back in part: Motif 3 names hidden expert collapse and gives monitors for it, so plan row K7 is not "without a source" (E1); it still has no reported case. The routing finding F1 now has two first-hand sources (E2), and "the rank that reports a timeout is often a victim" is first-hand (E3). Three gaps in built kinds: the attention-logit rule misses a slow rise (H1, reproduced on `main`), the replica check does not say which ranks agree (H2, read), and a fault in Adam's state is seen by neither `update_spike` nor the weights-not-updating check (H3, run on CPU; the second rule's condition emulated). No new capture defect.
- **Added last of all on Nov 18, 2026 (section 11).** The three missing papers (Laguna, Mamoda2.5, LLMFT) and every page section 10 skipped. Laguna reports an expert collapse that ran to divergence, kind not stated; it is the second reported collapse after Motif 3's, and hidden collapse still has no reported case (J1). Laguna runs a replica hash check and says it mainly catches silent data corruption, with one example and no counts; it also caught four software faults (M2). A weight corrupted in memory is judged by no rule on its norm (M1, run on CPU). LLMFT's detector was never scored on healthy training. No new capture defect.
- **Added after that on Nov 18, 2026 (section 12).** DeepSeekMoE read in full: it names routing collapse as a risk and reports no collapse and no load number, so it supports no threshold. A web search for a paper showing hidden expert collapse found no failure case. One paper (Omi, Sen, Farhadi, arXiv 2506.14038) measures how alike expert outputs are at checkpoints through training while every expert stays in use. It was then added and read in full: it reports no failed run and gives no threshold, and it corrects four things the first account of it said. K7 still has no reported failure.

---

## 1. Ranked proposals

Buckets: **(a)** allowed now, no new incident kind. **(b)** a new kind, frozen, listed for when the freeze lifts. **(c)** outside the product.

| # | What to build or change | Source | Size | Bucket |
| --- | --- | --- | --- | --- |
| 1 | **Say what the replica-drift check can and cannot see of silent data corruption.** Change failure-list entry 2 from "same detector as 1" to a narrow statement (section 2, finding W1). Add one sentence to the `replica_drift` entry of `kind_checks` and to the agent instructions: equal replicas do not rule out corruption. | SCOUT pp. 1, 3; ByteRobust p. 5; Altenbernd et al. p. 9; LLM-PRISM p. 9; Ma et al. pp. 7–8 | Small | (a) claims |
| 2 | **Keep evaluation forward passes out of training-step records.** The router hook adds to its counts on every forward call and they are zeroed only when an optimizer step commits (`capture.py:262–282`). `session.phase("eval")` changes the heartbeat only (`capture.py:1329–1336`). So a validation pass between two steps lands in the next step's `moe` counts, and the layer hook folds it into that step's activation statistics (`capture.py:118–136`). Zero the buffers when the phase returns to `train`, do it in the Lightning and `Trainer` adapters, and add a test with an evaluation between two steps. **Reproduced on CPU** with a toy one-router model in a throwaway script outside the repository: every step routed 10 tokens, three evaluation batches of 10 ran inside `session.phase("eval")` between two steps, and the next step's record held 40 (`[0, 9, 15, 16]` against `[0, 3, 3, 4]`). Not checked: a compiled model, the adapters, and the layer statistics (the README's "Things to know" says the record keeps the last forward before the step, which would be the training one, so they may be unaffected). The README documents that a hook runs on every forward pass, under gradient accumulation; it does not mention evaluation, and I found no test for it. The probe pass already clears the buffers after its own forward (README, "Probe pass"), so the fix can reuse that path. A loop that never calls `session.phase` has the same problem and no phase change to hook the fix on. | `smdebug/core/modes.py`, `docs/api.md` "Modes", `smdebug/core/save_config.py` (a save configuration per mode) | Small to medium | (a) existing kinds |
| 3 | **Make replica drift name the rank by strict majority, with an "inconclusive" answer.** The plan's change compares each rank with the group's median. SCOUT votes on equal values, names a rank only when the healthy value holds a strict majority, and otherwise returns inconclusive "rather than naming an arbitrary culprit". Also record whether the fingerprints were bit-for-bit equal, beside the `1e-6` tolerance. | SCOUT pp. 6–7, 12 | Small to medium | (a) existing kind |
| 4 | **Correct four statements and two tables** (findings W2 to W6): the reason given for router-only capture, the kurtosis sentence, the TrainCheck figure and what it supports, the paper's title, and add SCOUT's `lm-resiliency` and SageMaker Debugger to the alternatives table. **Added Nov 18 (section 8, F1):** also say how a router that returns scores is read. The hook counts the top-k of the hooked scores. Where the model adds a bias or noise to the scores before choosing, that is not the real routing, and a balanced model is reported as collapsed. An `indices=` function reads it correctly. Both were run on a toy on CPU. | 2606.28116 pp. 3, 5–6; Wang et al. (arXiv 2408.15664) p. 4; FP8 paper p. 5; TrainCheck pp. 5, 10–11, 16; `lm-resiliency/README.md`; `sagemaker-debugger/README.md` | Small | (a) claims |
| 5 | **Run the detectors against public fault injectors and bug sets** (section 5). First: SCOUT's injector on a multi-rank job for result 1, and TrainCheck's 20 reproduced bugs to state honestly how many of them the 12 invariants catch. | `lm-resiliency/docs/fault_injection.md`; TrainCheck `docs/ae-eval-s5.1-silent-issue-detection.md` | Medium | (a) tests and claims |
| 6 | **Stall evidence: group the stopped ranks by stack and say which group is the odd one out.** `_describe_stuck` already puts ranks at the same place together in one sentence (`evidence.py:4335`). ByteRobust's step is to call the largest group the waiting majority and the rest the suspects. Put the groups and their sizes in the incident, and say so in the agent instructions. Second part: write an end marker when a local run closes, as `smdebug` does, so a finished file and a file that stopped growing do not read the same. **Added in section 9 (G2, D3):** say that the stacks cover the rank's own process and not its data-loader workers; one paper also uses repeated snapshots for slow jobs. | ByteRobust p. 8; SCOUT p. 10; `smdebug/core/access_layer/utils.py` (`END_OF_JOB_FILENAME = "training_job_end.ts"`) | Small | (a) evidence |
| 7 | **Threshold work on recorded and from-scratch runs.** (i) `activation_drift` compares with a module's first 50 records; on a model trained from scratch that baseline is the untrained network, and the papers predict the outlier ratio then grows far past 3× (finding W7; a toy run came to 2.6× of the 3× cutoff). Run it on nanoMoE or nanochat before result 2. (ii) Record a median-and-MAD score beside each ratio, so result 2 can compare rules without new captures. (iii) Decide whether single-step gradient spikes are "ordinary" (finding W8). | Massive Activations pp. 1–3; FP8 paper pp. 5–6; SPAM pp. 2–5; EnergyCal p. 5; TrainSDC pp. 5, 10 | Medium | (a) existing kinds |
| 8 | **Two more numbers from the probe pass, with no incident:** the language-model head's output-logit norm and mean, and the gap between its BF16 and FP32 logits on the same hidden states; and the per-token kurtosis of each block's output. Recorded the way `weight_trends` is: numbers only. | Chen et al. (quantization) pp. 2, 7–8; FP8 paper pp. 5–6 | Medium | (a) evidence |
| 9 | **A check for corruption that the replica-drift check cannot see.** Cheapest form: compare one per-layer activation statistic across data-parallel ranks at the same step and flag the rank far from the others' median (TrainSDC's residual gain guard). Strongest form: SCOUT's sampled replay of one layer on every replica with hashed outputs. | TrainSDC pp. 4–5, 10–11; SCOUT pp. 6–8, 12 | Medium (compare) to large (replay) | (b) new kind |
| 10 | **Output-logit growth in the language-model head** as a kind, once proposal 8 has numbers from real runs. None of the 32 kinds watches the head; `attention_logit_growth` is about attention. **Corrected in section 9 (D2):** the logit norm also grows in the paper's stable FP32 runs, so the thing to judge is the BF16-to-FP32 gap, not the growth. | Chen et al. (quantization) pp. 1–2, 5–8 | Medium | (b) new kind |

**Outside the product (c), noted and not proposed:** host-metric fault detection (Minder), automatic eviction, restart and checkpoint certification (ByteRobust, MegaScale, SCOUT's checkpoint gate), optimizer changes that remove spikes (SPAM, EnergyCal, SGD on the head), FP8 architectures (FOG), routing replay for RL (R3), routing control for agents (Li et al.), INT8 inference checks (SProbe), `smdebug`'s loss, overfit and class-imbalance rules, its batch-size and dataloader sizing hints, its profiler report and its stop-training action. An agent's diagnosis may name a published remedy; TensorQuill should not apply one.

### How the papers bear on the four candidates the plan already names

| Candidate | What the papers add | Standing after this review |
| --- | --- | --- |
| Per-token routing entropy | The only one with a direct study: 2606.28116 measures it on an 8B MoE under learning-rate and batch-size sweeps (pp. 5–6, 8). The paper gives no threshold, no lead time against the loss and no false-alarm rate. Healthy public checkpoints sit anywhere from 49% to 94% of the maximum (Table 1, p. 3), so only a self-baseline can work. It needs the router's logits; a router that returns indices cannot give it. | First. Measure its cost under `torch.compile` before anything else |
| Update size per parameter | Altenbernd et al. compute it from Adam's own state with no copy of the weights, flag a jump against the warm-up mean, and report about 1% cost and no false alarm on 21 clean runs of 60M to 1.3B models (pp. 6–9). Rotational Equilibrium gives a reference value for decayed weights under AdamW, `sqrt(2·lr·wd·(1−β1)/(1+β1))` per step (Table 1, p. 4). **Corrected in section 9 (D8):** that is the rotation; the RMS update that the kind measures is `lr·sqrt((1−β1)/(1+β1))` in the same table. 2606.28116 warns that update size alone missed its fault: 104% of baseline while the spectrum was at 51% (p. 7). AWS's `WeightUpdateRatio` rule is configured to save the weights themselves every 500 steps (`ruleConfigs.json`): a full copy, and a ratio between saved copies, not per step. | Second. The v3 note that a magnitude "has no threshold to judge it by" is now out of date |
| Slow ranks | Confirms the plan: step time cannot find one (SCOUT p. 3, Lin et al. p. 5, MegaScale p. 12). Adds that most slowness is workload, not hardware: sequence-length imbalance, the last pipeline stage, Python garbage collection (Lin et al. pp. 9–12). AWS's `StepOutlier` profiler rule flags a step duration beyond 3 standard deviations after ignoring 10 outliers; its docstring does not say whether that is per worker. | Third, and as an observation. Record tokens per rank per step beside the collective timings |
| Hidden expert collapse | Weakest support here. One case report finds experts that all receive tokens and compute the same thing, output cosine above 0.99 (Sha, pp. 1, 12). No paper in the folder measures "an expert adds less and less". **Corrected in section 9 (D1):** the case report's experts were frozen identical copies, so their equal outputs are the starting state. No paper in the folder measures experts growing alike in training. | Last. Needs expert outputs, which nothing captures today |

Proposal 9 belongs between the second and third rows if TensorQuill wants to keep saying it helps with silent data corruption.

---

## 2. Wrong or unsupported: claims and thresholds

Ordered by how much they matter. Two more came from the full read of the second batch and are written up in section 8: F1, router counts that are not the real routing (it would stand second in this list, after W1), and F2, a measurement for W9.

**W1. "Silent data corruption on a faulty GPU: same detector as 1" is not supported.** **See section 10, E5 and H2, for where a fault can sit and which of those places the check sees.**
The failure list (entry 2, and its summary row) says the cross-replica fingerprints cover it. The papers say corruption does not show up there:

- A corrupted gradient is reduced with the others and the same wrong result goes back to every rank. "The corrupted value no longer identifies its source" (SCOUT p. 1; also p. 3). ByteRobust: "a single corrupted gradient from one node can contaminate the global parameter update across all participating workers" (p. 5). Altenbernd et al. say the same and add that they did not test it (p. 9).
- So after the step every data-parallel replica holds the same wrong weights. Their fingerprints agree.
- LLM-PRISM found faults in weight gradients had little effect on perplexity, and explains it by momentum and by averaging across GPUs, which dilutes one faulty rank by 1/N; faults in the forward output and in the input gradients "act on local computation before any cross-rank communication" (p. 9). The 1/N is the paper's explanation, not a separate measurement. Either way the damage reaches the weights through the shared update.
- The "parameters drifted gradually" result the failure list quotes (Ma et al. p. 7) is drift between a faulty node's run and a separate healthy run of the same job. It is not drift between replicas inside one job. LLM-PRISM's weight divergence is also against a fault-free run (p. 7).
- What the papers use instead: recompute and compare. SCOUT replays a sampled layer on every replica and compares hashes (pp. 6–8). Ma et al. propose a shadow replica whose gradients are compared before the reduction (p. 8) and found checksummed matrix multiplication unreliable (p. 8). TrainSDC compares a per-layer activation statistic across ranks (p. 5). Altenbernd et al. watch the size of the optimizer's update (pp. 6–7).

What `replica_drift` does catch: a rank that applies a different update, or whose weights change after the reduction. That is the BLOOM bug, a broken synchronisation hook, and corruption inside the optimizer step or in memory. That is worth saying plainly. The failure list's own sentence, that SDCHunter compares "outputs of replicas that saw the same batch", already describes the other method.

**W2. The reason given for router-only capture overstates the paper.**
`sdk/README.md` ("Capture-mode decision") says a 2026 paper "proposes router-level monitors such as routing entropy, which router-only capture records". The paper (2606.28116) measures the entropy of each token's full softmax before the top-k choice (eq. 9, p. 5). TensorQuill records the entropy of the load: `_router_stats` computes it from the per-expert token counts (`capture.py:1607–1615`). The paper says these differ: probabilities "can become more concentrated while the selected expert set and aggregate assignment counts remain unchanged", and "aggregate load balance does not exclude this state, because different tokens can saturate toward different experts" (pp. 5–6). So the paper does not show that what router-only capture records is an early signal. Its early-warning measurement (about 4,000 steps before the loss turned, p. 7) is for attention update spectra under an injected FlashAttention fault, in one run.

The v3 plan (line 346) also lists "router weight similarity" among the paper's early warnings. Version 3 of the paper argues against it: similarity can rise with no change in routing, and healthy checkpoints range from about 0 to 0.51 (p. 3).

**W3. "`abs_max / std` tracks the same outliers" as kurtosis is not supported.**
The FP8 paper's kurtosis is taken per token across the hidden dimension, without centring, and then averaged over all tokens (p. 5). TensorQuill's ratio is over a whole block output and is set by the single largest value. Massive Activations shows that value sits at the first token and a few delimiter tokens, in two or so fixed dimensions (pp. 1–3). A per-token average barely sees that. The two statistics answer different questions. The paper's early signal was on the query, key and value tensors, which TensorQuill does not hook (p. 6). Two TensorQuill texts also disagree: the failure list says a fourth moment is "as cheap as the statistics TensorQuill already computes"; the README says it "would need a second pass".

**W4. The TrainCheck figure, and what it supports.**
The focused plan says always-true rules caught 18 of 20 and older methods "caught at most 3". The technical report (pp. 10–11) says the signal-based detectors (spike, trend, and three anomaly detectors on loss and accuracy) "collectively only detect 2 errors", and PyTea/NeuRI detect 1. "At most 3" is consistent with that but imprecise: for signal monitoring, which is the comparison the plan draws, the figure is 2.

More important: the 18 come from invariants inferred from clean reference runs, at the level of API calls and variable updates (Table 2, p. 6). Most of the 20 bugs are of that kind: a missing `zero_grad`, a wrong autocast type, dropout left on in evaluation (Table 3, p. 11). By my reading, TensorQuill's own kinds could plausibly catch about five of the 20 (the three out-of-sync cases, and the two where the model stops updating). That is untested. (Section 9, D7: with the bug table read in full the count is up to seven.) The result supports the idea of invariants. It is not evidence for TensorQuill's 12. Two more points from the same paper help TensorQuill: TrainCheck's instrumentation "interferes with JIT compilation tools like torch.compile" (p. 16), and it leaves optimisation problems out of scope (p. 5), which is where TensorQuill's anomaly kinds sit.

**W5. The title.** The plan calls 2606.28116 "Mechanism-driven monitors for LLM training instability". The PDF in the folder (v3, Oct 18, 2026) is titled "Mechanism-Aligned Observability for Training Instability of Large Language Models". An earlier version may have carried the other title; I could not check.

**W6. "None of the three debuggers detects" multi-GPU failures leaves out two tools.**
The statement is true of the three debuggers named. But SCOUT ships as open source in `lm-resiliency` (BSD 3-Clause, cloned beside this repo). Its README says it localises hangs, stragglers and silent data corruption to a rank, with "less than 1% amortized overhead". The paper itself says throughput overhead was not measured (p. 13), so that README figure has no published measurement behind it. SCOUT does not look at layers, routing, attention or gradients, so it is not a per-layer debugger. It is still the closest tool on claim 1 and it is missing from the alternatives table. SageMaker Debugger is missing too (section 4).

**W7. `activation_drift` is likely to fire on a healthy large model trained from scratch. Predicted, not shown.**
The rule compares the recent median of `std`, or of `abs_max / std`, with the module's first 50 records, and reports a 3× move (`evidence.py:76`, `754–762`). It was calibrated on pretrained Granite and SmolLM2, where the outliers already exist: 740 to 840 times the standard deviation in healthy Granite. Massive activations are learned during pretraining (Massive Activations p. 2), reach thousands of times the median (pp. 1–3), and the FP8 paper shows kurtosis growing through healthy training (pp. 5–6, Fig. 4). At initialisation the ratio is a few units. So on a real from-scratch run the ratio should pass 3× its first 50 records as a matter of course.

I tried it on a toy and it did **not** fire: a 4-block, 96-wide character model trained from scratch on CPU for 3,000 steps with full capture and no fault (loss 5.65 to 1.23) reported no incident. The largest move from the first-50 baseline was 2.62× in `std` (last block) and 2.05× in `abs_max / std` (from about 3.5 to 6.3). That is under the cutoff, with far less margin than the 1.27× seen on pretrained Granite. A toy this small does not grow massive activations, so it does not settle the question. nanochat and nanoMoE, the candidates for result 2, are from-scratch runs and would.

**W8. Single-step gradient spikes may not be "ordinary".**
The plan found 277 to 473 single-step jumps of 5× in one parameter on each healthy Granite capture and built the three-in-a-row rule around them. SPAM reports the same thing in LLaMA and Pythia training, frequent short spikes with LayerNorm weights the worst (p. 3), which matches TensorQuill's captures. But SPAM also measured that zeroing those spikes improves final perplexity (p. 5), and that one spike stays in Adam's second moment for more than 200 steps in a simulation (p. 5). So "healthy" is the wrong word for those steps; "not reported" is the accurate one. SPAM's threshold was 50× on single elements, not 5× on a norm, so the numbers are not comparable. (Section 9, D6: the perplexity result is one 60M model and depends on the threshold.)

**W9. The `1e-6` drift tolerance.** The failure list calls it a guess and worries that nondeterminism may exceed it. The papers point the other way. Replicas that apply the same reduced gradient should be bit-for-bit equal: SCOUT compares sampled weights after the real optimizer step with exact signatures (p. 12), and Ma et al. got identical results on any two healthy nodes under a deterministic compiler (p. 3). A tolerance above zero can hide small real drift. Record exact equality in the two-GPU run and see.

**W10. `stalled_weights` and split weights.** FlashOptim stores a low-precision weight plus a correction held by the optimizer, in place of a float32 master copy (pp. 2–3). Training runs on the 16-bit weight, and "the correction term ρ remains local with the optimizer states" (p. 5). So the visible tensor can stay unchanged for many steps while the true weight moves. TensorQuill reads the visible tensor. Its incident already says "evidence, not a verdict"; it should also say when the optimizer holds a correction. Not checked against FlashOptim's code, which is not in the folder, and not tried.

**What the papers confirm.**

| TensorQuill choice | Support |
| --- | --- |
| The loss can look healthy while the run fails | Ma et al. p. 7 (loss identical on faulty nodes); LLM-PRISM pp. 7, 10; 2606.28116 p. 7; FP8 paper p. 6 (kurtosis off trend at 3B tokens, gradient norm at 12B, loss at 15B, one run); TrainCheck p. 4 |
| Judge each module against its own history, not a fixed outlier cutoff | Massive Activations pp. 1–3; 2606.28116 Table 1 |
| Three ranks to name a drifted rank | SCOUT p. 6 |
| Step time cannot find a slow rank; use per-collective timing | SCOUT p. 3; Lin et al. p. 5; MegaScale pp. 8, 12 |
| 5× as the factor for a gradient or loss jump | ByteRobust uses "5× increase in loss/gradient norms" in production (p. 6). A practice, not a validation |
| Silence thresholds of 300 s and 900 s | Same range as ByteRobust's 10 minutes without traffic (p. 6) and Minder's 4 minutes of persistence (p. 8). NCCL's own timeout is 30 or 60 minutes (ByteRobust p. 5) |
| A heartbeat with a status, and a watcher outside the training process | MegaScale pp. 6–7; SCOUT pp. 2, 10. SCOUT goes further: a separate CPU process per rank, because a thread inside a blocked trainer may not report. TensorQuill's README already states that limit |
| bf16 weights whose updates round away are a real failure | Sha p. 14 (Table 2; weak, see section 9, D5: its stuck case does not need rounding to explain it); Chen et al. pp. 2, 7–8 ("BF16 true" fails first); FlashOptim p. 3 (1/256 relative bound) |
| Anomalies do not fire the trigger by default | SCOUT p. 4 calls own-history references "unstable" for a workload that changes phase |
| The probe's recompute of attention | 2606.28116 p. 2 calls a recompute pass "unacceptable in a large training run". That is about the training path. TensorQuill's is one fixed batch every 1,000 steps, measured at 0.22% to 0.34% |

---

## 3. The papers, one row each

### What the plans already cite, and what the full paper adds

Six of the 22 are cited somewhere in the plans, the failure list or the SDK README. The other 16 are not. The failure list uses the massive-activations definition without citing that paper.

| Paper | Cited in | What the plan took | What the full text adds |
| --- | --- | --- | --- |
| 2606.28116 | Focused plan, v3 plan, SDK README. Read "from search excerpts only" | Spectral entropy of weight updates, router weight similarity and per-token routing entropy as early warnings; loss and norms are late | The entropy is per token, before top-k, and the paper says load statistics can miss it (W2). It argues against router weight similarity. The early-warning number is for attention updates under one injected fault, not for routers. Update size alone missed that fault. Signals are judged against a reference run. No threshold, no false-alarm rate, no code |
| TrainCheck (2506.14813) | Focused plan, v3 plan, failure list, SDK README | 18 of 20; baselines at most 3 (from slides); it hashes parameters and checks that a step changes them | Signal detectors caught 2. The invariants are inferred from reference runs and are mostly API-level (W4). Under 2% false positives; overhead under 2% on realistic jobs. It does not work under `torch.compile` and leaves hyperparameter instabilities out of scope |
| SCOUT (2608.11034) | Focused plan | A rank is named when it disagrees with a strict majority, which takes three ranks | An explicit "inconclusive" answer; exact signatures, no tolerance; corruption is found by replaying a layer, because reduced gradients hide the source (W1); timing needs equal work and a median-and-MAD test; an observer in a separate process; the code and a fault injector are public (W6). Evaluated at 16 GPUs with injected faults only |
| Ma et al. (2502.12340) | Failure list, entry 2 | Loss stays close to healthy while parameters drift; fine-tuning spikes and one zero-accuracy run | The drift is against a separate healthy run, on one node, with no data parallelism (W1). Gradient noise is at most 5.1%. Checksums did not detect reliably; the authors recommend recomputation |
| Altenbernd et al. (2604.00726) | Failure list, entry 2 | Exponent and sign flips cause NaNs, short spikes or lasting divergence | In this paper only exponent bits mattered. Backward faults do the damage. A detector on update size from Adam's state with about 1% cost. The infinite-norm clipping trap. Single GPU; distributed propagation not tested |
| FP8 paper (NeurIPS 2025; the failure list cites it as 2505.20524) | Failure list, entry 8 | Sustained low kurtosis predicted stable FP8 training | How kurtosis is defined (per token, uncentred, averaged) and where it is measured (W3). One failed run with the 3B, 12B and 15B token marks. Larger models diverge later |

### One row per paper

Bearing: **N** nothing, **C** confirms a choice, **X** contradicts or weakens a claim, **S** suggests a change. Proposal numbers refer to section 1.

| # | Paper | What it shows | Bearing | Change it suggests |
| --- | --- | --- | --- | --- |
| 1 | **Mechanism-Aligned Observability for Training Instability of Large Language Models.** Huang, Wang, Fang, Huang, You, Wang, Wu, Zheng. 2026 (arXiv 2606.28116v3). Two identical copies in the folder | Measured on one 8B MoE (128 experts, top-6). An injected FlashAttention fault (low fraction bits cleared to imitate lost precision, Appendix C; not a hardware fault) moved the spectrum of 500-step weight updates to 51% and 87% of baseline at step 15,000, with raw weights at 96–97% and update norm at 104%; the loss turned near step 19,000 (p. 7). Per-token routing entropy fell earlier and deeper at higher learning rates and smaller batches (p. 8). No threshold, no false-alarm rate, no code found | X, S | W2, W5; first freeze-lift candidate; update size alone is not enough |
| 2 | **Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks** (TrainCheck). Jiang, Zhou, Xu, Liu, Xu, Huang. 2025 (arXiv 2506.14813v1; OSDI '25) | Measured: 18 of 20 reproduced silent bugs caught within one iteration by inferred invariants; signal detectors on loss caught 2 (pp. 10–11). Under 2% false positives on 63 clean programs (pp. 13–14). Overhead usually under 2%, up to 1.6× on toy jobs (p. 15). Breaks under `torch.compile` (p. 16) | X, C | W4; proposal 5 |
| 3 | **SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training.** Zhuang Wang. 2026 (arXiv 2608.11034v1) | Names the faulty rank by strict majority among equivalent replicas, for hangs, stragglers and corruption. Measured on at most 16 A100s with injected faults: 344 of 344 numerical faults localised, 30 of 30 clean runs quiet (pp. 12–13). Overhead, false positives at scale and recovery time not measured (p. 13) | C, X, S | W1, W6; proposals 3, 5, 9 |
| 4 | **Understanding Silent Data Corruption in LLM Training.** Ma, Pei, Lausen, Karypis. 2025 (arXiv 2502.12340v1) | Measured on 15 real faulty nodes: gradient noise at most 5.1% of the true norm (p. 6); pretraining loss identical to the healthy node while weights drift from the healthy run (p. 7); fine-tuning spikes on some nodes, one ending at zero accuracy (p. 8). Checksummed matmul did not detect reliably (p. 8). One node, tensor parallel only | X, C | W1 |
| 5 | **Exploring Silent Data Corruption as a Reliability Challenge in LLM Training.** Altenbernd, Wiesner, Kao. 2026 (arXiv 2604.00726v1) | Measured by flipping bits in GPU matmul instructions on 60M to 1.3B models: only exponent bits matter (p. 3); backward faults cause loss bumps (p. 4); an infinite gradient norm clips to a zero update, and without clipping freezes Adam (p. 5). A detector on update size from Adam's state: no false alarm on clean runs, about 1% cost (pp. 8–9). Single GPU only | S | Update-size candidate; name the "non-finite then stalled" pattern in the agent instructions |
| 6 | **LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training.** Tyagi, Hukerikar, Saxena, Huang, Shirvani, Tung, Zhu. 2026 (arXiv 2604.10390v2) | Measured over 7,664 runs of GPT-2 small and medium: 57–76% of runs unaffected; FP8 runs either recover or crash (41.9% crash), FP16 degrades silently by up to about 600% perplexity (pp. 7, 9–10). Four failure modes, one of them loss-scale collapse (p. 7) | C, X | W1; supports `loss_scale_trouble` as a real fault path |
| 7 | **TrainSDC: Characterizing and Mitigating Silent Data Corruption in Large Language Model Training.** Xia, Xu, Yun, Lin, Liu, Li, Zhuo. 2026 (arXiv 2608.30769v1) | Measured: faults on the query/key path do lasting damage and "can escape both loss-based and activation-magnitude monitoring" (pp. 3–4). A guard that compares each rank's residual gain with the other ranks' median: 1 alarm in an 814-step fault-free trace on Llama 3.2-1B and none on Qwen3-0.6B; recall 43% to 86% on Llama and 41% to 78% on Qwen, by fault size; 1.65% and 6.76% overhead (pp. 7, 10–11) | S | Proposal 9; `activation_spike` cannot see query/key faults |
| 8 | **Massive Activations in Large Language Models.** Sun, Chen, Kolter, Liu. 2024 (arXiv 2402.17762v2; COLM 2024) | Measured on trained models: a handful of activations above 100 and about 1,000× the median, in fixed dimensions, at the first and delimiter tokens (pp. 1–3); zeroing four of them breaks LLaMA2-7B (p. 2) | C, X | W3, W7; proposal 7 |
| 9 | **Towards Fully FP8 GEMM LLM Training at Scale.** Hernández-Cano, Garbaya, Schlag, Jaggi. 2025 (NeurIPS 2025) | Measured: FP8 for every matmul in the block, 35–43% faster than BF16 Llama3 (p. 7), up to 43% at 8B (p. 2). One failed run where query/key/value kurtosis left its trend at 3B tokens, the gradient norm at 12B and the loss at 15B (p. 6). Larger models diverge later (Table 2, p. 8) | X, S | W3; proposal 8 |
| 10 | **Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks.** Kosson, Messmer, Jaggi. 2024 (arXiv 2305.17212v4; ICML 2024) | Derived and checked on ResNets and GPT-2: with weight decay a weight vector's per-step rotation settles at a value set by the hyperparameters; for AdamW `sqrt(2·lr·wd·(1−β1)/(1+β1))` (Table 1, p. 4). Not a failure study | S | A reference value for the update-size candidate |
| 11 | **Understanding Stragglers in Large Model Training Using What-if Analysis.** Lin, Jiang, Song, Zhao, Yu, Wang, Wang, Shi, Shi, Jia, Liu, Wang, Lin, Liu, Panda, Li. 2025 (arXiv 2505.05713v2; OSDI '25 by its acknowledgments) | Measured on five months of ByteDance jobs: 42.5% of jobs at least 10% slower; waste 7.8% at the median and 45% at the 99th percentile (pp. 2, 7–8). Causes are mostly workload imbalance, not a few bad machines (pp. 9–12) | C, S | Slow ranks as an observation; record tokens per rank |
| 12 | **Minder: Faulty Machine Detection for Large-scale Distributed Model Training.** Deng, Shi, Jiang, Zhang, Zhang, Zhang, Li, Song, Zhu, Liu, Li, Wang, Lin, Ye, Yu. 2025 (arXiv 2411.01791v2) | Measured in production: finds the faulty machine from host metrics in 3.6 s, precision 0.904, F1 0.893, on 150 confirmed faults (pp. 1, 8). Compares machines with each other and requires 4 minutes of persistence | C | None; host metrics are outside the product |
| 13 | **Robust LLM Training Infrastructure at ByteDance** (ByteRobust). Wan, Liu, Song, Wang, Zhang, Sheng and 29 others. 2025 (arXiv 2509.16293v4; SOSP '25) | Measured over three months: 38,236 explicit and 5,948 implicit failures; job hangs 9.9%, NaN 0.3% of incidents (pp. 2–3). Groups stack traces of all ranks; the dominant group is healthy, the rest are suspects (p. 8). 97% effective training time on 9,600 GPUs (p. 1) | C, S | Proposal 6 |
| 14 | **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs.** Jiang, Lin, Zhong and 29 others. 2024 (arXiv 2402.15627v1; NSDI '24 by its acknowledgments) | Measured: 55.2% utilisation on 12,288 GPUs (p. 1); over 90% of faults handled automatically (p. 11); about 0.5% of machines much slower, found by per-rank timing of code segments (p. 8) | C | None beyond the slow-rank design |
| 15 | **The Role of Quantization in Loss Spikes.** Chen, Somasundaram, Li, McLeish, Kirchenbauer, Goldstein, Goldblum. 2026 (arXiv 2610.38277v2) | Measured on 28M to 1.4B models at high learning rates: late, unrecoverable divergence follows growth of the output logits; the BF16-to-FP32 logit error grows by orders of magnitude first; an FP32 head stays stable across five seeds (pp. 2, 7–8) | S | Proposals 8 and 10 |
| 16 | **SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training.** Huang, Zhu, Jin, Liu, Wang, Liu. 2025 (ICLR 2025; arXiv 2501.06842v2) | Measured on 60M to 1B models: gradient spikes up to 1,000× typical, frequent, worst in LayerNorm weights (pp. 2–3); zeroing them improves perplexity (p. 5) | C, X | W8 |
| 17 | **EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training.** Masato Fujitake. 2026 (arXiv 2610.37062v1) | Measured on one 4B MoE, three seeds: loss-spike burden down 99.51%, episodes from 37.3 to 3.3 (p. 5). The author says the comparison does not isolate the cause (pp. 2, 5, 7). Defines a spike as a loss above `max(m+0.5, 1.2m, m+8σ)` over the last 101 losses (p. 5) | S | A robust spike definition for proposal 7 |
| 18 | **Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers** (R3). Ma, Zhang, Zhao, Song, Wang, Sui, Luo. 2026 (arXiv 2510.11370v3) | Measured: float noise flips a near-tied top-k choice, so the training and inference engines route differently; policy divergence 1.5e-3 for the MoE against 6.4e-4 for a dense model (pp. 4–5, 8) | N | RL is a frozen pack. One fact for `compare_runs`: two runs that should match can part first at a router |
| 19 | **Sparse Mixture-of-Experts Routing in Visual Diffusion Transformers: Diagnosis, Boundary Calibration and Evolutionary Roadmap from Routing Collapse to Selective Deadlock.** Haiying Sha. 2026 (arXiv 2605.19378v1) | A case report on one 5B video model: experts gave the same output (cosine above 0.99; they were frozen identical copies, see section 9, D1), a third of layers fell to one expert and a larger auxiliary loss did not help (pp. 12–13); bf16 updates truncated to zero (p. 14). One author, one setup, not peer reviewed | C | Hidden-collapse candidate; confirms per-layer reporting |
| 20 | **Structuring MoE Expert Selection for Agentic Reinforcement Learning.** Li, Hu, Hsieh, Chowdhury, Tuzel, Vemulapalli. 2026 (arXiv 2610.07332v1) | RL post-training of agents. Measured on AppWorld and AutomationBench with Qwen3 MoE models: routing-control losses move four RL algorithms' scores by between −2.4 and +12.2 points, mostly up (Table p. 8) and raise rollout throughput from 224 to 316 tokens per GPU-second (p. 8). Routing control alone made training collapse; it stayed stable for 200 steps with a gate on policy entropy (p. 7). No limitations section in the pages read | N | None. RL is a frozen pack |
| 21 | **FlashOptim: Optimizers for Memory-Efficient Training.** Gonzalez Ortiz, Gupta, Rinard, Blalock. 2026 (arXiv 2602.23349v2) | Measured on image models, GPT-2 pretraining and Llama-3.1-8B fine-tuning: AdamW memory from 16 to 7 bytes per parameter, peak memory down 36%, a loss curve that tracks AdamW for 20,000 updates (pp. 1, 6–7). Each weight is split into a 16-bit part the model uses and a correction the optimizer keeps (p. 5) | S | W10 |
| 22 | **Syndrome Decoding for Silent Data Corruption in Quantized Integer GPU Arithmetic.** Napolean, Srivastava, Periathambi. 2026 (arXiv 2609.19743v1) | Measured for INT8 inference: every injected fault detected, at 11% to 49% of the matmul's time and 30% of throughput in one model (p. 1) | N | None; inference only |

---

## 4. SageMaker Debugger (`smdebug` and its rules-config package)

Read from two clones. `sagemaker-debugger` at commit `3b82de31` (Oct 31, 2023), version 1.0.34, in part (section 6 says which parts). `sagemaker-debugger-rulesconfig` at commit `afeb292` (May 12, 2021), version 1.0.2, in full; it is about 1,200 lines and two JSON files.

**Licence.** Both are Apache 2.0. `smdebug` carries `NOTICE`, `THIRD-PARTY` and `LICENSE-THIRD-PARTY` files; the rules-config package carries a `NOTICE` (copyright 2019, Amazon). This review copies no code from either. If TensorQuill ever takes code from them, the file must say so, keep the copyright and licence notice, and state the changes. The same holds for `lm-resiliency` (BSD 3-Clause) and TrainCheck (Apache 2.0). Ideas need no notice, but the plan should name where they came from.

**Where the built-in rules are.** In three places, and two can be read.

- **`smdebug`** has the rule interface and the rule runner, and no built-in rule. Its README says "the built-in rules that AWS provides are only available for SageMaker training".
- **The rules-config package** has every rule's name and default configuration.
  - For the 21 debugger rules that is the rule's name and, for nine of them, a collection to save. There is no threshold and no logic (`debugger_rules/rule_config_jsons/ruleConfigs.json`).
  - For the profiler rules it is one class each, with default thresholds and a docstring (`profiler_rules/rules.py`).
- **The rule code** runs in AWS's container images. It is in neither clone.

What that settles, and what it does not:

- **Names: verified** for every rule below, at that May 2021 commit. It may not be AWS's list today.
- **Profiler defaults: verified.** I quote the value in the code. Twice the docstring gives another number, and the row says so.
- **What a debugger rule computes, and its thresholds: still unread.** For those rows the verdict rests on the rule's name and is marked "by name". `smdebug`'s docs show one simplified example, a gradient tensor whose mean absolute value is under `1e-7` (`docs/analysis.md`).

### Rule map: the 21 debugger rules

"Group" is the package's own grouping (`debugger_rules/_ruleGroups.py`): universal means all four frameworks, deep learning means TensorFlow, MXNet and PyTorch. "Saves" is the collection the rule's configuration asks for.

| Rule | Group; what its configuration saves | TensorQuill | Note |
| --- | --- | --- | --- |
| `VanishingGradient` | Deep learning; `gradients` every 500 steps | **Lacks it** | Per-parameter gradient norms are recorded in full capture; no rule reads them for being too small. `stalled_weights` sees only the end state. This is the one rule with an example in `smdebug`'s docs |
| `ExplodingTensor` | Deep learning; nothing named | **Has it** (by name): `non_finite` | Neither clone says what "exploding" means. If it is NaN and infinite values, `non_finite` covers activations, gradients, the clipping norm and replica weights. If it also means large finite values, `grad_spike` and `activation_spike` are the nearer kinds |
| `LossNotDecreasing` | Universal; `losses` every 500 steps | **Lacks it, on purpose** | Scalars stay with MLflow and W&B. Bucket (c) |
| `PoorWeightInitialization` | Deep learning; tensors matching `.*relu_input` every 500 steps | **Lacks it** | Rules that compare with a run's own history cannot judge step 0 |
| `DeadRelu` | Deep learning; tensors matching `.*relu_output` every 500 steps | **Has it differently** (by name): `dead_activations` | A rise in a block's zero fraction above its own baseline. TensorQuill needs no tensor of a given name |
| `SaturatedActivation` | Deep learning; nothing named | Lacks it (by name) | |
| `AllZero` | Universal; nothing named | Has it differently (by name): `dead_activations` | The zero fraction of each block's output |
| `UnchangedTensor` | Universal; nothing named | **Has it differently** (by name): `stalled_weights` | A float64 norm unchanged on three checks, with optimizer ownership and dtypes |
| `TensorVariance` | Deep learning; nothing named | Has it differently (by name): `activation_spike`, `activation_drift` | `std` judged against the module's own history |
| `WeightUpdateRatio` | Deep learning; `weights` every 500 steps | **Lacks it, on purpose** | The update-size candidate. The rule is handed whole copies of the weights 500 steps apart: its configuration names no reduction, and with none `smdebug` saves the raw tensor (`smdebug/core/hook.py:189–190`) |
| `SimilarAcrossRuns` | Universal; nothing named | Has it differently (by name): `compare_runs` and its divergence step | |
| `StalledTrainingRule` | Universal; nothing named | **Has it, further** (by name): `job_stalled`, `rank_silent` | Phases, heartbeats, per-rank stacks and Flight Recorder dumps |
| `Overfit` | Universal; nothing named | Lacks it, on purpose | Evaluation scalars. Bucket (c) |
| `Overtraining` | Universal; nothing named | Lacks it, on purpose | Evaluation scalars. Bucket (c) |
| `ClassImbalance` | Universal; nothing named | Lacks it, on purpose | Data content. Bucket (c) |
| `Confusion` | Universal; nothing named | Lacks it, on purpose | Evaluation results. Bucket (c) |
| `CheckInputImages` | Application; tensors matching `.*hybridsequential0_input_0` every 500 steps | Lacks it, on purpose | Data content |
| `NLPSequenceRatio` | Application; tensors matching `.*embedding0_input_0` every 500 steps | Lacks it, on purpose | Data content |
| `TreeDepth` | XGBoost; nothing named | Lacks it | XGBoost |
| `FeatureImportanceOverweight` | XGBoost; `feature_importance` every 500 steps | Lacks it | XGBoost |
| `CreateXgboostReport` | In no group; `metrics`, `feature_importance`, `predictions` and `labels` every 5 steps, `hyperparameters` at step 0 | Lacks it | A report, not a check. Bucket (c) |

### Rule map: the profiler rules

Defaults are the constructor's (`profiler_rules/rules.py`). `patience` is the number of datapoints a rule collects before its first judgement; the sampling interval is not in this clone, so it cannot be turned into steps or seconds here. Utilisation thresholds are validated as percentiles between 0 and 100.

| Rule | Defaults in the code, and what the docstring says it does | TensorQuill | Note |
| --- | --- | --- | --- |
| `LowGPUUtilization` | `threshold_p95=70`, `threshold_p5=10`, `window=500`, `patience=1000`. For each GPU: true when the 95th quantile of utilisation is below 70, or when it is above 70 and the 5th quantile is below 10 (fluctuation) | **Lacks it** | TensorQuill reads no GPU utilisation anywhere in `sdk/src` or `server/src`. `gpu_throttled` is a different reading. Bucket (b): frozen, and nothing asks for it |
| `CPUBottleneck` | `threshold=50`, `gpu_threshold=10` (the GPU counts as under-used), `cpu_threshold=90` (the CPU counts as high), `patience=1000`. True when bottlenecks occur more than 50% of the time | Has it differently: `input_starvation` | TensorQuill times the wait for the next batch itself: its median share of step time over 20 steps is at least 0.3, or has risen by 0.2 |
| `IOBottleneck` | `threshold=50`, `gpu_threshold=10`, `io_threshold=50` (I/O wait counts as high), `patience=1000`. Same test with I/O wait in place of the CPU. The docstring gives the GPU default as 70; the code says 10 | Has it differently: `input_starvation` | As above |
| `GPUMemoryIncrease` | `increase=5`, `window=10`, `patience=1000`. Compares a moving average of GPU memory use with the previous one | **Has it differently:** `memory_growth` | Allocated memory up at 6 readings in a row, by at least 5% of the first and 64 MiB. The two fives are different quantities: there a rise in a moving average, here a fraction of the first reading. TensorQuill asks for a sustained trend and ignores a one-off jump |
| `LoadBalancing` | `threshold=0.5`, `patience=1000`. "Difference between 2 histograms" of utilisation per GPU | Has it differently: `memory_imbalance` | A rank whose peak allocated memory is 1.25 times the ranks' median and 1 GiB above it. Memory, not utilisation |
| `MaxInitializationTime` | `threshold=20`, in minutes, for the first step to appear | Has it differently | The watchdog's `starting` state: `--start-seconds`, 1,800 by default, before a rank that has not stepped is called `silent` |
| `StepOutlier` | `stddev=3`, `n_outliers=10`, `mode=None` (train and eval steps). True when a step's duration "is larger than stddev * standard deviation"; 10 outliers are ignored first | **Lacks it** | No rule flags a slow step. Step time is recorded when `data_wait=True`, and only `input_starvation` reads it, as a divisor; every record also carries the step's end time. Bucket (b): frozen, and nothing asks for it |
| `BatchSize` | `cpu_threshold_p95=70`, `gpu_threshold_p95=70`, `gpu_memory_threshold_p95=70`, `window=500`, `patience=1000`. The docstring gives patience as 100; the code says 1000 | Lacks it | A sizing hint. Bucket (c) |
| `Dataloader` | `min_threshold=70`, `max_threshold=200`: dataloader processes as a share of CPU cores. Defined, and in the profiler report's list, but not exported by the package | Lacks it | A sizing hint. Bucket (c) |
| `OverallSystemUsage` | No threshold. Its docstring says it cannot trigger | Lacks it | Nothing to map |
| `ProfilerReport` | Runs the ten rules above and writes a report. `opt_out_telemetry=False` | Lacks it | A report. Bucket (c) |

**Counting up.** Of the 21 debugger rules TensorQuill covers 7: six with an incident kind and one, `SimilarAcrossRuns`, with `compare_runs`. It lacks 4 that concern model internals, one of them (the weight update ratio) on purpose. It leaves the other 10 out as scalars, data content, XGBoost or reports. Of the 11 profiler rules it covers 5, all differently, lacks 2 that would be new kinds, and leaves 4 out. Every verdict on a debugger rule is by name, except `VanishingGradient`, which has the one example in `smdebug`'s docs.

TensorQuill kinds with no counterpart in either clone: the two MoE kinds, the three attention kinds, `replica_drift`, `duplicate_batches`, `epoch_order_repeated`, `data_stalled`, `repeated_steps`, the three resume kinds, `loss_scale_trouble`, `grad_spike` as a rule against a parameter's own history, `collective_hang`, `collective_desync`, `hardware_errors`, `allocator_retries`, `memory_pressure`, `gpu_throttled` and `traincheck_violation`. That is 22 of the 32. Neither clone has a rule that compares workers.

### Storage and the reading API, in brief

- **Layout** (from `smdebug/core/locations.py` and `core/access_layer/utils.py`): `events/<12-digit step>/<step>_<worker>.tfevents` holds the tensors; `index/<prefix>/<step>_<worker>.json` gives each tensor's name, offset and length, so a reader fetches a byte range, from S3 too; `collections/<prefix>/<worker>_collections.json` says what each worker saved; `training_job_end.ts` marks the end. The same layout serves local disk and S3.
- **Reading** (`smdebug/trials/trial.py`, `docs/analysis.md`): `create_trial(path)` gives a trial; `trial.tensor_names()`, `trial.steps(mode=)`, `trial.tensor(name).value(step, worker=)`, `.reduction_value(...)`, `trial.workers()`, `trial.wait_for_steps(...)` and `trial.has_passed_step(...)`. A trial refreshes itself as new steps arrive.
- **For TensorQuill:** per-step folders of protobuf event files suit raw tensors. TensorQuill's one JSON line per step, and Parquet parts on the server, suit small fixed statistics better. The six MCP tools are the nearer counterpart of the trial API.

### Design ideas worth following

| Idea in `smdebug` | Where | TensorQuill today | Worth doing |
| --- | --- | --- | --- |
| **Modes.** Every saved step carries a mode (train, eval, predict, global), each mode has its own save schedule, and a trial maps a mode's step to the global step | `core/modes.py`, `core/save_config.py`, `trials/trial.py` | A phase exists for the heartbeat only. Evaluation forwards flow into the next training step's record | **Yes: proposal 2** |
| **An end-of-run marker**, so a reader can tell "finished" from "stopped" | `training_job_end.ts` in `core/access_layer/utils.py`; `NoMoreData` in `exceptions.py` | The server knows how a stream ended. A local file has no marker; the README says a hung job and a finished one look the same | Yes: proposal 6 |
| **Rules apart from capture.** A rule is a small class with `invoke_at_step(step)`, run in another process over saved data, and it may read other runs (`other_trials`) | `rules/rule.py`, `rules/rule_invoker.py` | Capture, records and detectors are already apart, and the server judges out of process. There is no supported way for a user to add a check | Later. A user-written check over `load_records` would be the cheapest way to learn of "a failure none of the 32 catch". Not a new kind, but each one invites one |
| **A rule declares the tensors it needs** before it runs | `rules/req_tensors.py` | `kind_checks` with `needs`, and it goes further | Already has it |
| **Named states for missing data:** never saved, not for this step, not yet, training ended | `docs/analysis.md` "Exceptions" | `checked`, `not_configured`, `unavailable`, `enough_history`, and a rank that is `unknown` | Already has it, and exposes it to the reader. `smdebug`'s rule runner hides it (below) |
| **Collections with a save schedule each:** an interval, a start step, an end step, or a list of steps | `core/collection.py`, `core/save_config.py` | `capture=`, `weights=N`, `probe_every`, `replica_every`. No start or end window | Only for the parts outside the compiled graph (weights, probe, replica). The block statistics cannot be windowed without a second compiled variant, as the README explains |
| **A step is complete when every worker has written it,** with a window (1,000 steps) so one dead worker cannot block the reader | `trials/trial.py` (`last_complete_step`, `INCOMPLETE_STEP_WAIT_WINDOW`) | The all-ranks view marks a rank without a current report as unknown | Already close. The explicit window is a clean rule if `--rank all` ever has to wait |
| **Reductions chosen per collection** (min, max, mean, std, norms, absolute variants) instead of raw tensors | `core/reduction_config.py` | Fixed statistics per block | No; fixed statistics are what keep the compiled graph stable |
| **A rule's configuration names the data it needs,** so choosing a rule turns on its capture | rules-config `ruleConfigs.json` (`CollectionConfigurations`) | The other way round: the capture is chosen, and `kind_checks` says which kinds that left `not_configured` and what each `needs` | Low priority. A `kinds=` shorthand at `init` that turns on what the named kinds need would be a small convenience. Not proposed |
| **Wait before judging.** Six of the profiler rules collect `patience` datapoints first, 1,000 by default, and `StepOutlier` ignores its first 10 outliers | rules-config `profiler_rules/rules.py` | `records_needed` and `enough_history` for each kind | Already has it, and tells the reader, which the rule configuration does not |
| **Thresholds checked when the rule is built:** a percentile must lie in 0 to 100, a count must be positive, and an unknown rule or parameter is refused with the accepted names | rules-config `profiler_rules/rules.py`, `utils.py` | `--thresholds` refuses unknown keys, wrong types and negative values before the server listens | Already has it |

### What TensorQuill already does better or differently

- **Cost and `torch.compile`.** `smdebug`'s PyTorch hook registers a forward hook on every submodule and a hook on every parameter's gradient, and moves each saved tensor to the CPU as a NumPy array (`pytorch/hook.py`, `tensor_value.to(cpu).data.numpy()`). The docs list PyTorch 1.2 to 1.6. Nothing in `smdebug/pytorch`, `smdebug/core` or the docs mentions `torch.compile` or TorchDynamo. TensorQuill keeps statistics on the device, hooks the repeated blocks, and checks graph breaks and recompiles in CI.
- **Overhead claims.** I found none in the README or the docs I read. The only performance statement is that `save_all` "can reduce the performance of the training job significantly" (`docs/api.md`). There is no number and no method. TensorQuill publishes its protocol and its misses (the A100 runs over 1%).
- **Every step, cheaply.** `smdebug` saves every 500 steps by default (`save_config.py`), and each built-in debugger rule that names a collection asks for it at that same interval, bar the XGBoost report at every 5 steps (`ruleConfigs.json`). A one-step spike falls between saves. TensorQuill records every optimizer step.
- **Choosing what to watch by structure, not by name.** By default, four rule configurations pick their tensors by a name pattern: `.*relu_input`, `.*relu_output`, `.*embedding0_input_0` and `.*hybridsequential0_input_0`. A model whose tensors are named otherwise gives those rules nothing to read unless the caller passes other collections. TensorQuill picks the repeated blocks from the model's structure.
- **What a step is.** `smdebug` counts a step at each forward call of the registered module (`forward_pre_hook`). Gradient accumulation and evaluation make more "steps". TensorQuill counts optimizer steps.
- **Several GPUs.** `include_workers` defaults to `"one"`: only one worker's tensors are saved (`docs/api.md`, `core/hook.py`). `docs/distributed_training.md` is three lines ending in a TODO. I found no rule that compares workers. TensorQuill's replica, duplicate-batch, stall and all-ranks checks have no counterpart.
- **What was checked.** `invoke_rule` catches "tensor unavailable" and "step unavailable" and logs them at debug level (`rules/rule_invoker.py`). A rule over data that was never saved reports nothing, and that reads as "no problem". TensorQuill's `kind_checks` exists to prevent exactly that.
- **All incidents, not the first.** On SageMaker "the rule evaluation ends when the rule evaluation condition is met" (`docs/analysis.md`). TensorQuill lists every incident and never closes one.
- **Where it runs.** Built-in rules need SageMaker, results go to CloudWatch, messages to SNS. TensorQuill writes a file any agent can read.

### What not to follow

| Do not follow | Why |
| --- | --- |
| Saving raw tensors per step to disk or S3 | Each save is a copy to the CPU and a sync. It cannot stay near 1% and it breaks the compiled graph |
| Hooks on every leaf module and every parameter | TensorQuill measured leaf hooks at about 4.5 times the cost of block hooks; TrainCheck reports that this style of instrumentation interferes with `torch.compile` (p. 16) |
| Counting a step per forward call | Wrong under gradient accumulation and evaluation |
| Saving one worker by default | It hides the failures TensorQuill exists for |
| A rule that stops the job | `StopTrainingAction` calls SageMaker's stop API (`rules/action/stop_training_action.py`). The plan keeps TensorQuill explain-only until "act, with approval" |
| Swallowing missing data in the rule runner | See "what was checked" above |
| A network call on the tool's own behalf | `smdebug/core/utils.py` prepares a request to a profiler telemetry address, and the profiler report's flag is `opt_out_telemetry=False` by default (rules-config `profiler_rules/rules.py`): on unless turned off. TensorQuill's "no telemetry" is a selling point |
| Picking tensors by a layer-name pattern | See "by structure, not by name" above |
| Defaults whose documentation disagrees with the code | `BatchSize`'s patience (docstring 100, code 1000) and `IOBottleneck`'s GPU threshold (docstring 70, code 10). TensorQuill prints the thresholds that were applied in every report, which is the cure |
| Building a statement from configuration text and running it | `validate_action_str` formats the action into a Python statement and runs it with `exec` (rules-config `actions/utils.py`). TensorQuill's trigger hands incident fields to the user's command in the environment and never formats them into the command (`trigger.py:505–524`) |
| Rule arguments that must be strings | Done so rules can be made from JSON in AWS's API (`docs/analysis.md`). TensorQuill's typed, refused-if-wrong thresholds are better |
| Supporting four frameworks from one hook base class | Much of the code is TensorFlow, MXNet and XGBoost glue. It last changed in 2023 |

### How its tests exercise rules

Thinly. One test defines a dummy `Rule` subclass and runs it over generated data to check that a trial is not refreshed mid-rule (`tests/analysis/rules/test_rule_no_refresh.py`). `tests/rules/action/` tests the message action. The trial tests are fuller and include several-worker cases for whether a step has passed (`tests/analysis/trials/test_has_passed_step_scenarios.py`). There are no tests of built-in rules, because the rules are not here.

The rules-config package's tests cover three things: which rules belong to which framework, the profiler report's parameter checks and its telemetry flag, and the action objects (`tests/core/`). They cannot test rule logic, which is not there either. Its build specification sets `run_pytest_core` to `"disable"`, and the test script runs them only when it is `enable` (`config/buildspec_core.yml`, `config/tests.sh`), so as configured the build does not run them. I did not run them.

TensorQuill's golden files, parity tests and mutation checks are well ahead of both.

---

## 5. Benchmarks and public failure sets

*Added Nov 18, 2026: section 10 lists the public material the eleven added papers point to.*

Nothing was downloaded. "Local" means a clone already sits beside this repository.

| What | Where | Use for TensorQuill | Notes |
| --- | --- | --- | --- |
| SCOUT's fault injector: JSON campaigns of faults on weights, biases, optimizer state, tensors, processes and collectives, for PyTorch, TorchTitan, Megatron and DeepSpeed; its README describes a campaign of 21 failure types on one eight-GPU host | **Local:** `../lm-resiliency/docs/fault_injection.md`, `examples/torchrun/`; upstream `github.com/LMResiliency/lm-resiliency`; BSD 3-Clause | Result 1 (two-GPU seeded cases) and result 4 (real multi-rank captures for the agent benchmark). It gives planted faults with a known rank and step | I read the README and the first tables of the injection guide, not the code. It needs GPUs |
| TrainCheck's 20 reproduced silent bugs: reproduction scripts, pre-collected traces and curated invariants | `github.com/OrderLab/TrainCheck-Evaluation-Workloads` (named in the local `../TrainCheck/docs/ae-eval-s5.1-silent-issue-detection.md`); not cloned | Run TensorQuill on the reproduction scripts and publish how many it catches, beside TrainCheck's 18. This is the honest version of the claim in W4 | The local doc marks 12 of the 14 listed bugs as supported for artifact evaluation. Many are tiny models on old library versions |
| TrainCheck's overhead benchmarks | `github.com/OrderLab/TrainCheck-Benchmarks` (named in `../TrainCheck/docs/benchmarks.md`) | A fourth arm for result 3, the head-to-head under `torch.compile` | TrainCheck says it does not support compiled code (p. 16); that is itself a result |
| Loss-spike reproductions: small models at high learning rates that diverge late | `github.com/tingtang2/quantization-loss-spikes` (Chen et al., p. 1) | A from-scratch failure that no injected fault produces: test `attention_logit_growth`, `grad_spike`, and proposal 8's numbers. Models start at 28M parameters, so a CPU or one small GPU may do. **Corrected in section 9 (D4):** the paper's models are JAX and Flax, and every run uses QK-norm, so this is not a direct test for a PyTorch hook library or for `attention_logit_growth` | Not looked at |
| Massive-activation analysis code | `github.com/locuslab/massive-activations` (p. 1) | Check which token positions and dimensions carry the outliers in Granite and SmolLM2, for proposal 7 | Not looked at |
| Rotational-equilibrium code | `github.com/epfml/REQ` (p. 1) | Reference update sizes, if the update-size candidate is built | Not looked at |
| FOG (FP8) training code | `github.com/anonymous4375934/FOG`, as printed on p. 7 | A failing FP8 run with kurtosis traces | An anonymous address from the review period; it may not resolve |
| Straggler analysis | `github.com/ByteDance-Seed/StragglerAnalysis` (Lin et al., p. 1) | Possibly traces with per-operation timings per rank | The paper calls it a project page. I did not confirm that the trace is public |
| SPAM | `github.com/TianjinYellow/SPAM-Optimizer` (p. 1) | Spike-prone small LLaMA runs | Not looked at |
| Instruction-level fault injector for GPU matrix multiplies (NVBit) | `github.com/aaltenbernd/llm-sdc-training` (Altenbernd et al., p. 2). Added in section 9 | Seeded faults for the update-size kind on a real GPU | Not looked at. Needs a GPU |

Not public as far as the pages I read say: Minder's 150 fault instances, ByteRobust's incident records, the faulty nodes of Ma et al., LLM-PRISM's injection engine, TrainSDC's code.

---

## 6. What I did not read or could not verify

*Added Nov 18, 2026: for the eleven added papers and posts, what was and was not read is in the last table of section 10.*

I read the text of each PDF as extracted by `pdftotext`, which was already installed. For most papers I read the first pages in full and then pulled the sentences on method, results and limits by keyword from the remaining main pages. Figures were not viewed, so every number above comes from text or a table. One table (Rotational Equilibrium, Table 1) was checked on the rendered PDF page. **Section 9 replaces this account:** the 22 papers were read page by page later, and its last table says what was seen.

| Paper | Read | Not read or not verified |
| --- | --- | --- |
| 2606.28116 | pp. 1–9 in full, Appendix E (p. 19) | References; the proofs in Appendices A–D (pp. 14–18). The second file is byte-identical. Earlier versions and their title not seen |
| TrainCheck | pp. 1–7 in full; pp. 10–16 by key sentences, Table 3 and the limitations | pp. 8–9 skimmed. Table 4's per-bug marks did not survive extraction, so the per-bug outcome is from the text. My count of bugs TensorQuill could catch is a reading, not a test |
| SCOUT | pp. 1–5 in full; pp. 6–13 by key sentences | pp. 14–16. Table 3's layout. The MoE replay catalogue. The `lm-resiliency` code: README and part of one doc only |
| Ma et al. | pp. 1–2 in full; pp. 3–9 by key sentences | Appendices (pp. 10–23), the checksum experiment's details among them |
| Altenbernd et al. | p. 1, pp. 7–9 in full; pp. 2–6 by key sentences | The detection rate at the chosen setting is only in a figure |
| LLM-PRISM | p. 1; pp. 2–11 by key sentences | The RTL fault model (pp. 3–6) only skimmed; figures |
| TrainSDC | p. 1; pp. 2–11 by key sentences | p. 12. Whether code is released |
| Massive Activations | pp. 1–3 in full; pp. 4–12 by key sentences | Appendix (pp. 17–32). Nothing in what I read says when in training the outliers appear |
| FP8 paper | pp. 1–2 in full; pp. 3–10 by key sentences; limitations; Table 2 | Appendices and the checklist (pp. 11–26) |
| Rotational Equilibrium | pp. 1–2 in full; p. 4 on the page; pp. 3–9 by key sentences | Appendices (pp. 10–37). The example value in the candidates table is the formula only, not a measurement |
| Lin et al. (stragglers) | p. 1; pp. 2–12 by key sentences | The monitoring-system section and after (pp. 13–19) |
| Minder | p. 1; pp. 2–9 by key sentences | pp. 10–17 |
| ByteRobust | p. 1; pp. 2–14 by key sentences; two tables from the layout text | I counted 35 authors and name six; evaluation figures |
| MegaScale | Key sentences of pp. 1–13 | The efficiency sections were only skimmed; pp. 14–16. The venue is not on the pages I read |
| Chen et al. (quantization) | pp. 1–2 in full; pp. 3–10 by key sentences | Appendix (pp. 11–27) |
| SPAM | p. 1; pp. 2–7 by key sentences | Experiments (pp. 8–9) and appendix; the optimizer's default settings |
| EnergyCal | p. 1; pp. 2–9 by key sentences | Appendix (pp. 10–14) |
| R3 | pp. 1–2 in full; pp. 3–13 by key sentences | Appendix (pp. 15–33) |
| Sha (visual MoE) | p. 1 in full; pp. 2–15 by key sentences | pp. 16–23. No independent check of any claim; one author, one setup |
| Li et al. (agentic MoE) | Key sentences of pp. 1–10 (the main body) | The appendix (pp. 11–47); figures. The results table lost its layout in extraction, so only the figures quoted in the text are used |
| FlashOptim | Key sentences of pp. 1–9 | Appendix (pp. 10–15); the code. W10 is an inference from the paper's description (p. 5) |
| SProbe | p. 1; pp. 2–6 by key sentences | pp. 7–10 |

**`smdebug`.** Read: `README.md`; `docs/analysis.md` (the rules and exceptions sections, and the headings of the rest); `docs/api.md` (hook, collections, modes); `docs/pytorch.md`; `docs/sagemaker.md` (the rules sections); `docs/distributed_training.md`; `smdebug/rules/` (`rule.py` and `rule_invoker.py` in full, the actions by outline); `smdebug/core/` (`modes.py`, and by search `save_config.py`, `config_constants.py`, `reduction_config.py`, `locations.py`, `hook.py`); `smdebug/pytorch/hook.py` by search; the method list and step-completeness logic of `smdebug/trials/trial.py`; the test folder listing and one rule test's outline. Not read: the TensorFlow, MXNet and XGBoost hooks; the event-file writers and the index reader; the S3 access layer; the profiler's parsers and analysis code (file names only); `docs/tensorflow.md`, `mxnet.md`, `xgboost.md`, `env_var.md`; the examples; most tests. Nothing was run. The storage layout in this review is from path-building code, not from files written by a run.

**The rules-config package.** Read in full: every Python file, both JSON files, the README, `setup.py`, the tests and the build configuration. Nothing was run or installed. Still unread, because it is in neither clone and I fetched nothing: the code of every built-in rule, and so what each debugger rule computes and its thresholds. The has-or-lacks verdicts for debugger rules therefore rest on the rules' names. The rule list is the one at a May 2021 commit; whether AWS has added or renamed rules since is not checked.

**TensorQuill.** Read: the focused plan in full; `sdk/README.md` lines 1–307 and the sections on thresholds, attention rules and "Things to know"; the failure list in full; `sdk/AGENT_INSTRUCTIONS.md` in full; `INCIDENT_CLASSES` and the constants in `evidence.py`; the parts of `capture.py` and `distributed.py` cited above. The v3 plan was searched, not read. No test in the repository was run. Proposal 2 was reproduced once, on CPU, with a toy model and router counts only. Finding W7 comes from reading the rule and the papers; one toy from-scratch run on CPU did not trigger it (2.62× against a 3× cutoff), which neither confirms nor settles it. Both scripts ran from a scratch folder with the repository's `.venv` and wrote nothing into the repository.

**Not in the folder.** Papers the plans cite that I did not see: Motif 3 (2608.09119), the Laguna report and the others in the failure list's sources. SDCHunter and OLMo 2 were not there on Nov 17; they were added on Nov 18 and are covered in section 7.

---

## 7. Second batch: twelve more papers, read on Nov 18, 2026

Twelve PDFs were added to `../researchpapers/` after sections 1 to 6 were written. I read them to check `Training failure research survey.md`, which cites them. This section records what they say and where that survey goes further than they do. The survey file itself was left unchanged, at the author's request. Sections 1 to 6 are as they were, except the last line of section 6.

**How they were read.** First pages in full, then keyword passes over the remaining main pages, with tables read from layout-preserving text. Figures were not viewed. The long appendices of OLMoE (63 pages) and OLMo 2 (58 pages) are unread.

**Corrected later.** The papers were then read page by page (section 8). Seven statements in this section did not hold or were too loose. Each is marked below with its number in section 8's corrections table (C1 to C7).

### The papers

| Paper | What it shows | Bearing on TensorQuill |
| --- | --- | --- |
| **SDCs in the Wild: Characterizing and Diagnosing SDC-defective GPUs in Production LLM Training.** Zheng, Wang, Zhang, Han and 15 others (Shanghai Jiao Tong University, ByteDance Seed). OSDI '26 | Measured on 23 defective GPUs from production. On real traces, loss and gradient-norm monitoring covered 25.0% of them, vendor hardware tools 30.0%, the paper's SDCHunter 100% (Table 5). Under injection: loss 16% after 127 steps, gradient norm 17% after 109, SDCHunter 91% after 3 steps at 3.9% overhead (Table 6). Loss-based methods named the faulty device in 0% of cases (Table 7). ECC counts, clocks, thermals and machine-check exceptions showed nothing (p. 7). Synthetic stress tests miss over 60% (p. 2) | Backs finding W1 with production numbers. Loss, gradient and non-finite checks are low-recall for corruption, and clean hardware telemetry does not rule it out |
| **Safeguarding LLM Training at Scale: Online SDC Detection and Insights from 35 Million GPU Hours** (AEGIS). Lei, Zheng, Li and 14 others (Tsinghua, ByteDance). OSDI '26 | Measured in production: 18 incidents on 13 GPUs in 3.5e7 GPU-hours; only 3 of the 18 showed as a visible failure (p. 11). Every incident reproduced on the same GPU (pp. 13–14). Vendor diagnostics caught 2 of 8 known-faulty machines, AEGIS 8 of 8 (Table 2). A checksum accumulated in bfloat16 needs about 10,000 times the perturbation that a float32 one does (p. 13) | An online detector exists at under 3% cost. Two of its three mechanisms are within reach of a hook library (below) |
| **OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training.** Zhou, Zhao, Zhu, Wang, Bai, Zhang, Wang, Lin, Huang (University of Michigan, ByteDance Seed). OSDI '26 | Compares a faulty run with a reference run and reports the first model-level operator whose output is not bit-for-bit equal. Measured: 20 production bugs diagnosed (p. 3); 21 machines with corruption that had passed vendor pre-flight checks (p. 13); online mode about 1.00 to 1.01 times, trusted mode 1.25 to 1.45 times, full mode about 1.8 to 1.95 times (p. 14) | It instruments compiled code (below). A model for `compare_runs`: name the first boundary that differs |
| **CCL-D: A High-Precision Diagnostic System for Slow and Hang Anomalies in Large-Scale Model Training.** Gu, Wang, Fu, Sun and others (University of Chinese Academy of Sciences, Ant Group). PPoPP '26 | Measured on a 4,000-GPU cluster over a year: hang diagnosis from 47 hours to 6 minutes, slow from 74 hours to 2 minutes (pp. 9–10); overhead 0.12% to 0.95% (p. 11). Of hangs, 11.8% were Not-Entered, 58.9% Inconsistent and 29.3% Hardware-Fault (p. 4). It works by extending the communication library | Rules for the slow-rank candidate and for `collective_hang`; published support for the 300-second silence default (below) |
| **Greyhound: Hunting Fail-Slows in Hybrid-Parallel Training at Scale.** Wu, Wang, Yu, Yang, Wu, Duan, Yang, Wang, Qu, Zhang (HKUST, Alibaba). USENIX ATC '25 | Measured: 16 of 27 large jobs hit a fail-slow, average completion time up 34.59% (p. 5). Change-point detection on iteration time, then a check that the mean moved by 10%: 100% accuracy on computation fail-slows and 99.1% on communication, with 2.3% missed (p. 11). Overhead 0.39% on average, 1.1% at most. Change-point detection alone gave 18% to 34% false positives. A communication group is suspect when its time exceeds the median of comparable groups by over 10% (p. 7) | A trigger for the slow-rank candidate. The 100% rests on 6 positive cases among 392 single-node jobs, and locating the slow part pauses the job (p. 8) |
| **Evolution of Aegis: Fault Diagnosis for AI Model Training Service in Production.** Dong, Qian, Zhang and others (Alibaba Cloud). NSDI '25 | Measured: 100 to 230 critical failures a week (p. 3); 73% of failed tasks fail in the first 10 minutes (p. 10); 71% of "distributed" failures turn out not to be the network (p. 7). An outlier is a host above the mean plus two standard deviations for ten minutes (p. 10) | The same timing inversion as CCL-D: a group whose communication time is under 0.8 of the typical is the slow computer (p. 10) |
| **Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.** Wang, Gao, Zhao, Sun, Dai (DeepSeek-AI, Peking University). Preprint, 2024 | Defines MaxVio as the largest expert load over the expected load, minus one. Measured on 64 experts with 6 active: 0.04 with bias-based balancing against 0.72 (1B model) and 0.52 (3B) with an auxiliary loss, counted over a validation set (Table 2, p. 6). The paper gives the 3B model's training as 200B tokens in three places and 300B in its conclusion | The one MoE balance measure with published good and poor values. It must be reported with its scope: per batch or over a whole set |
| **OLMoE: Open Mixture-of-Experts Language Models.** Muennighoff, Soldaini, Groeneveld and others (Allen Institute for AI and others). ICLR 2025 per the survey; arXiv v2, Mar 2025 | Without the load-balancing loss, "initially all tokens in the first layer are assigned to the 6th expert" and the other experts stay unused (p. 13). After 1% of pretraining, up to about 60% of top-8 routing already matches the final checkpoint (p. 19) | A from-scratch MoE collapse that needs no injected fault: the proof run for `moe_imbalance` on nanoMoE |
| **2 OLMo 2 Furious.** OLMo Team (Allen Institute for AI). Technical report, v3 Oct 2025 | Defines a spike score: the share of values at least seven standard deviations from a rolling average of the last 1,000 (p. 14). A new initialisation took the gradient-norm score from 0.40 to 0.03. Gradient spikes often came before loss spikes (p. 10) | A second, published definition to report `grad_spike` against, beside EnergyCal's (proposal 7) |
| **TTrace: Lightweight Error Checking and Diagnosis for Distributed Training.** Jiang, Zhu, Zhang and others (New York University, Amazon Web Services). Preprint, v2 Jan 2026 | Compares a distributed run's tensors with a trusted single-device reference, with tolerances derived from an analysis of rounding error. Found 11 known and 3 new silent bugs in Megatron-LM; one in 54 seconds against 6 hours 40 minutes from loss curves | A second bug set for proposal 5. Its tolerance approach is disputed by OpGuard (below) |
| **Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes.** Liu, Cao, Li, Zhou (Tsinghua, University of Tokyo). Preprint, v5 Nov 2026 | In float32 the gradient of the correct class rounds to exactly zero once its logit leads the others by `23·ln 2`, about 16 (p. 3). Classifier updates went from about 1e-5 to two modes at ±4e-4 at the spike (p. 7). Casting only the loss computation to float64 removes the spikes (p. 2) | A CPU-scale reproduction for proposals 8 and 10, and a second mechanism beside Chen et al. for output-logit trouble (**for language models the paper found the reverse, C7**) |
| **Understanding Silent Data Corruption in LLM Training** (the ACL 2025 version of row 4 in section 3). Ma, Pei, Lausen, Karypis | Same paper, published in the Proceedings of the 63rd Annual Meeting of the ACL, pages 20372–20394 | Confirms the venue. Its finding on checksums is that they missed corruption: they flagged errors on one node only |

None of the three ByteDance systems appears to be released; I found no code link in any of them.

### What changes in sections 1 and 2

- **W1 is stronger.** It rested on reasoning from four papers. Zheng et al. and Lei et al. now give production numbers for the same point: 3 of 18 incidents visible, and 25% coverage for loss and gradient norms.
- **W6 needs one more name.** OpGuard's wrappers "preserve the original execution mode (eager, torch.compile, CUDA Graphs)" (p. 8), and it reports at least 95% kernel coverage on a compiled pretraining pipeline (p. 14). Its overhead figures are not split by compilation, so TensorQuill can still be first to publish an overhead number under `torch.compile`. It cannot say that no other tool instruments compiled code.
- **W9 has three more supporters.** All three ByteDance papers compare replicas bit for bit under enforced determinism. OpGuard measured what a tolerance costs: without determinism controls and with a tolerance of `1e-3`, it placed the fault exactly in 7 of 20 cases (p. 13). It also reports that TTrace flagged 11 of its 20 cases and misses corruption smaller than its tolerance (p. 12).
- **Proposal 9 has two concrete designs.** Neither is SDCHunter as built, which is diagnosis by replay after an alert: half the data-parallel groups re-run the identical batch as the other half (pp. 3, 10–12).
  - *A shared probe batch.* TensorQuill's probe pass already runs one fixed batch; if every rank ran the same one, their block statistics could be compared across ranks. This is my reading, not a published method. **Wrong: Ma et al. propose a close relative (C3).**
  - *Forward against recomputation.* AEGIS fingerprints an operator's output in the forward pass and again when the backward pass recomputes it, and treats any difference as corruption (p. 8). TensorQuill's hooks already see recomputation under activation checkpointing, and skip it.
  - AEGIS's third mechanism is out of reach: its checksum comes from the float32 accumulator inside the matmul kernel (p. 7).
  - Its 0.86% overhead is with sampling tuned to a budget. Unsampled, it averages 1.61% or 2.35% and peaks at 2.6% (p. 12).
- **The slow-rank candidate is half reachable** (**"half" depends on the cluster, C6**)**.** Flight Recorder's per-collective durations can support the compute-slow rule, where the rank that spends least time in the collective is the slow one (CCL-D p. 7, Aegis p. 10). The communication-slow rule and the hardware-hang rule need send and receive counters inside the communication library, which CCL-D and Aegis both customise.
- **`collective_hang` should not assume one rule.** In CCL-D the rank whose counter did not advance is the root cause only for Not-Entered hangs, 11.8% of the total. For Inconsistent hangs "the non-hang ranks are the root cause", and for Hardware-Fault hangs it is the rank with the fewest send and receive counts (p. 7).
- **The 300-second silence default now has a source** (**too strong: the source is one cluster's checkpoint time, C5**)**.** CCL-D sets its hang bound at five minutes because "97% of cases exceeding this limit cannot recover" (p. 8). Section 2 could only say the default was in the same range as other systems.
- **Section 5 gains two bug sets,** neither with a repository link I could find (**wrong for TTrace: its table gives issue numbers and fixes, C4**): TTrace's 11 to 14 Megatron-LM bugs, and OpGuard's 10 open-source issues from Megatron-LM, DeepSpeed and others (p. 12).

### Where the survey goes further than these papers

| Survey statement | What the paper says |
| --- | --- |
| "The rank with the lowest collective sequence number is the hang's root cause" (summary and finding 5) | True for 11.8% of CCL-D's hangs. See `collective_hang` above |
| K2 "should follow SDCHunter" | SDCHunter is replay after an alert, on a deterministic job. AEGIS is the online detector |
| "Any in-process checksum must accumulate in float32 or wider" (finding 2) | Correct as a constraint, but AEGIS meets it inside the kernel, where a hook cannot go |
| Ma et al. found "false positives on healthy nodes in bf16" (finding 2) | Ma et al. report misses. The false-positive description is AEGIS's, when citing them (p. 4). **Wrong: the survey was right (C1)** |
| K2 "must be tolerance-based" | The three ByteDance papers are bitwise, and OpGuard measured the loss from a tolerance |
| "No surveyed tool reports instrumentation overhead under torch.compile" | Holds narrowly. OpGuard runs under it and reports coverage there |
| Greyhound: "100% accuracy on computation fail-slows" | Six positive cases |
| Aegis's 0.8 and 1.5 cut-offs "should be verified in the paper" | Verified, in the direction the survey gives |
| OLMoE: all tokens go to one expert, "then a second" | The first half is in the text. The second is only in a figure I did not view. **Wrong: both are in the text (C2)** |

Every number the survey quotes from these twelve papers matched the PDF. With C1 and C2 the table above has seven rows that stand, not nine.

### Not read or not verified

- Figures in all twelve.
- OLMoE beyond pages 1–22 and OLMo 2 beyond pages 1–16.
- The appendix of Liu et al., which holds the model size and seed the survey's seeded recipe quotes.
- OLMo 2's z-loss weight, which the survey notes is given two ways; the equation did not survive text extraction.
- Whether any of these systems has released code beyond what the PDFs say.

---

## 8. Full read of the twelve papers (Nov 18, 2026, later the same day)

Section 7 was written to check the survey's numbers. I read first pages and then searched the rest by keyword. That finds what one already knows to look for, so it was not a search for new ideas. The author asked whether the papers had been read in full. They had not. This section is the result of reading them page by page, and of checking each new point against today's code.

Two things came out of it that matter more than the rest:

- **Seven statements in section 7 did not hold or were too loose.** Two of them were corrections of the survey that were themselves wrong. They are listed first, because the plan's commit `93e7d77` copied four of the seven.
- **TensorQuill counts router load from the scores it hooks, not from the routing the model does.** Where a model adds something to the scores before choosing experts, the counts are wrong. Reproduced on CPU (finding F1).

### Corrections to section 7

| # | Section 7 said | What the paper says | Copied into `93e7d77`? |
| --- | --- | --- | --- |
| C1 | The survey was wrong that Ma et al. found "false positives on healthy nodes in bf16"; that was AEGIS's description | The survey was right. Ma et al., Appendix D.2: "when running the check as-is at bf16, we observed several false positives on healthy nodes" (p. 17). Their main experiment ran at float32, where the checksum missed: it flagged often on Node 14 and little or not at all on the other bad nodes (p. 6) | Yes, in the count: "nine places where it goes further" |
| C2 | OLMoE's "then a second" expert is only in a figure | It is in the text: "Eventually, the model also starts assigning some tokens to the 1st expert", and the rest "remain largely flat" (p. 13). That figure's model has 8 experts | Yes, the same count. With C1 the nine become **seven** |
| C3 | A shared batch on two ranks is "my reading, not a published method" | Ma et al. propose it: a "shadow data-parallel replica" takes one rank's inputs, and the two gradients are compared at the all-reduce; they liken it to Gemini's corruption scanners on hot standbys (p. 9). Proposed, not built or measured, and it compares gradients, not block statistics. (Section 9, D9: they did build a same-input comparison as a measuring tool.) The survey already listed it | Yes, in the K2 row and the Decisions entry |
| C4 | TTrace's and OpGuard's bug sets have no repository link | TTrace's Table 1 (p. 9) gives every bug its issue number and its fixing commit or pull request, and all were reproduced from "the corresponding history commits" on a model that fits one device (p. 10). See F12 | No |
| C5 | "The 300-second silence default now has a source" | The source is tied to one cluster: CCL-D's five minutes "reflects checkpoint and normal barrier durations" there (p. 8). Greyhound cites a 100B-parameter checkpoint that takes "nearly 100 minutes" to write (p. 8). The default is safe only because TensorQuill pauses the check during a marked phase (`sdk/README.md`, "Evaluation and checkpoints are not hangs") | Yes |
| C6 | The slow-rank candidate is "half reachable" | The share depends on the cluster. CCL-D: of slow cases, 81.8% compute-slow, 11.1% communication-slow, 7.1% mixed (p. 4). Greyhound, on a shared network: 42 of 107 four-node jobs slowed by congestion against 6 of 392 single-node jobs slowed by compute; at 512 to 1,024 GPUs, 13 of 16 slowed jobs were network only (pp. 3–5). So the reachable half is most cases on one cluster and few on another | Yes, in the K6 row |
| C7 | Liu et al. is "a second mechanism for output-logit trouble" | For language models the paper found the reverse. On a 110M nanoGPT with no weight decay, the mean logit after 100,000 steps was 183 with the loss in float32 and 498 in float64; it blames token-frequency imbalance, not precision (p. 9). With mini-batches a visible spike is unlikely "except in highly structured and nearly noiseless tasks such as modular arithmetic"; on CIFAR-10 with batch 256 "no Slingshot spike is observed" (p. 9). The survey already had the two figures | No |

### New findings, each checked against the code

"TensorQuill today" was read on Nov 18, 2026 at commit `93e7d77`. Bucket (a) is a wording, documentation or small code change; (b) is a new detector kind, for the plan's K list; (c) is outside the product.

| # | Finding | Source | TensorQuill today | Change | Bucket, size |
| --- | --- | --- | --- | --- | --- |
| F1 | **Router counts can be the wrong routing.** In loss-free balancing the experts are chosen on score plus a per-expert bias, while the layer's output uses the unbiased score. The bias is updated every step from the last batch's load. The gate there is a sigmoid, not a softmax | Wang et al., p. 4 (Algorithm 1), p. 5 | For a router that returns floating-point scores, the hook takes the top-k of the hooked tensor (`capture.py`, `_router_hook`). On a toy with a balancing bias the real routing was 1,703 / 1,406 / 1,003 / 1,008 tokens; TensorQuill recorded 5,120 / 0 / 0 / 0 and raised `moe_imbalance` with three dead experts. The README does not mention this case, and neither do the plan, the failure list or the router notes. With an `indices=` function that applies the bias, the same toy recorded 1,703 / 1,406 / 1,003 / 1,008, so the fix that exists works. The toy shows the mechanism only; no real model was tested | Say in the README and the agent instructions that the default reading is right only when the model picks the plain top-k of the hooked tensor, and that otherwise `indices=` (or hooking the module that returns the chosen indices) is needed. The same applies to any change made after the hooked module, such as added noise or a capacity limit (my reading) | (a), small |
| F2 | **The drift tolerance hides most single-bit flips; the fingerprints themselves do not.** OpGuard measured the cost of a sum in place of an exact fingerprint: 14 of 20 faults placed as exactly as its full setup places them, because a sum "misses subtle one-token or one-bit corruptions". That was on activations between two runs, not on weights between replicas. Its exact fingerprint costs about as much as `torch.sum` | OpGuard p. 13, Appendix A.4 | Tested on CPU with `parameter_fingerprints` and `spread_from_fingerprints` on one 1,000 by 1,000 float32 matrix; the detector itself was not run, the spread was set against the tolerance by hand. A flip of the lowest mantissa bit of one weight moved the sum and projection by 7e-13 of the norm, and bit 12 by 3e-9: both far under the `1e-6` tolerance. A sign flip gave 1.3e-5, over it. The norm column did not move at all for two of the five flips. An integer checksum of the raw bits differed in all five | Report "the replicas are not bit-for-bit equal" as its own fact, apart from the tolerance. An integer sum of the bits is exact and does not depend on summation order. This is finding W9 and proposal 3 with a measurement | (a), small to medium |
| F3 | **A healthy MoE routes a narrow domain unevenly.** For arXiv text one first-layer expert is "nearly 100% specialized"; on general web text the same model is balanced. The first layer settles more slowly than the others and can swing. After 1% of training up to about 60% of top-8 routing already matches the final model, and the load distribution "does not change substantially" in fine-tuning | OLMoE pp. 18–22 | `moe_imbalance` and `starved_experts` read per-step load. `AGENT_INSTRUCTIONS.md` has no line on data domain | Tell the agent: before calling a collapse, check whether the batch or fine-tune is one domain, and whether only the first layer is affected. A large routing shift during fine-tuning is the unusual thing | (a), small |
| F4 | **A rerun tells hardware and races from data and code, if the rerun is deterministic.** Ten reruns from one checkpoint on a faulty GPU spiked at "completely different steps", or not at all. A race in a seven-year-old PyTorch kernel showed at step 3081 in one run and about 1535 in another. But OLMo 2 saw the same data spike under one data order and not another, so the rule needs the same seed and data order | Zheng et al. p. 5; OpGuard p. 11; OLMo 2 p. 12 | Not in the agent instructions | Add it, with the precondition | (a), small |
| F5 | **Slow growth of the gradient norm is a second warning sign, separate from spikes.** It went with more frequent spikes and ended in divergence. Filtering the data removed many spikes and "had no effect on the slow growth" | OLMo 2 pp. 10, 13, 15 | No kind sees it. `grad_spike` compares with a rolling median, which follows a slow rise | Put the level of the gradient norm against its early level, and the spike score (section 7), into the `grad_spike` evidence. A kind of its own would be bucket (b) | (a), small; or (b) |
| F6 | **Repeated text in a batch causes spikes.** Batches at spikes held long repeats of one short token pattern. Not every such batch spikes. Removing documents with 32 or more repeats of a pattern of 1 to 13 tokens cut the spikes | OLMo 2 pp. 12–13 | No data check looks at token content; `duplicate_batches` hashes sample indices | Tell the agent to look at the batch when `grad_spike` fires. A repeat check on token ids would be a new kind | (a) for the instruction; (b) for a check |
| F7 | **A seeded case for the update-size kind, with published numbers.** At a slingshot spike the last layer's Adam update went from about 1e-5 to two modes at ±4e-4, over 40 times larger (the paper's Figure 2a and a worked example with betas 0.9 and 0.95; it does not say which run the figure is from). On CIFAR-10, spikes came in all 20 runs of four architectures, most of them full batch | Liu et al. pp. 6–7; setup in Appendix D | `update_spike` is being built now (the plan's K1) | Use it as a test that needs no injected fault. The paper's modular-division setup is the cheap one: mod 97, a 2-layer decoder with LayerNorm off, Adam at 1e-3 with betas 0.9 and 0.999, no weight decay, batch 512, float32, 100,000 steps, about an hour on a GPU. The paper does not give the update figures for that setup, so they would have to be measured. Before the spike the gradients are near zero and Adam's step grows toward learning rate over epsilon; reading that as an early sign is my inference, not the paper's | (a), a test |
| F8 | **For the cross-rank corruption check (K2), four design facts.** (i) In production, the method that compares an output with its own recomputation found 12 of the 18 incidents; the checksums and their warnings found 6. The 12 are not split between recomputation inside the attention kernel, which a hook cannot reach, and recomputed activations, which it can. (ii) It fingerprints only the last operator of a recomputed chain. (iii) A mismatch between a forward output and its recomputation also comes from software: two of OpGuard's 20 bugs were recomputation bugs. (iv) It needs deterministic kernels and a job that recomputes | AEGIS pp. 6, 9, 11; OpGuard Table 3 | The hooks see recomputation and skip it (`capture.py`, `_recomputing()`) | Record these with the K2 design choice. A mismatch should be reported as "bug or hardware", never as hardware | (b), design input |
| F9 | **For the slow-rank candidate (K6), what the timing cannot show.** (i) Four of six compute slowdowns on single-node jobs were CPU contention from neighbours, with healthy GPUs. (ii) When one part is slow, the utilisation of all GPUs falls together, so utilisation cannot name the rank. (iii) Real compute-slow causes were uneven data, garbage collection and uneven compute. (iv) Stacks are "identical across ranks" for a slow rank, so they help with hangs only | Greyhound pp. 3–4, 6; CCL-D pp. 4, 9 | No host CPU or garbage-collection reading (`system.py` has power, temperature and throttle reasons). No rule names a rank from utilisation | Add process CPU time against wall time, and collector pauses, when K6 is built. Point (iv) limits proposal 6 to hangs. **Qualified in section 9 (D3):** ByteRobust uses repeated stack snapshots for slow jobs too, so this is contested | (a), small, with K6 |
| F10 | **Heavy instrumentation can make the fault stop showing.** Error rates on two faulty GPUs fell under instrumentation and new NaNs appeared. A checksum's overhead lowered utilisation and with it the corruption rate. A tracer that synchronises made 4 of 20 bugs unreproducible | Zheng et al. p. 9; Ma et al. p. 17; OpGuard p. 13 | Full capture is offered as the way to look closer | Say in the README and the agent instructions that a fault which vanishes under full capture is not thereby gone | (a), wording |
| F11 | **Fixed tolerances fail in both directions.** Every pair of absolute and relative tolerance tried with `torch.allclose` gave false alarms, misses or both on TTrace's bugs; a bug's error was about 100 times rounding error | TTrace p. 11 (Table 3) | `where_runs_part` already calls its answer "where to look first, not a cause" | Keep that wording. Do not describe it as finding the first faulty operator | none |
| F12 | **A reproducible public bug set.** Fourteen Megatron-LM and TransformerEngine bugs with issue numbers and fixes. Bug 9 (issue 663, "No parameter update") is what `stalled_weights` names. Bugs 6 and 12 (issues 599, 1446) are weights whose gradients were not synchronised across ranks, the family `replica_drift` belongs to, though across sequence-parallel ranks. Bug 5 (issue 656) is a tied embedding and output layer that came untied. The table does not say which repository each number belongs to | TTrace p. 9 (Table 1), p. 7 | Proposal 5 named this set without the list | Try bug 9 and one of 6 and 12 first | (a), a test |
| F13 | **Cloud providers say they cannot look inside the model.** Alibaba chose to instrument the communication library because in-model tools "would be rejected by most of our customers" | Alibaba's Aegis pp. 2–3, 6, 13 | TensorQuill is installed by the model's owner | A positioning point: it sits where providers say they cannot | (c) |

### What the full read confirms

| TensorQuill does | Source |
| --- | --- |
| Records the gradient norm before clipping (`clip_grad_norm`) | OLMoE clips at 1.0, which hides its gradient spikes from the update (p. 15) |
| Refuses to say where two runs part when their rank counts differ (`compare.py`) | The loss gap between 64 and 128 GPUs was larger than the gap a faulty GPU caused (Zheng et al. p. 9) |
| Gives a rank 1,800 seconds before its first step | Initialisation "usually takes between 5 to 20 minutes", and 73% of failed tasks fail in the first 10 (Alibaba's Aegis p. 10) |
| Watches for a hang before the first stored step | The same 73% |
| Hooks at module boundaries | OpGuard chose the lowest Python operators that start device work, for the same reason: they exist in every correct run (p. 8) |
| Has `attention_logit_growth` | QK-norm was adopted to stop attention logits growing (OLMoE p. 16, OLMo 2 p. 5) |

### What changes in the ranked proposals

The list stays at ten. Nothing is added as a new row.

- **Proposal 4** gains the router-reading note of F1, which is now written into its row in section 1. Of everything in this section it should be done first: it is a wrong incident on a healthy model, which costs more trust than a missed one. Proposal 4 also gains the wording of F10, and the agent instructions gain F3, F4 and F6.
- **Proposal 3** gains the measurement in F2.
- **Proposal 5** gains the list in F12 and the natural case in F7.
- **Proposal 6** is limited to hangs (F9). Qualified in section 9 (D3): contested, since ByteRobust uses repeated snapshots for slow jobs.
- **Proposal 9** gains F8 and correction C3.

### What was read this time, and what still was not

Text only, as extracted by `pdftotext`. No figure was viewed.

| Paper | Read page by page | Still not read |
| --- | --- | --- |
| Zheng et al. | pp. 2–15 | References |
| AEGIS (Lei et al.) | pp. 2–15 | References |
| OpGuard | pp. 2–15; Appendix A.4, A.5; the two bug tables (p. 23) | Appendix A.1 to A.3 (pp. 16–21) |
| CCL-D | pp. 1–12 | References |
| Greyhound | pp. 2–13 | References and after |
| Alibaba's Aegis | pp. 2–13 | The appendix (pp. 17–18) |
| Wang et al. | pp. 1–9, 12–14 | pp. 10–11 (Appendix A and B) |
| OLMoE | pp. 4–23; Appendix F, G, H (pp. 54–62) | Appendix A to E (pp. 44–53), I, J |
| OLMo 2 | pp. 5–18 | p. 19 onward: data, post-training and the appendix |
| TTrace | pp. 6–12; Table 1 with its layout | pp. 2–5 came out doubled and partly garbled, so their detail is unsure; the appendix (pp. 13–17) |
| Liu et al. | pp. 2–9; Appendix D | Appendix A to C (the proofs) |
| Ma et al. (ACL version) | pp. 3–9; Appendix D | Appendix A to C and E |

Not done: no real MoE model was run for F1, no real multi-GPU job for F2, and none of the agent-instruction or README changes above was made. Nothing in the repository was changed except this file.

---

## 9. Page-by-page read of the first 22 papers (Nov 18, 2026, third pass)

Sections 1 to 6 were written on Nov 17 from first pages and keyword search (section 6 says so). The author asked for the same page-by-page read that section 8 gave the second batch. This section is that read, with each new point checked against the code.

**Two places hold the code.** "Main" is the main checkout at commit `93e7d77`. "The branch" is `claude/lane-c-update-size-18d2d4`, where the update-size kind (`update_spike`) is built and not yet merged. I read the branch with `git show` and did not check it out.

**How it was read, and where the read fell short.** Text only, as extracted by `pdftotext`. To skip figure debris I filtered out short lines. A check afterwards showed the filter also dropped real prose in three two-column papers: 29% of Rotational Equilibrium, 37% of FlashOptim and 50% of the INT8 paper. It dropped under 4% elsewhere. Four reads were also cut short at the end. I then re-read without the filter: all of Rotational Equilibrium's main text, all of FlashOptim's, the four cut ends, and the pages of MegaScale I had skipped. **One paper is still only partly read:** the INT8 inference paper, at 43% of its text. Appendices are mostly unread, as before. The table at the end of this section says what was seen for each paper. Five figures and tables were checked on the rendered PDF page, because their numbers are not in the text.

Three things matter most:

- **Nine corrections to sections 1 to 8.** Three overturn a statement (D1 to D3). Two of those three are in the plan (rows K5 and K7). The other six narrow or sharpen a statement.
- **Finding F1 is not a corner case.** Three more training setups in these papers pick experts by something other than the plain top-k of the router's scores. One is confirmed by its citation, one is my inference in a reinforcement-learning setting, and one is second-hand (G1).
- **Nothing found here is a new capture defect.** The new points are wording for the agent instructions and README, and design input for kinds not yet built.

### Corrections to sections 1 to 8

| # | The review said | What the paper says | In the plan? |
| --- | --- | --- | --- |
| D1 | Hidden expert collapse has one case report behind it: experts that all get tokens and compute the same thing, output cosine above 0.99 (section 1, candidates table; section 3, row 19) | That is the starting state, not something training produced. The two routed experts in each layer are exact copies of one dense network's weights, and they are frozen: "trainable parameters = gating networks + shared experts (routed experts frozen)" (Sha pp. 5–6, 12). Two frozen identical copies give the same output by construction. **No paper in the folder measures experts growing alike in training.** I searched the three MoE papers of the second batch as well; the one that looks at overlap between experts, OLMoE, finds "little redundancy" (p. 20). The other nine of that batch are systems papers | Yes. Row K7 names this as its one source **Superseded in part by section 10, E1: Motif 3 names hidden collapse and gives monitors for it. The reading of Sha here stands.** |
| D2 | Proposal 10: output-logit growth in the head as a kind | The logit norm grows in every AdamW run in the paper, the stable ones too: it "grows under every AdamW baseline, including FP32 ... but it destabilizes training only in reduced precision" (Chen et al. p. 2, Fig. 1 caption; p. 8). What separates the failing runs is the gap between BF16 and FP32 logits on the same hidden states. A kind on growth alone would fire on healthy runs. The second number in proposal 8 is the one to judge | Yes. Row K5 is worded as growth |
| D3 | Section 8, F9 (iv): stacks help with hangs only, and proposal 6 "is limited to hangs" | Too strong; the papers disagree. CCL-D says a slow rank's stacks look like everyone's. ByteRobust uses stacks for slow jobs anyway: it repeats the stack grouping every 10 seconds, flags the group with the most outliers each time, and after 5 rounds takes the group flagged most (p. 8). It names a group of machines, not a rank, and gives no accuracy figure. So: one snapshot cannot name a slow rank; whether repeated snapshots can is contested | No |
| D4 | Section 5: the loss-spike repository is a from-scratch failure TensorQuill can be run on, to test `attention_logit_growth` among others | The paper's models are written in JAX and Flax and trained mostly on TPUs (p. 3). TensorQuill hooks PyTorch. Every run also uses QK-norm, which removes attention-logit growth on purpose (p. 3), so that kind could not be tested there in any case. The repository itself was not looked at | No |
| D5 | "What the papers confirm": bf16 weights whose updates round away, citing Sha p. 14 first | Sha's stuck case is a block with both weight matrices set to exact zero. I checked on CPU in float32: both matrices get exactly zero gradient and never move (appendix). No rounding is needed to explain it. The paper's fix changed two things at once: float32 master weights and a small random start. Chen et al. and FlashOptim still support the row | No |
| D6 | W8: SPAM "measured that zeroing those spikes improves final perplexity" | True, with three limits the review left out. One model, LLaMA-60M (p. 5). The gain depends on the threshold: perplexity 34.2 untouched, about 33.45 at a threshold of 100, about 33.95 at 50 (Fig. 5, read from the rendered page). And a "spike" there is one gradient element against its own average, counted over the first 1,000 steps only (p. 3) | No |
| D7 | W4: TensorQuill's kinds could plausibly catch "about five of the 20" TrainCheck bugs | With the bug table read on the rendered page, the count is up to seven: the three out-of-sync cases (B3, B9, B10), the two where the model stops learning (B1, B8), data-loader workers with one seed fetching the same data (B6, the family of `duplicate_batches`), and a hang from conflicting collective arguments in an MoE layer (B16). B1 and B3 are doubtful. Still untested. The two that loss monitoring caught are B1 and B8, each after 4 epochs (Table 4) | No |
| D8 | Candidates table: Rotational Equilibrium gives `sqrt(2·lr·wd·(1−β1)/(1+β1))` as a reference for the update-size candidate | That is the paper's expected rotation of a decayed weight vector. The value that matches what the kind measures is in the column beside it: the RMS update, `lr·sqrt((1−β1)/(1+β1))` per element, with no weight decay in it (Table 1, p. 4). It is 0.229 × lr at β1 = 0.9. See G5 | No |
| D9 | Section 8, C3: Ma et al. proposed a same-input comparison of two ranks and did not build it | They built one as a measuring tool. In their per-module study the faulty host and a healthy host run in step on the same inputs, and a function compares each tensor across the two before the reduction (arXiv version, Appendix B and C, pp. 14–15). The detector, a shadow replica, is still only proposed | No |

Smaller facts corrected in place in section 3: MegaScale appeared at NSDI and the straggler study at OSDI (both acknowledgments). The "injected FlashAttention fault" of paper 1 is not a hardware fault or a bug: the authors clear low-order fraction bits in two tensors to imitate lost precision, and call it "our controlled stress test" (Appendix C, p. 15).

### New findings, each checked against the code

Bucket (a) is wording, documentation or a small change. Bucket (b) is design input for a kind in the plan's K list.

| # | Finding | Source | TensorQuill today | Change | Bucket |
| --- | --- | --- | --- | --- | --- |
| G1 | **Three more setups route by something other than the top-k of the router's scores** (finding F1). (i) A 4B MoE trained with "a separate traffic-driven router-bias controller", citing the loss-free balancing paper. (ii) Routing replay: in training, the expert mask is the one recorded at inference, while the gate still uses the training scores; about 10% of routers pick differently between the two. That this breaks a top-k count is my inference. (iii) A 128-expert video model with a sigmoid gate and a loss-free expert bias, as one paper describes another | EnergyCal p. 4; R3 pp. 6–7; Sha p. 3 | Main: unchanged since section 8. The README still does not say how a score-returning router is read | The same change as F1. This raises its weight | (a) **Section 10, E2: DeepSeek-V3 and Motif 3 confirm the bias design first-hand.** |
| G2 | **A stall's stacks stop at the rank's own process.** "Root causes may reside in subprocesses spawned by the main training processes for tasks like data fetching or checkpointing; merely analyzing the stacks from the primary training processes is insufficient" | ByteRobust p. 8 | Main: `thread_stacks` reads `sys._current_frames()`, the Python threads of one process (`flight.py`; README, "Where a silent rank is stuck"). Data-loader worker processes are not in it. `data_stalled` and `input_starvation` cover the data path by timing | Say in the README and agent instructions that the stacks do not include worker processes. Reading their stacks would be new code | (a) wording |
| G3 | **The rank that reports a collective timeout is often not the cause.** SCOUT cites a PyTorch blog post on Flight Recorder: "almost all observed timeouts arise from desynchronization", so the reporting rank and the collective seen at that moment "may both be victims". Second-hand: I did not read the post | SCOUT p. 10 | Main: `collective_hang` and `collective_desync` read Flight Recorder. The agent instructions have no line on this (searched for "victim", "first to report") | One sentence in the agent instructions | (a) wording **Section 10, E3: now read in the post itself.** |
| G4 | **For slow ranks (K6): what to measure, and when there is no rank to name.** (i) A collective's time splits in two: waiting for the last peer to start, and the transfer after that. Only the wait points at a rank, and it needs start times on a common clock. (ii) In one production case the lag passed back and forth between ranks: "Rank A may initially lag behind Rank B but might eventually surpass". The cause was code, such as irregular garbage collection. No rank was slow. (iii) A collector pause stalls the forward pass only, because the backward pass is launched from C++. Uneven sequence lengths slow forward and backward together (correlation 0.9 or more in 21.4% of slow jobs). (iv) Slowdown is steady from step to step, so a few sampled steps are enough | Lin et al. pp. 5, 8, 11–12; MegaScale p. 12 | Main: no slow-rank rule. The plan's K6 row already says the rank that waits least is the slow one, which agrees with (i) | Add (ii) and (iii) to the K6 row: the answer may be "no single rank", and forward-only against forward-and-backward tells collector pauses from uneven data | (b) |
| G5 | **For the update-size kind (K1): a reference value, an outside test, and a public injector.** (i) The paper's steady value, 0.229 × lr at β1 = 0.9, held roughly on a CPU toy while the loss was still high: 0.268 at steps 100 to 200. The first step was 0.985 × lr. By step 3,000, with the loss at 0.02, it had fallen to 0.055 (appendix). So the level is not steady over a run, and the first steps are about four times the settled value. (ii) TrainSDC re-implemented the detector from Altenbernd et al., with replay of flagged steps. It cost 11.2% to 12.1% there, against about 1% in the original. On faults in 10 activation elements it left most of the damage (final loss gap 1.08e-2 against 1.72e-2 unprotected). On 100,000 elements it kept training finite where the unprotected run did not. Their faults are in forward activations, a different fault model, so this is a second data point and not a refutation. (iii) The original authors' injector is public: `github.com/aaltenbernd/llm-sdc-training` (p. 2). It needs a GPU | Rotational Equilibrium p. 4; TrainSDC p. 7 (Table 1, rendered page); Altenbernd et al. p. 2 | The branch: reads only `torch.optim.Adam` and `AdamW`, or a subclass that leaves the step alone; anything else is recorded as `update_not_read` with its reason (`_not_adam` in `capture.py`). The baseline is the mean of the last 100 unflagged changes (the `update_window` option, `evidence.py`). I searched the branch's evidence code for the update over the learning rate and did not find it | Put the update's RMS over the learning rate in the evidence, beside the expected value. Measure the cost on a GPU before quoting 1% | (b) |
| G6 | **For the cross-rank compare (K2): what the one published version needed.** 100 clean steps to calibrate, a threshold floor for each layer and path, a 64-step history, and an alarm only when the worst rank's score is 1.5 times the next. At the loosest setting it flagged 16.5% of clean steps; at the chosen one, 1 clean step in 1,628. Recall was 41% to 43% for faults in 10 elements and 78% to 86% for 100,000. Most misses were faults in attention weights and attention logits. The test ran one 1,024-token sequence per rank, six ranks, 814 steps | TrainSDC pp. 6, 10–11 | Not built. The plan's K2 row already says different batches per rank force a tolerance | Record these with the K2 design choice | (b) |
| G7 | **Two published healthy runs that support the prediction in finding W7.** Still a prediction: `activation_drift` was not run on either. A GPT-2 trained from scratch with nanoGPT's defaults for 50,000 iterations ends with a largest block-output value near 950 from layer 5 on. In a healthy 1.5B Llama 3 at BF16, the kurtosis of block outputs rises from about 10 to about 100,000 in the first 20 billion tokens (both read from rendered figures). The second is per-token kurtosis, which finding W3 says is a different statistic from TensorQuill's `abs_max / std`, so it is suggestive only. The first is a largest value, which is closer. Also, a run that survived the steady phase diverged when the learning rate was cooled, with "considerable changes in model statistics" after the decay | Massive Activations p. 10, Appendix B.4; FP8 paper pp. 4, 6 | Main: the rule compares with the module's first 50 records and reports a 3× move. Still predicted, not shown on TensorQuill | Run it on a from-scratch model (the plan's M2). Tell the agent that a shift at a learning-rate change is expected | (a) |
| G8 | **Five named causes worth a line each in the agent instructions.** (i) A router whose scores are all equal at the start picks the first experts every time: "top-k defaults to returning the first index, leading to global deadlock on expert 0". (ii) The gains of a query-key normalisation layer, trained without weight decay, grew until a run with FP8 attention diverged. (iii) Skipping the batch at a spike only postpones it, and freezing the suspected matrices for the one update before a spike did not stop two examined spikes. (iv) A starved expert's weights still move, from momentum. (v) Long runs are restarted with changed data mixes, context lengths and code as routine: 28 restarts in 10 days on one job | Sha pp. 8–9; FP8 paper p. 4; Chen et al. p. 4, EnergyCal p. 8; EnergyCal p. 7; ByteRobust pp. 3–4 | Main: none of the five is in `AGENT_INSTRUCTIONS.md` | Add them. For (v): check for a restart with a changed setup before reading drift as a fault | (a) wording |
| G9 | **Which bit flips change the result.** In two injection studies only high bits mattered. Mantissa and sign flips in matrix-multiply instructions "did not show noticeable differences". Flips in the low 24 bits of an INT32 accumulator changed no answers. Published practice is still exact comparison: a difference of 1e-8 in one value counts, a bit-wise test of one training step per machine, and XOR fingerprints | Altenbernd et al. p. 3; SProbe p. 6; TrainCheck p. 12; ByteRobust p. 7; TrainSDC p. 4 | Main: `1e-6` tolerance on float fingerprints (finding F2) | Context for F2, and the link is my reading: these studies injected passing faults in computation, while F2 is about a weight that stays flipped. With that caveat, the small flips the tolerance hides are of the kind these studies found harmless. Exact equality is still the cheap, standard thing to record | (a) |
| G10 | **An optimizer can use and free a gradient before the step ends.** FlashOptim's "gradient release" applies the update during backpropagation as each gradient is computed. It cites LOMO, AdaLOMO and Adam Accumulation as doing the same (p. 2). Its weights are also split (finding W10) and its Adam state is 8-bit | FlashOptim pp. 2, 4–5 | Main: per-parameter gradient norms are read at the optimizer step. With gradient release there may be nothing left to read. Not tested; my inference from the description | A line beside W10 in "Things to know" | (a) wording |

### What the full read confirms

| TensorQuill does | Source |
| --- | --- |
| Says the drift check compares the wrong ranks under tensor parallelism unless told the group (`sdk/README.md`, "Tensor parallelism") | The BLOOM bug was between tensor-parallel ranks, in the under 1% of weights that are copied and not split; in the reproduction it showed "only under" that condition (TrainCheck pp. 4, 8, 12) |
| Reads a router that returns a tuple by taking its first integer tensor as the chosen experts (`_router_hook`) | The public DeepSeek-style gate returns `(topk_idx, topk_weight, aux_loss)` (Sha pp. 8–11) |
| Refuses to read an update from an optimizer that is not Adam, and says why (the branch) | Matrices trained with Muon, Adam state stored as 8-bit integers, and an optimizer that resets its moments every 500 steps all exist (EnergyCal p. 2; FlashOptim p. 4; SPAM pp. 6, 8) |
| Checks a resumed run against the run it resumed | At restart, operators roll back a few steps and check the loss curves stay "bitwise consistently aligned" (ByteRobust p. 4) |
| Has `duplicate_batches`, `allocator_retries`, `gpu_throttled`, `hardware_errors` | Workers with one seed fetching the same data (TrainCheck p. 11); memory fragmentation as a slowdown cause (Lin et al. p. 13); thermal throttling found in 10 seconds and matched to lost throughput (ByteRobust p. 11); memory errors are 38.9% of the faults counted (Minder p. 4) |
| Gives a rank 1,800 seconds before its first step | Setting up the communication groups alone took 1,047 seconds on 2,048 GPUs with stock PyTorch (MegaScale p. 5) |
| Compares replicas by fingerprint | Altenbernd et al. cite a 2023 engineering post that recommends "checksumming of parameters and optimizer states" across replicas, and note it does not cover faults that pass through the gradient reduction (p. 2). That is finding W1 from another side |
| Judges against peers or own history, not a fixed level | The normal level depends on the job: 70 °C is a fault at one clock speed and normal at another (Minder p. 4) |

### What changes in findings W1 to W10 and the ranked proposals

The list stays at ten. No row is added.

- **W1** stands, and is stronger. LLM-PRISM's job was data-parallel on 16 GPUs with one faulty rank (p. 7), so its result is not a single-GPU one. Its weight divergence is still measured against a fault-free run.
- **W4** changes its count (D7). TrainCheck's "under 2%" is a rate per invariant: on one real case 100 violations were reported, 52 of them true (pp. 15–16).
- **W7** gains two published runs (G7).
- **W8** is weaker than written (D6).
- **W9** gains G9.
- **W10** gains G10.
- **Proposal 4** gains G1 and the wording of G2, G3 and G8.
- **Proposal 5** gains the public injector (G5) and loses the loss-spike repository as a direct test (D4).
- **Proposal 6:** whether it reaches slow jobs is contested (D3). It gains G2: say which processes the stacks cover.
- **Proposals 8 and 10:** judge the BF16-to-FP32 gap, not the growth (D2).
- **Proposal 9** gains G6.
- **Candidates table:** the fourth row has no paper behind it (D1). The second row's reference value is corrected (D8).

### What was read this time, and what still was not

"Share seen" is the share of the pages' extracted text that reached me after the filter and the cuts. No figure was viewed except the five named.

| Paper | Read | Share seen | Still not read |
| --- | --- | --- | --- |
| 2606.28116 | Main text in full on Nov 17. Appendix pp. 14–19 skimmed now | First part of each appendix page | The proofs |
| TrainCheck | pp. 1–16; Tables 3 and 4 on the rendered pages | All | References |
| SCOUT | pp. 1–14 | All | References; the `lm-resiliency` code |
| Ma et al. | Main text on Nov 18 (ACL version). Appendix A to C and E of the arXiv version skimmed now | Part; the tables are only partly legible | The appendix in full |
| Altenbernd et al. | pp. 1–9 | All | Figure 4, which holds the detection rate |
| LLM-PRISM | pp. 1–11 | All | Figures |
| TrainSDC | pp. 1–12; Table 1 on the rendered page | Nearly all | Nothing in the text; figures |
| Massive Activations | pp. 1–12; Appendix B.4; Figure 10 rendered | 97% | Appendix A, B.1 to B.3, C |
| FP8 paper | pp. 1–9; Figures 4 and 5 rendered | 98% | Appendices and checklist |
| Rotational Equilibrium | pp. 1–9, after the unfiltered re-read; Table 1 on the rendered page (Nov 17) | All | The appendices (pp. 10–37) |
| Lin et al. | pp. 1–16 | 96% | References |
| Minder | pp. 1–12 | 91% | The appendix |
| ByteRobust | pp. 1–14 | 97% | References |
| MegaScale | pp. 1–13 | Nearly all | References |
| Chen et al. | pp. 1–10 | All | Appendix (pp. 11–27); the repository |
| SPAM | pp. 1–10; Figure 5 rendered | Nearly all | Appendix |
| EnergyCal | pp. 1–9 | Nearly all | Appendix (pp. 10–14) |
| R3 | pp. 1–11; part of pp. 12–13 | 84% | Appendix (pp. 15–33) |
| Sha | pp. 1–17 | 95% | pp. 18–23 |
| Li et al. | pp. 1–10 | Nearly all | Appendix (pp. 11–47) |
| FlashOptim | pp. 1–8, after the unfiltered re-read; p. 9 in part | Nearly all | The appendix; the code |
| SProbe (INT8 inference) | pp. 1–9 through the filter | 43% | Not re-read. It is inference only and nothing here rests on it beyond one table row |

Not done: none of the README or agent-instruction changes above was made, and the plan was not edited. No GPU run, no real model. The two CPU checks are toys. G1 (ii), G10 and the count in D7 are readings, not tests. Nothing in the repository was changed except this file.

---

## 10. The eleven added papers and posts (Nov 18, 2026, fourth pass)

Thirteen files were added to `../researchpapers/` on Nov 18. Eleven are new: nine papers and two blog posts. Two are fresh downloads of papers already read (TrainSDC and Ma et al., same file names and arXiv versions; their extracted text differs from my first copy by under 0.5% of the words, which I did not trace, and I did not re-read them). **Traced in section 11, row J8: the text is the same.** The folder now holds 45 distinct papers and posts.

**How they were read.** Nine of the eleven: the whole main body, page by page, with no filter on the text. Two in part: DeepSeek-V3 (18 of 53 pages) and Motif 3 (13 of 32). Appendices are mostly unread. **Section 11 reads the rest.** The table at the end of this section gives the pages for each. Every table I quote was re-read with the page layout kept, because plain extraction mixed up rows and columns. The code was read on `main` at `b11b2c6`, where the four lanes are merged, so `update_spike` is now on `main`.

**The one thing to act on first.** Section 9 (row D1) and plan row K7 say hidden expert collapse has no paper behind it. That is no longer true. Motif 3 names it and gives monitors for it (row E1 below). The plan was corrected to "without a source" at `f3524d4` on my word, and that correction should be taken back in part.

### What this pass changes in sections 1 to 9

| # | Earlier statement | What the added papers say | Effect |
| --- | --- | --- | --- |
| E1 (a second reported collapse: J1, section 11) | D1 and plan row K7: hidden expert collapse has no paper in the folder | Motif 3, pp. 16-18, "Expert-health monitoring" and Table 3. It names "hidden collapse in which an expert receives tokens but its output projection contributes increasingly little" and "functional collapse in which nominally distinct experts learn similar representations". It gives a monitor for each: the minimum-to-median ratio of the experts' output-projection norms (above 0.8 "typically healthy", a sharp fall below 0.5 means an expert "may be dying"), and the largest cosine between experts' mean outputs (below 0.5 "typically healthy", above 0.8 "severe functional redundancy") | **D1's "no paper" is superseded.** K7 has one source for its design and working values. What stays true: Motif 3 reports no occurrence of either hidden failure, only visible routing collapse in early layers (p. 16). The values are one team's experience from one run, with no detection rate. D1's reading of Sha (frozen identical copies) stands. The focused plan (line 235) cited Motif 3 "from search excerpts only"; it is now read, in part |
| E2 (Mamoda2.5 is now read: J2, section 11) | G1: three more setups route by something other than the top-k of the hooked scores; one confirmed only by citation, one second-hand | Two first-hand confirmations. DeepSeek-V3, p. 9, eq. 16: a per-expert bias is added to the score to pick the top-k, "the bias term is only used for routing", and the gate value still comes from the score without the bias. Motif 3, p. 16: the same bias design, and Gaussian noise added to the router logits early in training (eq. 28), decayed to a floor | F1 now rests on two of the best-known MoE recipes. Noise is a second way hooked scores differ from the routing; plan row S7 already names it. Mamoda2.5 is not in the folder, so that one setup stays second-hand |
| E3 | G3: "the rank that reports a collective timeout is often a victim" was second-hand through SCOUT | The post itself, pp. 5 and 8: the rank that first raises the timeout "is rarely the culprit", the collective running at the timeout "may not be the cause", and "almost all" timeouts in Meta's fleet come from ranks falling out of step, not from a slow or hung collective. Causes on the CPU side are above 60% (p. 10); network or hardware is 20 to 30% (p. 15) | G3 is first-hand. Scope: one company's fleet, four training frameworks, three of them recommendation systems and one LLM. The per-cause breakdown is a figure (Fig. 3) I did not read values from |
| E4 | D3: "stacks help only for hangs" is contested (ByteRobust p. 8) | Two more sources pair collective records with the main thread's Python stack on every rank. The Flight Recorder post, p. 22: Flight Recorder "needs to be coupled with" main-thread stacks to tell the cause, PyTorch provides no tool for it, and py-spy is suggested. Mycroft, pp. 8 and 12: on a trigger it dumps every rank's Python stack, groups identical stacks, and the rank with a different stack is the stuck one (case two: one rank in the data loader, its peers waiting in a broadcast) | TensorQuill's stall stacks plus its Flight Recorder reading is the pairing both describe. Grouping stacks across ranks and naming the odd one is a small step TensorQuill does not take today (I searched `evidence.py`, `flight.py`, `trigger.py` and `server/src` for stacks being grouped or compared and found none) |
| E5 | W1 and proposal 1: what `replica_drift` can see of silent corruption. I also said the gradient-aggregation paper (PAFT) "bears on W1 directly" | It bears on the case `replica_drift` already sees. PAFT's fault is different noise on each rank's copy of the reduced gradient, so the replicas drift apart (pp. 3-4). It says twice that a corruption shared by all ranks is outside what it handles (pp. 8, 9). Elsen et al. give the cleanest split, in the table below | W1 stands and gets sharper. My "directly" was too strong. `sdk/AGENT_INSTRUCTIONS.md` line 109 on `main` already says a clean drift check does not rule out corruption |
| E6 | I named "Dai et al." as the paper on expert specialisation and hidden-state geometry | The paper Sha cites is Wang, Hayou and Nalisnick, "The Myth of Expert Specialization in MoEs" (arXiv 2604.09780). That is the file added. It is Sha's reference [11], "the expert specialization myth". Dai is an author of other papers that Li et al. cite, among them DeepSeekMoE | My mistake in the name. The paper does not support K7 either: it is a study of released models at inference, with no training run |
| E7 (the sign is not fixed: J6, section 11) | D2, K4 and K5: judge the BF16 to FP32 gap, not logit growth | Wortsman et al. are the source of the output-logit instability. Their sign is not the norm: the logits "diverge and become very negative" (p. 4), plotted as the output-logit mean (Fig. G.2, p. 18). It comes late in training, with no weight decay, and a small penalty on log Z removes it | K4 already plans the mean. Add log Z (the log of the softmax's normaliser) beside it. It costs nothing more on the probe's logits |
| E8 | Plan row K6 quotes "42 of 107 four-node jobs slowed by congestion" | Verified on the page (FALCON, p. 4, Table 1): 43 of 107 slowed, 42 by network congestion and 1 by CPU contention. One node: 6 of 392 (4 CPU contention, 2 GPU). At 512 to 1,024 GPUs: 16 of 27 (13 network, 3 several causes), from one month of traces inspected by hand | The figure holds. More for K6 in row H6 below |

**Where a silent fault can sit, and what TensorQuill sees (Elsen et al., pp. 5, 9-13; PAFT, pp. 8-9).**

| Where the fault is | What the replicas look like | `replica_drift` | Anything else on `main` |
| --- | --- | --- | --- |
| One rank's local gradient | All ranks get the same wrong sum and stay equal. "No way to detect this" without repeating the computation (Elsen p. 9) | Cannot see it | `update_spike` fires if the gradient norm rises (CPU check below). A small error is seen by nothing |
| The reduction, first half (reduce-scatter) | The same: a wrong chunk reaches every rank (Elsen p. 13) | Cannot see it | As above |
| The reduction, second half (all-gather) | Every rank after the fault in the ring gets the wrong chunk. Elsen saw one third of the ranks agree on one checksum and two thirds on another (p. 10) | Sees a spread. Cannot show the pattern (row H2) | None |
| One rank's parameter update | One rank differs from all the others | Sees it | None needed |

Elsen's footnote 9 and PAFT's limitations (p. 9) agree on one more limit: under full sharding (FSDP full shard, ZeRO stage 3) no parameter is held twice, so there is nothing to compare. Motif 3 trains with a sharded dimension of 8 inside a replicated dimension (p. 7), so the check applies across the replicated dimension only. The README's note covers tensor parallelism and the docstring names hybrid FSDP; neither says the check has nothing to run on under full sharding alone.

### New findings

| # | Finding | Source | Checked against the code |
| --- | --- | --- | --- |
| H1 | **`attention_logit_growth` misses a slow rise.** The rule compares each probe with the median of the last five probes that met no rule. A logit that rises steadily never gets ahead of that moving baseline unless it grows by about 26% or more per probe, which is a doubling within three probes. A rise from 10 to 8,078 at 25% per probe gives no incident. **Probing more often makes the rule blinder**, because the same rise is then spread over more probes. Chen's rise (about 700 times in about 5,500 updates) is about 3.3 times per probe at the default of one probe per 1,000 steps, which would fire, and about 1.13 times per probe at `probe_every=100`, which would not. A run ten times longer with the same total rise is missed at the default too | The kind of failure it matters for: Chen et al. (BF16 FlashAttention-3). Two of 24 layers reach a logit scale in the thousands while the other 22 stay moderate (p. 4). A second run, with the known forward fix, ends within 0.01 to 0.02 nats of the FP32 run with small gradients, yet layers 5 and 11 sit at a logit scale of about 39,000 and 49,000 and a third layer is above 1,000 (p. 9, Table 3; Fig. 3) | **Reproduced on `main`** at `b11b2c6` with the detector unchanged (appendix). Fires at 27% per probe and above, not at 25% and below. The two figures for Chen's run are my arithmetic from its Table 3, not a run. `attention_collapse` uses the same moving baseline (read, not run) |
| H2 | **`replica_drift` throws away which ranks agree.** Each rank's fingerprint reaches every rank in the `all_gather`, and the code keeps only the largest minus the smallest. The incident names the parameters and the spread, never the ranks | Elsen et al., p. 10: the pattern is the diagnosis. One rank apart points at that rank's update. A run of neighbouring ranks apart points at the network during the gather. They also could not tell which group was right: a majority is not proof | Read: `distributed.py` (`replica_spread`, `spread_from_fingerprints`), `evidence.py` (`detect_replica_drift`). Not run. It also compares parameters only, not optimizer state; a state fault shows as parameter drift one check later, so this matters less |
| H3 | **A fault in Adam's state is not seen by the two rules that could see it.** With one tensor's second moment wrong by 1e12 and the gradient untouched, that tensor's update falls to about 3e-7 of normal at once and stays there: for the rest of the run at β2 = 0.999, for about 500 steps at β2 = 0.95. `update_spike` does not fire, because it watches the largest update across parameters and needs the gradient norm to rise. The weights-not-updating check needs a norm that is exactly unchanged, and here it never is | He et al. (ISCA '23), pp. 2, 5-9: of 2.9 million injected faults, 9.7% to 17.7% changed the training outcome. For every slow or hidden outcome, large values in the optimizer's history or in a normalisation layer's moving variance were a necessary condition, present within two iterations. For faults in the backward pass the training loss "appears normal throughout" (p. 8). A fault can change the history without changing the gradient (p. 11) | **Run on CPU against `main`** (appendix): `update_spike` fires on a gradient fault and stays silent on the state fault; 0 of 15 norm checks exactly equal. The second rule's condition was emulated, not the detector. A toy MLP, no clipping, 1,500 steps |
| H4 | **What a stuck collective's record says about the cause.** The post maps Flight Recorder's fields to four causes (p. 19). A missing rank or a state mismatch: a CPU-side stall or code path that differs across ranks (data loading, checkpointing, compilation, an `except` block that swallows or blocks, one rank running out of data early; pp. 10-13), or a hung GPU kernel. A size or dtype mismatch: wrong arguments. A mismatch in the collective's type or call stack: code paths that differ across ranks. All ranks started and none finished: the network | Flight Recorder post, pp. 8-19. Raising the timeout does not fix a desync (p. 8). The watchdog's default is 10 minutes (p. 4) | Read: `collectives.py` already reports `missing_ranks`, `state`, `all_ranks_entered` and the analyzer's mismatch line. It labels a collective with a missing rank `hang`; the post counts a missing rank as desync, its most common case. The plan's 300-second stall is under the 600-second watchdog, so on defaults the stacks are read while the processes are alive (the 300 is the plan's number; I did not find the constant) |
| H5 | **A monitor table for expert health, from a real 314B-parameter run.** Six signals with working values. TensorQuill can compute two today and part of a third (table below) | Motif 3, pp. 16-18. One reported incident: routing balance in the earliest MoE layers "still collapsed occasionally" despite a stronger balancing loss there; they replaced the layer's router and biases with a neighbour's (p. 16) | Read: router counts give the first two. The 50-step per-parameter norm gives the third only when each expert's output projection is its own parameter; one stacked tensor gives one norm (my reading of the README, not run) |
| H6 | **Three more things a slow-rank observation (K6) must allow for.** (i) In synchronous training one slow part slows every rank at once, so step time does not name the rank; FALCON separates groups that transfer for longer than 1.1 times the median from groups that sit waiting (p. 7). (ii) Slowdowns are passing: mean 10, 24 and 72 minutes for the three settings (pp. 1, 4-5), so an observation needs a start and an end. (iii) Some slow ranks are the data's doing: under document packing one rank had 3.54 times the mean attention work (Motif 3, p. 13), and rank 0 often carries more by design (Mycroft, p. 13) | FALCON, Motif 3, Mycroft. FALCON's own detector: a change-point test on iteration time, then a check that the mean moved by 10% or more; 99.1% to 100% accurate on its 499 labelled jobs, where a plain 10% sliding window missed 12% to 25% of the slow cases (p. 11, Tables 4-5) | Plan row K6 already limits itself to compute-slow and to "may be no single rank". These add the passing nature and the data cause. Not built, so nothing to run |
| H7 | **Healthy models that would look like a routing fault.** (i) In gpt-oss-20b, unrelated prompts activate the same experts in the last layers; keeping 12 of 32 experts in the last two layers moves the loss on prompt text by under 10% (pp. 9, 11). (ii) Router confidence drops on shuffled or reversed text (p. 6), so per-token routing entropy moves with the data as well as with the router. (iii) A balancing loss applied on small batches is reported to erase specialisation (p. 5, citing Qiu et al.; second-hand) | Wang, Hayou and Nalisnick. All measured on released models at inference; no training run. The "dead router" case is their hypothesis, which they say they did not test (p. 6, footnote 1) | Bears on `moe_imbalance`, `starved_experts` and plan row K3. It supports G7 and plan row S8 ("a narrow data domain routes unevenly in a healthy MoE"). Not run |
| H8 (lead time and scope corrected by J4 and J5, section 11) | **A named cause for the agent: BF16 fused attention, backward pass.** Healthy for 25B tokens, then the gradient norm grows about a thousandfold with no NaN and the loss ends 0.2 nats above FP32 attention (pp. 1, 9). The attention statistic leads: at 23.1B tokens the two layers already have a logit scale near 5,400 and 86% of rows at near one-hot attention, at least about 500 updates before the gradient norm starts to grow near 25.2B (Table 3, p. 9; Fig. 3 caption; 4.19M tokens per update, Appendix G; one checkpoint, and the subtraction is mine) | Chen et al., arXiv 2609.34272. One 450M model, FlashAttention-3 on Hopper GPUs, head dimension 64, a preprint. Their test for it: recompute the backward of the suspect layers in FP32 with the forward held fixed; the gradient norm falls by more than 100 times (p. 4) | TensorQuill's probe reads the attention scores from the weights with its own forward, so the kernel's fault does not hide the logits from it. I expect `attention_logit_growth` and `attention_collapse` to fire on the first run: a rise of about 700 times inside the first six of twelve default probes is far above 26% per probe. That is my arithmetic, not a run. TensorQuill cannot do the paper's test: it has no backward recompute |
| H9 | **Operating facts for the instructions, each from one team.** A faulty node usually shows within 1,000 seconds of a deterministic rerun, rarely after 10,000 (Elsen p. 9). A modern transformer going from fine to NaN in one step "almost never" happens without a hardware problem (Elsen, footnote 1; an opinion). Mycroft's real case: a defective GPU, every rank stuck at once, no error, the job killed by the 10-minute timeout, the fault reproduced only after six hours of offline tests (p. 2) | Elsen et al.; Mycroft | Wording only. The NaN line is an opinion and should be passed on as one |

**Motif 3's expert-health signals (pp. 17-18, Table 3) against what TensorQuill records.**

| Signal | Their working values | Failure it points at | TensorQuill today |
| --- | --- | --- | --- |
| Tokens to the least-used expert over the median expert | Above 0.7 healthy, below 0.3 likely starvation | Starved or dead expert | Computable from router counts. `starved_experts` uses another rule (share below 10% of an even share over 50 samples) |
| Largest token count of any expert | None given | Overload | Recorded |
| Output-projection norm, least over median | Above 0.8 healthy, sharp fall below 0.5 | Hidden collapse: tokens arrive, the output shrinks | Only with one parameter per expert |
| Routed output RMS over shared output RMS | Near 0: the shared expert does the work. Above about 0.3: routed experts contribute | Shared-expert dominance | Not recorded |
| Largest cosine between experts' mean outputs | Below 0.5 healthy, above 0.8 severe | Functional collapse | Not recorded. Needs expert outputs |
| Largest absolute value over RMS, routed and shared | Above about 10 is an early warning | Outliers | Recorded for watched blocks, not per expert. Do not copy the 10: healthy Granite block outputs reach 740 to 840 times their standard deviation (README) |

Their monitoring runs on the last microbatch of a step only, at the interval of their utilisation heatmap, to keep the cost down (p. 17). They also report that the FFN output RMS kept rising through training after they removed the isolated outliers (p. 17): growth alone is not a failure, the same lesson as D2.

**Reference values for attention logits. The statistics differ, so the columns are not comparable.**

| Source | Statistic | Value | What it marks |
| --- | --- | --- | --- |
| Motif 3, p. 11 | Largest pre-softmax logit per head, read every 10 steps from the attention kernel | 100 in main training, 200 at 256K context | The level they clip back to, not a failure line |
| Wortsman et al., p. 7 | Largest attention logit, block 0, step 2,000 | Above 1e4 | Every such run diverged |
| Wortsman et al., p. 17 | Logit scale forced to a constant in a 10M model | About 1e3 | Loss gets worse; at 1e4 it is worse than a model with no attention and no MLP layers |
| Chen et al., p. 9 | Scale × RMS of queries × RMS of keys × √d, per layer (an RMS-type number, not a maximum) | 7 to 8 in the two layers of the healthy runs (26 to 27 at most in any layer); above 1e3 in the failing ones | Healthy against failing, one model |
| TensorQuill | Largest logit per layer on the probe batch, against a moving baseline | Twice the baseline | `attention_logit_growth` |

Motif 3 had to extend the attention kernel to get the largest logit, because fused attention never builds the score matrix (p. 11). TensorQuill gets it by running its probe through plain attention. That works on a small probe batch and would not on every step.

### What this changes in the findings and proposals

- **W1 and proposal 1.** Partly done on `main` since the lanes merged: a checked `replica_drift` kind now carries a note in `kind_checks`, and the agent instructions say a clean check does not rule out corruption. What is left is the failure list's entry 2 and this: add the table above: `replica_drift` sees two of four places a fault can sit. Add row H2 to the same proposal: report how many distinct fingerprints there were and which ranks held each. The fingerprints are already gathered.
- **Proposal 6 (stacks) and D3.** Stronger. Two production teams pair collective records with Python stacks. Group the stacks across ranks (E4).
- **Proposals 8 and 10, plan rows K4 and K5.** Add log Z to the probe numbers (E7).
- **Plan row K1.** `update_spike` is on `main` and fires on a gradient fault in my check. It does not cover a fault in the optimizer's own state (H3). The per-parameter update is already in the records, so a rule for one parameter's update collapsing could be judged from what is stored; whether it stays quiet on healthy runs is not known. Motif 3 trains its matrices with Muon and only embeddings, the output projection and vectors with AdamW (p. 10), so on a run like it most parameters would be `update_not_read`.
- **Plan row K7.** Take back "without a source" (E1). The row still needs expert outputs, which nothing captures, and still has no reported case. (Section 11, J1: Laguna reports a second collapse, kind unstated; still no reported hidden case.)
- **Plan rows S7 and S8.** S7 gains two first-hand sources (E2). S8 gains H4, H8 and H9 as wording.
- **The attention rule (H1).** The smallest fix with a reproduced gap. Two designs: also compare with a baseline frozen from the first probes, or add an absolute bound. Neither can be given a number from these papers alone: the reference values above use three different statistics.
- **The list of ten stays ten.** H1 to H3 are defects or gaps in built kinds, not new proposals.

### The eleven, one row each

The "Not read" column is as it stood after this pass. Section 11's table says what has been read since.

| Paper or post | Kind of evidence | Read | Not read |
| --- | --- | --- | --- |
| Liu et al., "Flight Recorder: A New Lens for Understanding NCCL Watchdog Timeouts", PyTorch blog, Mar 25, 2026 | A company's fleet experience; no numbers beyond three shares | All 26 printed pages | Figures as images |
| Elsen, Hawthorne, Somani, "The Adventure of the Errant Hardware", Adept blog, Sep 19, 2023 | One team's incident account; values "illustrative" | All 15 printed pages | Nothing |
| Tang et al., "Identifying and Mitigating Errors in Gradient Aggregation of Distributed Data Parallel Training", ICML 2026 (PAFT) | Theory plus simulated Gaussian noise on 4 to 32 GPUs; no real fault | pp. 1-9 | Appendices, pp. 12-26, among them Appendix C on real error types. Table 1 would not extract; I quote none of it |
| Wortsman et al., "Small-scale Proxies for Large-scale Transformer Training Instabilities", ICLR 2024 | Measured, many small runs in JAX on TPUs in bfloat16 | pp. 1-9, appendix text and captions pp. 14-22 | pp. 10-13 (references), pp. 22-26 (more figures) |
| Chen et al., "Broken Symmetry in BF16 Attention", arXiv 2609.34272 | Measured, one 450M model, matched runs with one seed | pp. 1-10; Tables 1 and 3 re-read; the setup line of Appendix G | Appendices A to I otherwise, pp. 13-28 |
| Deng et al., "Mycroft", SOSP '25 | Production deployment plus fault injection on 32 GPUs | pp. 1-13 | References |
| Wu et al., "FALCON", arXiv 2410.12588 | One shared cluster; 499 probe jobs and 27 large jobs inspected by hand | pp. 1-13; Tables 1, 4 and 5 re-read | Appendix, p. 17 |
| He et al., "Understanding and Mitigating Hardware Failures in Deep Learning Training Accelerator Systems", ISCA '23 | Simulated bit flips in an open accelerator design, not a GPU; mostly CNNs, one Transformer, 8 devices | pp. 1-11; Table 4 re-read | Figures 2 to 5 as images |
| Wang, Hayou, Nalisnick, "The Myth of Expert Specialization in MoEs", arXiv 2604.09780 | Measured on five released models at inference; two propositions, one informal | pp. 1-11 | Appendices, pp. 14-31 |
| DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437v2 | A lab's report; ablations with benchmark tables | **In part.** pp. 8-9, 11-12, 14-18, 22, 26-27, 35, 44-48 | pp. 1-7, 10, 13, 19-21, 23-25, 28-34, 36-43, 49-53. The sentence on load imbalance that starts at the foot of p. 27 runs onto p. 28, which I did not read |
| Motif Technologies, "Motif 3: Technical Report", arXiv 2608.09119 | A lab's report on one training run; working values, no rates | **In part.** pp. 6-13, 16-19, 26; Table 3 re-read | pp. 1-5, 14-15, 20-25 (post-training and evaluation), 27-32 |

**Public material these papers point to (for section 5).** He et al. publish their fault-injection framework (`github.com/YLab-UChicago/FIdelityTraining`, their reference [1]); it is TensorFlow and models an accelerator that is not a GPU. FALCON says its cluster trace "will be released upon publication" (p. 1, footnote); I did not check whether it was. Motif 3 points to a training example repository with reference implementations of its system optimisations (reference [42]); I did not open it. Chen et al.'s failing run is a seeded, matched recipe (16 GPUs, seed 42, 11,930 updates) and so the closest thing here to a reproducible late-training failure, at a cost no CPU check can meet. None of this was downloaded.

**Not in the folder, so still unchecked.** Mamoda2.5, the Laguna report, LLMFT (Yu et al.) and the Dai et al. paper on DeepSeekMoE that DeepSeek-V3 builds on. (The first three were added later and are read in section 11; the fourth in section 12.)

**Not done in this pass.** No GPU run. No run of TensorQuill on any failure these papers describe. No reading of figure images except through their captions. No edit to the plans, the README or the agent instructions: plan row K7 and its "Decisions" paragraph still say "without a source".

## 11. Three more papers, and the pages section 10 skipped (Nov 18, 2026, fifth pass)

Three files were added after section 10. They are the three papers section 10 listed as missing: the Laguna M.1/XS.2 report (Poolside, arXiv 2605.27605), Mamoda2.5 (ByteDance, arXiv 2605.02641) and the LLMFT paper (Yu et al., SC '25). The folder now holds 49 files and 48 distinct papers and posts, because Ma et al. is there in two versions.

This pass also reads what section 10 left out: the rest of DeepSeek-V3 and Motif 3, and the appendices of five papers. Section 10 read two papers in part and skipped those appendices by my own choice. That was a weak choice, and this section makes up for it.

**How they were read.** The three new papers page by page: LLMFT all 15 pages, Laguna all 37, Mamoda2.5 pp. 1-28 with pp. 29-38 scanned (references, one benchmark table, picture pages). The appendix pages were read through a filter that drops short lines, so equations and table cells were not read as text; the tables I quote were re-read with the page layout kept. The table at the end gives the pages. The code was read on `main` at `b11b2c6`, unchanged since section 10.

**What to act on first.** Three things.

- A third production team checks replicas against each other, with hashes. It says the check "primarily" catches silent data corruption, which sits badly with finding W1. It gives one example and no counts (row M2).
- The BF16 attention finding H8 was drawn too widely and too narrowly at once. Four limits from the paper's appendices are in row J5.
- A weight corrupted in memory can pass the gradient rule and the update rule while its tensor's norm jumps by a million times, and no rule reads the weight norms (row M1, run on a CPU toy; only those two rules were tested).

### Corrections to section 10

J rows correct section 10. M rows further down are new findings.

| # | Section 10 said | What the page says | Effect |
| --- | --- | --- | --- |
| J1 | E1: Motif 3 reports "only visible routing collapse in early layers"; hidden expert collapse "still has no reported case" | Laguna, p. 9: in "preliminary experiments", Muon with no rescaling of its learning rate and standard weight decay "caused expert collapse beginning at around 450B tokens into training and propagating layer-by-layer through the first three MoE layers until the training diverged". Laguna, p. 17: in M.1, new special tokens with randomly initialised embeddings caused "gradient spikes, dead experts, and broader training instabilities". Motif 3, p. 1, names three failures its balancing guards against: "expert overload, starvation, and specialization collapse" | **A second reported collapse, and the first said to end in divergence.** Motif 3's (p. 16, row E1) was the first. Laguna does not say which kind it was (routing, hidden or functional), how it was seen, or how large the model in that experiment was. One sentence, no numbers beyond the token count. **Hidden collapse still has no reported case, as section 10 said.** Plan row K7 needs nothing beyond what E1 already asked |
| J2 | E2: Mamoda2.5 "is not in the folder, so that one setup stays second-hand" | Mamoda2.5, p. 7: the bias "is used only for the routing decision"; the gate weight still comes from the score without the bias. Its router uses a sigmoid and top-8 of 128 experts | F1 now has **three** first-hand sources (DeepSeek-V3, Motif 3, Mamoda2.5). Laguna is a fourth design: sigmoid scores normalised after the top-k, routed output multiplied by 2.5, and a balancing loss, not a bias (p. 5). As described, nothing is added to Laguna's scores before the top-k |
| J3 | The read table: PAFT's Appendix C is "on real error types", unread. Row E5 leaned on PAFT's main body only | Appendix C holds no data of the authors' own. Table 2 is Llama 3's interruption table (from Dubey et al.): 6 of the 419 interruptions listed were silent data corruption, 1.4%, and 7 were NCCL watchdog timeouts, 1.7%. Tables 3 and 4 are GPU error codes from the Fire-Flyer cluster (from An et al.): 12,970 errors in a year, 42.6% of them NVLink errors. Both are **second-hand here**. Two lines are asserted with no data: that many low-level silent errors go unreported (p. 18) and that noise at or above 0.01 "happen less" (p. 19). Appendix E.3: also synchronising the optimizer state helped only a little | E5 stands. PAFT's evidence for gradient-aggregation faults is simulation only. My hint that Appendix C might hold real cases was wrong. The 419 is my sum of the count column. One printed share does not fit it (148 faulty GPUs shown as 30.1%); I did not check the table against the Llama 3 paper |
| J4 | H8: the attention statistic leads the gradient norm by "at least about 500 updates", with growth starting "near 25.2B" tokens (from the caption of Fig. 3) | A second marker, not a replacement. Table 8, p. 20, gives the logged gradient norm of the failing run: 0.358 at 23.1B tokens, 3.871 at 29.4B, 375.5 at 33.6B. It first passes 10 at **29.9B**. Table 11, p. 23: on eight fixed documents the full-model gradient norm is already 36.7 at 27.3B tokens, and 2.3 to 2.5 once the attention backward is repaired | Two lead times, by two markers. To the start of growth near 25.2B: about 500 updates, as H8 said. To the logged norm passing 10: 6.8B tokens after the 23.1B checkpoint, about 1,600 updates (my division by 4,194,304 tokens per update). The fault is in the gradient at least 600 updates before the logged norm passes 10. The 23.1B and 27.3B points are single checkpoints, not onsets |
| J5 | H8 named the cause "BF16 fused attention, backward pass", from one 450M model | Four limits from the appendices. (i) The model is not a standard transformer: it is a "Full-AttnRes" design that reads earlier sublayer outputs through a learned softmax (Appendix H, p. 21). (ii) A plain 102M transformer did not show it: the backward differed from FP32 by under 1% at the base learning rate, and its largest query RMS stayed at 42.5 to 44.5 against above 200 in the failing run (Appendix K, p. 27). (iii) The backward error is not special to FlashAttention-3. On the same eight captured inputs the median relative error of the query gradient is 7.7 for FA3 and for FlashAttention-2, about 2.2 for FA3 with the repaired forward and for PyTorch's three fused attention backends, and 0.003 to 0.004 for PyTorch's unfused path (Table 13, p. 27). (iv) Of the five arms that ran the full 50B tokens, two passed a gradient norm of 10: unmodified FA3, and FA3 with key smoothing at 33.2B tokens, ending 0.24 nats above FP32 attention (Table 8) | The cause to name is narrower and wider than H8 said. Narrower: it needs the large-logit regime, which the plain model never reached. Wider: any fused BF16 attention shares the backward error at such inputs. A published fix (key smoothing) did not prevent the failure in this run. The arm with the repaired forward did not fail though it reached the same regime |
| J6 | E7: Wortsman's output logits go "very negative"; add log Z | Laguna, p. 9: the LM-head logits drifted "strongly positive" in M.1, with no penalty on the logits and a normalisation layer that does not subtract the mean | The sign is not fixed. Record the signed mean and log Z. Nothing on `main` records output logits (searched `evidence.py`, `capture.py`, `probe.py`) |
| J7 | The read table: the DeepSeek-V3 sentence on load imbalance runs onto p. 28, unread | p. 28: batch-wise balancing leaves two risks, imbalance "within certain sequences or small batches" and imbalance from a change of domain at inference. They answer the first with large batches and the second by serving extra copies of the busy experts, chosen from load statistics about every 10 minutes (p. 19) | Nothing in section 10 changes. It adds to H7: a released, healthy model is expected to route unevenly on one domain. Figure 9 plots per-domain expert load on a scale of 0 to 10 times an even load; I read the scale, not the values |
| J8 | The opening of section 10: two files are fresh downloads whose text differs "by under 0.5% of the words, which I did not trace" | Traced. TrainSDC: the new file's words are identical to my first extract; the difference was my own page markers. Ma et al.: the folder holds the arXiv preprint and the ACL 2025 version, both extracted in the earlier pass, and both unchanged since. The two versions differ from each other by about 2% of the words (339 and 68 tokens of about 14,000). The differing numbers I saw are page numbers; the list I printed was cut at 150 tokens | Settled. No re-read needed. I did not compare the two Ma versions sentence by sentence |

### New findings

| # | Finding | Source | Checked against the code |
| --- | --- | --- | --- |
| M1 | **A weight corrupted in memory is judged only through its effect on the gradient.** `weight_norm_trends` returns numbers "with no threshold and no incident". `update_rms` is read from the optimizer's state, so a change that did not come from an update is not in it. In a toy run, one weight multiplied by 1e8 took its tensor's norm from 5.3 to 21 million, and neither `detect_grad_spikes` nor `detect_update_spikes` fired when the activation was tanh. With ReLU both fired at the next step, from a factor of 100 up. A factor of 10 was missed in both, and did no harm in the toy | LLMFT, pp. 4-7: of 10,000 random single-bit flips per model, 88.6% to 96.7% were masked. Table 2 (p. 7): when the flipped value moved by under 10 times, 0.5% to 8.9% of runs failed; at 10 times or more, 30% to 36%; at 1e8 times or more, 39% to 79% | **Run on CPU against `main`** (appendix). Two rules tested, not all: the toy has no watched block, so `detect_activation_anomalies` was not exercised, and it is the rule most likely to see a fault in the forward path. The fault was put on the largest element of a 64 by 64 tensor |
| M2 | **A third team runs a replica check, and credits it with catching silent data corruption.** Laguna hashes the weights across data-parallel replicas "periodically" and asserts they are bit-identical. By its account the checks "primarily catch silent data corruption" from defective GPUs, in arithmetic logic that error-correcting memory does not cover. Its one example: a single machine with corrupt arithmetic, "surfacing as anomalous global gradient norms peaking at ∼10^6", which in earlier runs had caused "only subtle symptoms". "Beyond their original purpose" the checks also caught four software faults: data races, collective bugs, non-deterministic kernels in the optimizer, and a checkpoint loader that left rank 0 with 64-bit Adam betas while the other ranks had 32-bit ones | Laguna, pp. 8-9. Nothing is counted: no rate, no interval, no number of catches. One company. **This is in tension with W1**, which says the check sees only some places a fault can sit (Elsen's table in section 10). Laguna does not say where its faults sat. I leave the two side by side: one team's unquantified claim against another team's worked cases | Read: `replica_drift` is the same check with float fingerprints and a tolerance of 1e-6, where Laguna and Megatron compare hashes. **Run on CPU**: two replicas sharing one averaged gradient, one with 32-bit values of the betas. A weight bit differs from step 1. The largest fingerprint spread passes 1e-6 at the first 50-step check at a learning rate of 1e-3, and at the fourth at 1e-4; the largest spread in 20,000 steps was 7e-5. So the tolerance would see this case in the toy, later than a hash would. The script applies the tolerance to the spread itself; `detect_replica_drift` was not called |
| M3 | **Padding tokens can load one set of experts.** In Laguna M.1 about 5% of training tokens were padding. They were routed and counted in the balancing loss. Every padding token reaches the router with the same vector, so all of them go to the same experts at once, "potentially saturating that expert". XS.2 skips padding in routing and balancing | Laguna, p. 9 | Read: the router hook counts every token the router is called on, and the SDK has no handling of padding (searched). That is correct counting, since those tokens do go there. It is a cause the agent should name, not a bug. With 8 of 256 experts per token (XS.2's router; M.1's is not given), 5% padding puts about 2.6 times an even load on eight experts, far below the 0.35 share at which `moe_imbalance` fires. With 8 experts, one per token, and 30% padding, the share would be about 39%. Both are my arithmetic, not runs |
| M4 | **A reduction can be wrong on every rank with no hardware fault.** In Laguna M.1 the gradient entering the LM head was all-reduced across tensor-parallel ranks in BF16. As the logits grew, that reduction became "the dominant source of numerical error" and training became unstable. The fix was to do that one all-reduce in FP32 | Laguna, p. 9 | This is the second row of section 10's table of fault locations, reached by precision, not by a broken node. Every rank gets the same sum, so `replica_drift` cannot see it. It is also not a data-parallel reduction, so the check would not be looking there. Read, not run |
| M5 | **Three outcomes of a fault leave the loss near normal.** LLMFT sorts its failing runs into six kinds. Three end in NaN or no convergence. Three end in a worse model: a slightly changed loss, a higher final loss, and a late convergence. Its Figure 1 shows two loss curves that are hard to tell apart with accuracy at 53.83% and 49.86%. A higher final loss is tied to faults in the optimizer's state (p. 5) | LLMFT, pp. 2, 5. Fine-tuning BERT, GPT-2 and T5-small for a few hundred iterations in FP32 on TensorFlow; single-bit flips in parameters, gradients and optimizer state | A second source for H3: a fault in the optimizer's state can pass without a spike. He et al. was the first. Not run again |
| M6 | **A corrupted forward kernel can stall a run with a small gradient spike.** In Chen's comparison a second published attention method (Qiu and Yao's dynamic shift) fails in the forward pass. The run falls behind from 2.5B tokens, spikes once at 6.2B to 6.8B with a gradient norm of 6.4, then stalls 0.31 nats behind. By 12.6B tokens the attention output of about half the rows in the later layers is unrelated to exact attention. Swapping exact attention back in later does not help: the weights have adapted | Chen et al., Appendix G, pp. 20-21, Figure 4. One arm, stopped at 13.2B tokens | TensorQuill's probe computes attention with its own plain forward, so it would not see the kernel's wrong output. It would see the logits: the paper reports row maxima near 5,300 at 6.3B tokens, 5 to 100 times the other run's. `max_logit` is the same kind of number, a row maximum. Read, not run |
| M7 | **Routing can be frozen on purpose.** Motif 3 freezes the router weights and the expert-selection biases through all of its reinforcement-learning and distillation stages (p. 22). DeepSeek-V3 sets its bias update speed to zero for the last 500B of 14.8T tokens (p. 23) | Motif 3; DeepSeek-V3 | In those phases a change in expert load comes from the hidden states alone, and no balancing acts against it. Wording for the agent instructions and plan row S8. Not run |

**The six outcomes of a single bit flip in training (LLMFT, p. 5), and what TensorQuill has for each.** The right-hand column is my reading of the rules, not a run.

| Outcome | What the paper saw | A rule on `main` that could see it |
| --- | --- | --- |
| Immediate NaN | Gradients grow without bound in one backward pass | `non_finite` |
| Delayed NaN | The loss rises and wobbles for some iterations, then NaN | `non_finite` at the end; `grad_spikes` or `update_spike` before it if the norm rises |
| No convergence | An extreme value in the forward pass; the loss jumps and stays high | `grad_spikes` if the gradient follows. In M1's toy a huge weight behind a tanh left the gradient unmoved, though that run still converged |
| Slightly changed loss | A small early disturbance; the final accuracy is lower | None |
| Higher final loss | A fault in the optimizer's state moves the whole path | None (H3) |
| Late convergence | A loss spike that the optimizer recovers from; accuracy still lower | `grad_spikes` for the spike. Nothing says the model ended worse |

**What a norm-jump rule would have to clear.** In the recorded healthy GPU runs under `~/Downloads/runs` (five files, 8 or 13 weight checks each, 50 steps apart, 218 or 290 tensors), the largest change in any tensor's norm between two checks was a factor of 1.0007. In the toy, a factor-10 fault on one element moved the norm by 1.075 and a factor-100 fault by 4.1. The two sets of numbers are not on the same footing. The toy tensor has 4,096 elements; the real ones have up to millions, and a one-element fault moves a norm less as the tensor grows. So a factor-10 fault in a real tensor would sit well under 1.0007, and only very large faults would stand out. The 1.0007 is an observation from a few dozen intervals in short runs at one stage of training. It is not a threshold. Warm-up and training from scratch are not in it.

### Wrong or unsupported, in the three new papers

| Paper | Claim | Why it does not carry |
| --- | --- | --- |
| LLMFT | The detector reaches 97.5% precision and 97.6% F1 (p. 9, Table 3) | The training set is 1,000 injected faults, half masked and half failing; the test set is 1,000 more injected faults (p. 8). No fault-free step is ever scored, so the precision says nothing about false alarms on healthy training. The features include the epoch, batch, stage, layer and component of the injection (p. 6-7). The paper does not say how these are found in live training when no outlier points at a place; that gap is my reading |
| LLMFT | An outlier flag on the parameters is the key feature (p. 7) | By Table 2, 70% of T5's failing runs had a value that moved by under 10 times (791 of 1,137; my arithmetic), 23% of GPT-2's and 14% of BERT's. An outlier flag cannot see those |
| LLMFT | Between 3.32% and 11.37% of faults cause a failure (p. 4) | Table 2 gives BERT 3.08% by my sum (44 of 9,262 plus 263 of 738), under the stated range. GPT-2 gives 3.32% and T5 11.37%, as stated |
| LLMFT | "No delay is introduced" by the detector (p. 10) | The scan takes 63 to 71 ms against a training iteration of 182 to 255 ms, on three small models (BERT-base, GPT-2, T5-small). It reads every parameter each iteration. Nothing is shown at a larger size. The recall formula on p. 8 is printed with the wrong denominator |
| Laguna | Hash checks are "important to prevent silent and non-silent failures" (p. 8) | Four catches are named and none is counted. No interval, no cost |
| Laguna | XS.2 "proceeded without encountering further stability issues" (p. 9) | One run, the lab's own account |
| Mamoda2.5 | Sigmoid routing with a bias beats softmax with a balancing loss (p. 22) | "Preliminary experiments" in one configuration, on training loss, with no number given |
| DeepSeek-V3 | "We did not experience any irrecoverable loss spikes or perform any rollbacks" (pp. 1, 4) | The lab's own statement about one run. No curve of the loss or gradient norm is shown. Recoverable spikes are not ruled out by the wording |

Mamoda2.5 reports no training failure of any kind, and no expert-load number. For TensorQuill its use is row J2 and one sentence: it trains with fully sharded data parallelism in place of expert parallelism, which it says "avoids computational load imbalance" (p. 13).

### What the rest of the reading confirms

- **Motif 3, pp. 1-5, 14-15, 20-32.** Nothing in section 10's rows E1, E2, H5 or H6 changes. Two values to add to H5's table: the fine-tuning stage keeps a penalty on FFN output magnitude with a threshold of 128, and 1,024 for the last layer (p. 20). The work balancing in H6 (iii) moves whole packed sequences between replicas every step (p. 14), so which rank is slow can change from step to step.
- **DeepSeek-V3, the other 35 pages.** No token is dropped in training or serving (p. 10). The sequence-wise balance loss has a weight of 0.0001, "just to avoid extreme imbalance within any single sequence" (p. 23). The rest is benchmarks, serving, hardware advice and post-training.
- **Wang et al., appendices.** Definitions of the measures and more figures. One fact for reference values: the same architecture trained with AdamW has hidden-state norms of 25 to 35, and 12 to 14 with Muon (Figure 11). A healthy scale does not carry from one optimizer to another.
- **Wortsman et al., pp. 22-26.** Figures. Figure G.19: Adam's update RMS falls when the gradient RMS nears the optimizer's epsilon of 1e-8. That is a second, harmless way for one tensor's update to shrink. A rule for "one parameter's update collapsed" (row H3) would have to tell it from a fault.
- **FALCON, p. 17.** Formulas for the change-point test and for communication volumes. Nothing new.
- **Chen et al., Appendices A to F.** Proofs, the kernel's arithmetic and the test suite. The parts that bear on TensorQuill are in rows J4, J5 and M6.

### What this changes in the findings and proposals

- **Plan row K7 and row E1.** Nothing new to change. E1 already asks to take back "without a source". Laguna adds a second reported collapse, of unstated kind, that ended in divergence (J1). Hidden collapse still has no reported case, and the row still needs expert outputs.
- **W1 and proposal 1.** Three teams now run a replica check (Megatron, Adept, Poolside). Poolside says it mainly catches silent data corruption and also caught four software faults (M2). W1's limits stand on Elsen's cases; Poolside's claim is unquantified and does not say where its faults sat. What M2 does add: the check earns its place on software faults too. Add M4 as an example of the second row of the fault-location table.
- **Plan row S8 and the agent instructions.** Four causes to name. Padding routed to one set of experts (M3). New tokens with untrained embeddings: gradient spikes and dead experts (J1, Laguna p. 17). Fused BF16 attention in a large-logit regime, with J5's limits. A reduction done in low precision (M4).
- **Plan row K4.** Record the signed mean of the output logits and log Z (J6).
- **Plan row K1 and row H3.** A second source says a fault in the optimizer's state can end in a worse model with no spike (M5).
- **Weight norms.** They are recorded every 50 steps and judged by nothing (M1). A rule on a jump between two checks could be judged from what is stored. The healthy figure above is too thin to set it. This is a gap in a built record, like H1 to H3, not an eleventh proposal.
- **The list of ten stays ten.**

### What was read in this pass

| Paper | Kind of evidence | Read now | Still not read |
| --- | --- | --- | --- |
| Yu et al., "Exploring and Mitigating Failure Behavior of Large Language Model Training Workloads in HPC Systems", SC '25 (LLMFI and LLMFT) | Measured: over 300,000 injected bit flips into three small models being fine-tuned | All 15 pages; Tables 2, 3 and 5 re-read in layout | Figures 4, 6, 7, 8 and 9 as images |
| Poolside, "Laguna M.1/XS.2 Technical Report", arXiv 2605.27605 | A lab's report on two training runs; incidents told in a sentence or two each, no rates | All 37 pages | Figures as images |
| Mamoda Team, "Mamoda2.5", arXiv 2605.02641 | A lab's report on a video model; benchmark tables and three ablations | pp. 1-28 | pp. 29-33 (references), p. 34 (one benchmark table), pp. 35-38 (picture pages): scanned only |
| DeepSeek-V3 | As in section 10 | The 35 pages section 10 skipped. With section 10, all 53 | pp. 37-43 (references) scanned only; pp. 49-53 are heatmaps, read through their captions |
| Motif 3 | As in section 10 | pp. 1-5, 14-15, 20-24 in full; pp. 25, 27-32 through the filter | pp. 27-30 (references) scanned only |
| Tang et al. (PAFT) | As in section 10 | pp. 12-26 through the filter; Tables 2 to 4 re-read in layout | The proofs' equations (pp. 19-24); Figures 11 to 18 as images. Table 1 still would not extract. pp. 10-11 are the impact statement and references, scanned only |
| Chen et al. | As in section 10 | pp. 13-28 through the filter; Tables 8, 11 and 13 re-read in layout | The equations of Appendices A to C; Tables 2, 6, 7, 10, 12, 14 and 15; figures as images. pp. 11-12 are references, scanned only |
| Wang, Hayou, Nalisnick | As in section 10 | pp. 14-31 through the filter | Equations; Figures 11 to 29 as images. pp. 12-13 are references, scanned only |
| Wortsman et al. | As in section 10 | pp. 22-26 (captions) | The figures as images; pp. 10-13 (references) |
| FALCON | As in section 10 | p. 17 | Nothing |
| TrainSDC; Ma et al. | Read in section 9 | Compared word for word with my first extracts (J8) | Not re-read; no need |

**Public material (for section 5).** LLMFT gives two Zenodo records for its injection tool and its detector (DOIs 10.5281/zenodo.15279193 and 15279226, p. 14); both are TensorFlow 2.8 code for small fine-tuning runs. Laguna XS.2's weights are on Hugging Face under Apache 2.0 (p. 1); a released model, not a failure set. Chen et al. say their kernel errors "can be reproduced from the captures alone" (p. 18); I did not find where the captures are published. None of this was downloaded or opened.

**Not in the folder.** Dai et al., the DeepSeekMoE paper (arXiv 2401.06066). Section 10 lists it; the three other papers on that list are the ones read here. (It was added later and is read in section 12.)

**Not done in this pass.** No GPU run. No run of TensorQuill on a failure these papers describe. No reading of figure images. No edit to the plans, the README or the agent instructions: plan row K7 still says "without a source", which E1 asks to take back. Nothing is committed.

## 12. DeepSeekMoE, and a web search for a source for K7 (Nov 18, 2026, sixth pass)

Two things happened after section 11. The DeepSeekMoE paper was added to the folder (Dai et al., arXiv 2401.06066v1, 33 pages). And the user asked for a web search for a paper that shows hidden expert collapse, which is what plan row K7 lacks. The folder then held 50 files and 49 distinct papers and posts; with the paper added afterwards, 51 and 50.

**The answer in two lines.** No paper found shows hidden expert collapse as a failure in a training run. One paper comes close enough to be worth downloading: it measures how alike the experts' outputs are, through training, in runs whose load looks balanced.

**How this was done.** DeepSeekMoE: I read all 33 pages. pp. 1-11 as plain text, with Tables 1 and 2 one cell per line; pp. 12-24 and 31-33 with the page layout kept and no filter, which covers Tables 3 to 10. pp. 24-30 are the reference list, scanned only. Figures 1 and 3 to 7 are charts, and I could not read their values. Page numbers are PDF pages, found by extracting each page on its own: the charts in this PDF put false page breaks into a whole-file extract. I checked five other papers read with the same page printer (Laguna, Mamoda2.5, LLMFT, DeepSeek-V3, Motif 3); their page breaks match their page counts. I did not run that check on the papers of sections 7 to 10. The web search: four searches, with no detail about this project in them. Five papers came up. **I did not read those five** (one of them, Omi, Sen, Farhadi, was added afterwards and is read in full at the end of this section, which corrects what is said of it here). A fetch helper read their arXiv HTML pages and answered eight fixed questions with short quotes; I saw only its report. Every statement about them below is at that distance, and each should be re-checked when the PDF is in the folder. Nothing was downloaded.

### What K7 needs, and what each found paper gives

The test: experts still receive tokens, while their outputs shrink or become copies of each other. A paper passes fully if it has (a) an occurrence in a training run, (b) the statistic in a failing run and a healthy one, (c) a cost, (d) a threshold with a lead time or a false-alarm rate.

| Paper | What it measures | (a) Occurrence | (b) Signal | (c) Cost | (d) Threshold | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Omi, Sen, Farhadi, "Load Balancing Mixture of Experts with Similarity Preserving Routers", arXiv 2506.14038 | Mean pairwise cosine similarity of all experts' outputs on the same token ("PES"), on 4M sampled tokens, per layer, through training | In part. Two runs with the standard balancing loss (627M and 3.14B total parameters, 32 experts, top-4). Similarity is higher than with their method. No run is said to fail | Yes. The lowest PES of any layer: 0.0255 against 0.0044, and 0.0241 against 0.0028. The other layers are higher; their values are in a figure the helper did not report. Curves over training | Perplexity 14.086 against 13.685, and 8.517 against 8.304. The link is a correlation: the two runs also differ in the loss used | None | **Closest. Now in the folder and read: see "Omi, Sen, Farhadi, read in full" below, which corrects this row** |
| Do, Le, Tran, "SimSMoE", arXiv 2406.15883 | CKA between the outputs of two experts, on tokens routed to both | Weak. Figures only; the helper found no numbers for the baseline | No numbers | Bits per character 1.11 against 1.08 and similar, for the method, not for the collapse | A CKA level of 0.5 (range 0.3 to 0.7), used to switch a training loss on. Not tested as a detector | Second. A candidate statistic that, as I understand the helper's description, needs only the experts a token already used |
| Chi et al., "On the Representation Collapse of Sparse Mixture of Experts", arXiv 2204.09179 | The hidden states that go into the router, not the experts' outputs | Yes, a curve over 125K steps on a 12-layer encoder with 32 experts | No numbers in the text | Not measured for the collapse | None | A different failure from K7. Worth having because later papers cite it for "collapse" |
| Guo, Lu et al., "Advancing Expert Specialization for Better MoE", arXiv 2505.22323 | Expert overlap and routing variance, at the end of a short fine-tune (about 550 steps, LoRA) | End of run only | Partial | Benchmark scores for their method | None | Not a training-run source |
| Li et al., "Expert Divergence Learning for MoE-based Language Models", arXiv 2603.00054 | How much the loss rises when a layer's routing is shuffled; routing per data domain | At checkpoints | Figures only | Benchmark averages 35.59 against 36.34 to 36.65 | None | Not about expert outputs |

Three things follow for K7.

- **K7 still has no reported failure.** The closest paper shows experts that are more alike under one balancing loss than under another, alongside a perplexity gap of about 0.2 to 0.4. No run in it is said to collapse or fail. The numbers I have are each run's least-similar layer (0.0255 at most); how alike the worst layer gets is in a figure I have not seen.
- **It does show that a load monitor cannot see this** (corrected by P1 below: the usage number is for one run). In the same two runs the paper's own usage measure is 1.0, which is perfect, while similarity is six to nine times higher than in the other arm. That measure is per sequence, not the per-expert share TensorQuill records. So the statement "load looks healthy while experts overlap" is supported, at one remove.
- **The statistic is expensive.** PES runs every expert on every sampled token. The paper puts that at a multiplier of 3.6 to 4.9 on floating-point work; the helper did not say against what base (the base is a normal forward pass: P2 below). TensorQuill captures router scores, not expert outputs, so K7 needs a new capture either way. SimSMoE's version compares only experts that a token was already routed to. If that is right, it needs no extra expert passes, which would make it the cheaper design to test first. That is my inference from the helper's report, not something I read.

**No threshold comes out of any of the five.** If K7 is built, its numbers have to come from TensorQuill's own seeded runs.

### DeepSeekMoE: what the 33 pages say

Before reading it I wrote that this paper "is not what K7 needs". Having read it, that holds, with one useful addition (N2).

| # | What the paper says | Kind of evidence | Effect on TensorQuill |
| --- | --- | --- | --- |
| N1 | Routing collapse is named as a risk, with a citation to Shazeer et al. 2017: "the model always selects only a few experts, preventing other experts from sufficient training" (p. 7). No collapse is reported in any of its runs | A definition and a citation. No occurrence, no load number anywhere in the paper | It supports the wording of `moe_imbalance` and `starved_experts`. It supports no threshold. `OVERLOAD_THRESHOLD = 0.35` and `STARVE_RATIO = 0.1` stay TensorQuill's own |
| N2 | Redundancy between experts is tested on a finished model: for each token, mask the experts with the highest routing probability, route to the next ones, and measure the loss. The model that loses more is said to be less redundant (p. 13, Figure 4) | One comparison of two trained models, DeepSeekMoE 2B and GShard×1.5 (2.8B of expert parameters). The values are in a chart I could not read. Done at evaluation, not during training | Not a K7 source: nothing is tracked through training and nothing fails. It is a test for redundancy that needs no expert outputs, only a change to the routing at evaluation. TensorQuill does not run evaluations, so this is a note for later, not a proposal |
| N3 | Disabling the one shared expert and activating one more routed expert raises the loss from 1.808 to 2.414 (p. 13) | Measured, one model | A shared expert is always on and never passes through the router. TensorQuill's router hook cannot see it: it has no load to count. If a shared expert degraded, no MoE kind would say so. Not a defect; a limit to state |
| N4 | The first layer is kept dense in the 16B and 145B models "since we observe that the load balance status converges especially slower for the first layer" (p. 15; the 145B model does the same, p. 21) | The lab's observation. No number, no curve | A second report that early layers balance late (Motif 3, in E1, reported routing collapse in early layers). For routers with top-k of 3 or more, `moe_imbalance` compares a router with its own recent history, so a slow start should not fire it; smaller top-k uses the fixed share. I ran neither case |
| N5 | The balance loss weight was 0.01 for the 2B model, 0.001 for 16B, and 0.003 plus a device-level 0.05 for 145B. For 16B, a higher weight "will compromise the model performance" (pp. 9, 15, 22) | Settings, and one claim with no number | The weight varies by a factor of ten between sizes of one design. A healthy load pattern is not one fixed shape. This agrees with DeepSeek-V3 moving to a bias with a very small loss (section 11) |
| N6 | No tokens are dropped in the 2B and 16B runs (pp. 8, 15). The 145B run spreads experts over 4 devices and the paper does not say (p. 22) | Stated | Nothing. Token dropping is a GShard feature TensorQuill does not model |
| N7 | Routing is softmax over a dot product with an expert centroid, then top-k; the gate weight is the softmax score (p. 4, equations 3 to 5). No bias | Design | For this design the hooked scores are the real routing, so finding F1 does not apply. F1 applies to DeepSeek-V3, which added the bias later |

**Wrong or unsupported in the paper.**

| Claim | Why it does not carry |
| --- | --- |
| "DeepSeekMoE exhibits lower redundancy among routed experts" (p. 13) | It rests on one chart of two models that differ in expert size and count (16 experts against 63). A model with more, smaller experts losing more when its top choices are masked may reflect the masking ratio and not redundancy. No second measure is given |
| "nearly approaches the performance of its dense counterpart ... the upper bound of MoE models" (pp. 1, 11) | One pair of runs; no repeats are reported. Loss 1.808 against 1.806. On RACE the dense model is ahead by 2.3 and 1.3 points (Table 2) |
| Comparable with DeepSeek 67B "using only 28.5% (maybe even 18.2%) of computations" (p. 1) | The 145B model was trained for 13,000 steps on 245B tokens, which the paper itself calls preliminary (pp. 21-22). In Table 6 (p. 23) it trails the dense model on MMLU by 5.7 points and on HumanEval by 4.3 |

**What this changes in sections 1 to 11.** Nothing is overturned. Three lines move:

- The "Not in the folder" lines in sections 10 and 11 no longer hold for Dai et al.; both are marked.
- E1 and J1 stand: K7 has a design source (Motif 3) and no reported case. This section adds one paper, added to the folder afterwards, that measures the underlying quantity.
- Row E6 named "Dai et al." by mistake for another paper. The real Dai et al. is now read, and it has nothing on hidden-state geometry.

**Read table.**

| Source | Read | Not read |
| --- | --- | --- |
| Dai et al., DeepSeekMoE, arXiv 2401.06066v1 | All 33 pages. pp. 1-11 as plain text; pp. 12-24 and 31-33 in layout, unfiltered (Tables 3 to 10) | Values in Figures 1 and 3 to 7 (charts). Tables 1 and 2 were read one cell per line, not in layout. pp. 24-30, the reference list, scanned only |
| The five papers found on the web (Omi, Sen, Farhadi since read in full: see the last read table of this section) | Nothing by me. A helper read the arXiv HTML and reported with quotes | Everything, at first hand. The helper marked parts it had only as a summary: Chi et al. section 6 and appendices; Guo et al. Appendix E; Omi et al. Table 5; Li et al. Appendix H |

**Worth downloading, in order.** arXiv 2506.14038 (Omi, Sen, Farhadi), since added. Then 2406.15883 (SimSMoE). Then 2204.09179 (Chi et al.), for the definition. The other two add little.

### Omi, Sen, Farhadi, read in full (added later on Nov 18, 2026)

The user added the paper: `2506.14038v2.pdf`, 15 pages, version 2 of Oct 11, 2025. The helper had read version 1. I read all 15 pages with the page layout kept and no filter, one page at a time. p. 10 holds the limitations and the conclusion; the reference list runs from there to p. 13. Figures 1 to 5 are charts; I read their captions and not their values.

**The verdict on K7 does not change.** No run in the paper fails with its experts still in use. What the paper measures is a difference between two balancing losses, not a failure.

**Four corrections to what this section said above, from the helper's report.**

| # | Said above | What the page says | Effect |
| --- | --- | --- | --- |
| P1 | Usage is 1.0 "in the same two runs" | Table 3 (p. 6) gives usage for the large model only: 1.000 with the standard loss, 0.991 with theirs. For the medium model the paper says only that all experts are used on the validation set (Figure 2, p. 7) | "Load looks fine while similarity is higher" rests on one run for the usage number, and on a chart for the other. This may be a difference between versions and not the helper's error: its table numbers do not match version 2's, and I did not check version 1 |
| P2 | The base of the "3.6 to 4.9 times" cost was not known | It is one forward pass with every expert run on every token, against a normal pass (p. 5). The ratio of total to active parameters without embeddings gives the same two numbers: 550M/153M = 3.6 and 2,986M/607M = 4.9 (my arithmetic from Table 1) | The cost is per measured token, at checkpoints, offline. The paper does not compute it during training steps |
| P3 | The link between similarity and perplexity came from one pair of runs per size | For the medium model there are five runs of each: perplexity 14.051 with a standard deviation of 0.026 against 13.691 with 0.039 (p. 7). The large model has one run of each | The perplexity gap is solid at the medium size. It is still a gap between two losses, so similarity is not shown to be its cause |
| P4 | The reported number is each run's least-similar layer, the rest unknown | Confirmed, with the reason: the minimum over layers is used for "filtering out spikes in a single individual layer (common with LBL)"; they saw no dips by layer, "primarily jumps" (p. 8) | **One layer far above the others in expert similarity is common with the standard loss.** A K7 rule that compares layers with each other would fire on it. How far above is in Figure 3(a), which I could not read. The paper gives no count and does not say whether those runs were otherwise healthy |

**What else the paper has that bears on TensorQuill.**

| # | What the paper says | Kind of evidence | Effect on TensorQuill |
| --- | --- | --- | --- |
| P5 | With no balancing loss, the number of distinct experts used on the validation set falls during training (Figure 2, p. 7). "The no-loss baseline was truncated early" | Two runs, shown as charts. No numbers in the text; the runs were stopped, so no cost is given | A routing collapse seen in training, of the visible kind. `starved_experts` is the matching kind. Not K7 |
| P6 | With a routing bias alone (the method of Wang et al.) and no auxiliary loss, a sequence uses 0.505 of the experts (softmax) or 0.381 (sigmoid), and perplexity is as good or better: 13.708 and 13.618 against 14.086 for the standard loss. Adding the standard loss to the bias brings usage back to 0.997 and perplexity to 14.154 and 14.015 (Table 8, p. 14; Table 4, p. 8). All experts are still used over the validation set. The paper notes DeepSeek-V3 always pairs the bias with a loss (p. 9) | Measured, medium model, one run each | **A healthy run can be very uneven inside a sequence.** I read the rule in `evidence.py`: for a top-4 router, "overloaded" needs a drop against the router's own history, so steady unevenness does not fire it. The dead-expert path is different: it fires when a quarter or more of the experts get no token in a step. A step is a whole batch, so that needs a step of very few sequences. With such steps, a run like this one could raise `moe_imbalance` and be fine. Read, not run. Agrees with J7 (DeepSeek-V3, p. 28). Second-hand on the same page: Qiu et al. report the bias method is sensitive to the per-device batch size (p. 9) |
| P7 | That bias is added to the scores for choosing experts and "is not used in the weighting of the experts" (p. 14) | The paper's description of Wang et al.'s method, and its own implementation of it | One more description of the design behind finding F1. Second-hand for Wang et al.; first-hand for this paper's runs |
| P8 | The two arms differ in the router's weights, not in load: the Gram matrix of the router is 0.0311 from the identity with the standard loss and 2.1e-8 with theirs (Table 3); per layer in Tables 9 and 10 (p. 15), up to 0.1385 in the last layer of the large model | Measured | Nothing to act on. It shows the similarity measure tracks a design choice |
| P9 | All parameters are in bfloat16; training uses DDP on 8 devices (p. 4) | Setup | Nothing |

**Wrong or unsupported in the paper.**

| Claim | Why it does not carry |
| --- | --- |
| The standard loss "exacerbates redundancy during the majority of training" (Figure 3 caption, p. 7) | It rests on the sign of a rate of change read from a chart of one large run. No numbers, no second seed at that size |
| "Redundancy occurs when many distinct experts see similar tokens" (p. 7) | Stated as the mechanism. The paper measures neither which tokens the experts see nor how similar they are |
| "over 13 standard deviations lower ... very statistically significant" (p. 7) | The difference is 0.360 and the larger of the two deviations is 0.039, so the gap is real. The "13" divides by one arm's deviation from five runs; it is not a test |
| "36% faster convergence" (pp. 1, 6) | Read off the loss curves: about 50B tokens against 78.6B, and 12.7B against 19.9B. No tolerance for "the same loss" is given |
| Table 6 (p. 8) gives perplexity 8.317 and 8.536 with nothing dropped; Table 4 gives 8.304 and 8.517 for the same models | The paper does not explain the difference |
| Their similarity measure "correlates" with the masking test of Dai et al. (Figure 4, pp. 7, 9) | One chart, one model size, two points of comparison |

**What this means for K7, in three lines.**

- A source for the statistic exists: mean pairwise cosine similarity of expert outputs, per layer, on a fixed sample of tokens, at checkpoints.
- No source for a failure, a threshold, a lead time or a false-alarm rate exists in the folder. P4 adds a known source of false alarms.
- The statistic needs every expert run on the sampled tokens. That is an offline check at checkpoints, not a hook on a training step.

**Read table for this part.**

| Source | Read | Not read |
| --- | --- | --- |
| Omi, Sen, Farhadi, arXiv 2506.14038v2 | All 15 pages in layout, unfiltered; Tables 1 to 10 and the code in Figure 6 | Values in Figures 1 to 5. The references (pp. 10-13) were read through. Version 1 was not compared with version 2 |

The four other papers from the web search are still not in the folder and still not read by me.

**Not done in this pass.** No download. No run. No edit to the plans: K7 still says "without a source". Section 12 is not committed.

---

## Appendix: the script behind proposal 2

Run on Nov 17, 2026 from a scratch folder with the repository's `.venv`, against the code as it was that day. It writes `tensorquill-out/` into the folder it is run from, so do not run it from the repository root. Each step routes 10 tokens. Three evaluation batches run inside `session.phase("eval")` between steps 1 and 2. It printed `[0, 3, 3, 4]` for steps 0, 1 and 3, and `[0, 9, 15, 16]` for step 2.

```python
import json, glob, torch, torch.nn as nn
import tensorquill

class Router(nn.Module):
    def __init__(self, d, n): super().__init__(); self.lin = nn.Linear(d, n)
    def forward(self, x): return self.lin(x)          # float logits, last dim = n_experts
class Block(nn.Module):
    def __init__(self, d, n):
        super().__init__(); self.router = Router(d, n); self.experts = nn.ModuleList(nn.Linear(d, d) for _ in range(n))
    def forward(self, x):
        idx = self.router(x).argmax(-1)
        out = torch.zeros_like(x)
        for e, lin in enumerate(self.experts): out = out + lin(x) * (idx == e).unsqueeze(-1)
        return out
class Net(nn.Module):
    def __init__(self): super().__init__(); self.blocks = nn.ModuleList([Block(8, 4)])
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x

torch.manual_seed(0)
model = Net()
session = tensorquill.init(model, routers={"blocks.0.router": 4}, run_id="evalcheck", every=1)
opt = torch.optim.SGD(model.parameters(), lr=0.0)   # lr 0: routing identical every step
x = torch.randn(2, 5, 8)                              # 10 tokens per training step
def step():
    loss = model(x).mean(); opt.zero_grad(); loss.backward(); opt.step()
step(); step()
with session.phase("eval"):
    model.eval()
    with torch.no_grad():
        for _ in range(3): model(torch.randn(2, 5, 8))   # 30 eval tokens
    model.train()
step(); step()
session.close()
for f in glob.glob("tensorquill-out/*.jsonl"):
    for line in open(f):
        r = json.loads(line); print(r.get("step"), r.get("moe"))
```


## Appendix: the script behind finding F1

Run on Nov 18, 2026 from a scratch folder with the repository's `.venv`, at commit `93e7d77`. It writes `tensorquill-out/` into the folder it is run from. The router module returns raw scores that favour expert 0. The block adds a balancing bias and then picks the top expert, as loss-free balancing does. Each of 80 steps routes 64 tokens. It printed:

```text
real routing, all steps : [1703, 1406, 1003, 1008]
TensorQuill's counts    : [5120, 0, 0, 0]
incident kinds          : ['moe_imbalance']
   {'kind': 'moe_imbalance', 'class': 'anomaly', 'step': 0, 'layer': 1, 'router': 'blocks.0.router', 'diversity': 0.25, 'overloaded': [1], 'dead': [2, 3, 4]}
```

Run again with the router declared as `tensorquill.Router(N, top_k=1, indices=lambda out: (out + bias).argmax(-1))`, where `bias` is the block's balancing bias, TensorQuill's counts were `[1703, 1406, 1003, 1008]`, the same as the real routing. One `moe_imbalance` incident still came, at step 5, with no dead expert: on one 64-token step a single expert went over the overload line. That is the toy's small step, not the reading.

```python
import glob, json, torch, torch.nn as nn
import tensorquill

N = 4
class Router(nn.Module):                      # the hooked module: returns raw scores
    def __init__(self, d, n):
        super().__init__(); self.lin = nn.Linear(d, n)
        with torch.no_grad(): self.lin.bias[0] = 6.0      # raw scores favour expert 0
    def forward(self, x): return self.lin(x)

class Block(nn.Module):                       # routing is decided on score + bias, as in loss-free balancing
    def __init__(self, d, n):
        super().__init__(); self.router = Router(d, n)
        self.experts = nn.ModuleList(nn.Linear(d, d) for _ in range(n))
        self.register_buffer("balance_bias", torch.tensor([-6.0, 0.0, 0.0, 0.0]))
        self.real = torch.zeros(n, dtype=torch.long)
    def forward(self, x):
        s = self.router(x)
        idx = (s + self.balance_bias).argmax(-1)          # the real routing
        if self.training: self.real += torch.bincount(idx.reshape(-1), minlength=N)
        out = torch.zeros_like(x)
        for e, lin in enumerate(self.experts): out = out + lin(x) * (idx == e).unsqueeze(-1)
        return out

class Net(nn.Module):
    def __init__(self): super().__init__(); self.blocks = nn.ModuleList([Block(8, N)])
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x

torch.manual_seed(0)
model = Net()
session = tensorquill.init(model, routers={"blocks.0.router": N}, run_id="biascheck", every=1)
opt = torch.optim.SGD(model.parameters(), lr=0.0)
STEPS = 80
for _ in range(STEPS):
    x = torch.randn(4, 16, 8)                             # 64 tokens a step
    loss = model(x).mean(); opt.zero_grad(); loss.backward(); opt.step()
session.close()

seen = torch.zeros(N, dtype=torch.long)
files = glob.glob("tensorquill-out/**/*.jsonl", recursive=True)
for f in files:
    for line in open(f):
        r = json.loads(line); m = r.get("moe")
        if m:
            for v in m.values(): seen += torch.tensor(v[:N])
print("real routing, all steps :", model.blocks[0].real.tolist())
print("TensorQuill's counts    :", seen.tolist())
ev = tensorquill.build_evidence(files[0])
print("incident kinds          :", sorted({i.get("kind") or i.get("class") for i in ev.get("incidents", [])}))
for i in ev.get("incidents", [])[:2]: print("  ", {k: i[k] for k in list(i)[:8]})
```

## Appendix: the script behind finding F2

Run the same way on the same day. One float32 matrix of a million weights is copied, one bit of one weight is flipped in the copy, and the two are given to the replica fingerprint functions as if they were two ranks. The spread is relative to the norm, as `replica_drift` judges it against its `1e-6` tolerance. It printed:

```text
lowest mantissa bit of one weight          spread(norm,sum,proj)=['0.0e+00', '7.3e-13', '7.2e-13']  integer checksum differs: True
bit 12 of the mantissa of one weight       spread(norm,sum,proj)=['1.9e-14', '3.0e-09', '3.0e-09']  integer checksum differs: True
top mantissa bit of one weight             spread(norm,sum,proj)=['2.4e-11', '3.1e-06', '3.0e-06']  integer checksum differs: True
sign bit of one weight                     spread(norm,sum,proj)=['0.0e+00', '1.3e-05', '1.3e-05']  integer checksum differs: True
one exponent bit of one weight             spread(norm,sum,proj)=['2.1e-11', '6.5e-06', '6.5e-06']  integer checksum differs: True
```

```python
import torch
from tensorquill.distributed import parameter_fingerprints, spread_from_fingerprints
torch.manual_seed(0)
w = torch.randn(1000, 1000) * 0.02            # one float32 weight matrix, 1e6 elements, init-sized values
def flipped(t, index, bit):
    u = t.clone(); v = u.view(-1).view(torch.int32); v[index] ^= (1 << bit); return u
def intsum(t):                                # an exact, order-independent integer checksum of the raw bits
    return int(t.view(-1).view(torch.int32).to(torch.int64).sum().item())
cases = {"lowest mantissa bit of one weight": flipped(w, 123456, 0),
         "bit 12 of the mantissa of one weight": flipped(w, 123456, 12),
         "top mantissa bit of one weight": flipped(w, 123456, 22),
         "sign bit of one weight": flipped(w, 123456, 31),
         "one exponent bit of one weight": flipped(w, 123456, 27)}
fa = parameter_fingerprints([w])
for name, b in cases.items():
    spread = spread_from_fingerprints(torch.stack([fa, parameter_fingerprints([b])]))
    print(f"{name:42s} spread(norm,sum,proj)={[f'{x:.1e}' for x in spread.flatten().tolist()]}  integer checksum differs: {intsum(w) != intsum(b)}")
```

---

## Appendix: the two CPU checks behind section 9 (G5 and D5)

One throwaway script, run on Nov 18, 2026 with the repository's `.venv` on CPU, from a scratch folder outside the repository. It imports only `torch`. It does not use TensorQuill.

- **Part 1 (G5).** A three-layer MLP on 4,096 random-label samples, batch 32, AdamW at 1e-3 with β1 = 0.9 and no weight decay. Each step it measures the RMS of the change in all weights, divided by the learning rate. The paper's steady value is 0.229.
- **Part 2 (D5).** A two-layer block added to a working path, trained in float32. Once with both weight matrices exactly zero, once with a random start of standard deviation 1e-4.

```python
# (1) RMS of Adam's update / lr on a small MLP, with the loss and gradient norm printed beside it.
# (2) A two-layer block with both weight matrices exactly zero: do its weights ever move, in float32?
import torch, math
torch.manual_seed(0)
b1 = 0.9
net = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
lr = 1e-3
opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(b1, 0.999), weight_decay=0.0)
X = torch.randn(4096, 64); Y = torch.randint(0, 10, (4096,))
rows = {}
n = sum(p.numel() for p in net.parameters())
for step in range(3000):
    idx = torch.randint(0, 4096, (32,))
    loss = torch.nn.functional.cross_entropy(net(X[idx]), Y[idx])
    opt.zero_grad(); loss.backward()
    g = math.sqrt(sum((p.grad ** 2).sum().item() for p in net.parameters()))
    before = [p.detach().clone() for p in net.parameters()]
    opt.step()
    sq = sum(((p.detach() - q) ** 2).sum().item() for p, q in zip(net.parameters(), before))
    rows[step] = (math.sqrt(sq / n) / lr, loss.item(), g)
def avg(a, b, k): return sum(rows[s][k] for s in range(a, b)) / (b - a)
print("predicted RMS/lr %.3f" % math.sqrt((1 - b1) / (1 + b1)))
for a, b in ((0, 1), (1, 10), (100, 200), (1000, 1100), (2000, 2100), (2900, 3000)):
    print(f"steps {a}-{b}: RMS/lr {avg(a,b,0):.3f}  loss {avg(a,b,1):.3f}  grad norm {avg(a,b,2):.3f}")

print("--- zero-initialised two-layer block, float32, added to a working path")
torch.manual_seed(0)
class Block(torch.nn.Module):
    def __init__(s, std):
        super().__init__()
        s.base = torch.nn.Linear(32, 32)
        s.fc1 = torch.nn.Linear(32, 64); s.fc2 = torch.nn.Linear(64, 32)
        for l in (s.fc1, s.fc2):
            torch.nn.init.zeros_(l.bias)
            torch.nn.init.zeros_(l.weight) if std == 0 else torch.nn.init.normal_(l.weight, std=std)
    def forward(s, x): return s.base(x) + s.fc2(torch.nn.functional.gelu(s.fc1(x)))
for std in (0.0, 1e-4):
    m = Block(std); o = torch.optim.AdamW([*m.fc1.parameters(), *m.fc2.parameters()], lr=1e-3)
    x = torch.randn(256, 32); y = torch.randn(256, 32)
    first = None
    for step in range(300):
        l = ((m(x) - y) ** 2).mean(); o.zero_grad(); l.backward()
        if step == 0: first = (m.fc1.weight.grad.abs().max().item(), m.fc2.weight.grad.abs().max().item(), m.fc2.bias.grad.abs().max().item())
        o.step()
    print(f"init std {std}: step-0 max |grad| fc1.weight {first[0]:.2e}, fc2.weight {first[1]:.2e}, fc2.bias {first[2]:.2e}; after 300 steps |fc1.weight| {m.fc1.weight.norm().item():.4f}, |fc2.weight| {m.fc2.weight.norm().item():.4f}, |fc2.bias| {m.fc2.bias.norm().item():.4f}")
```

Output:

```text
predicted RMS/lr 0.229
steps 0-1: RMS/lr 0.985  loss 2.309  grad norm 0.744
steps 1-10: RMS/lr 0.447  loss 2.310  grad norm 0.747
steps 100-200: RMS/lr 0.268  loss 2.224  grad norm 0.921
steps 1000-1100: RMS/lr 0.303  loss 0.946  grad norm 2.991
steps 2000-2100: RMS/lr 0.172  loss 0.149  grad norm 1.822
steps 2900-3000: RMS/lr 0.055  loss 0.021  grad norm 0.381
--- zero-initialised two-layer block, float32, added to a working path
init std 0.0: step-0 max |grad| fc1.weight 0.00e+00, fc2.weight 0.00e+00, fc2.bias 1.95e-02; after 300 steps |fc1.weight| 0.0000, |fc2.weight| 0.0000, |fc2.bias| 0.6800
init std 0.0001: step-0 max |grad| fc1.weight 7.93e-06, fc2.weight 9.97e-06, fc2.bias 1.59e-02; after 300 steps |fc1.weight| 8.6112, |fc2.weight| 7.9492, |fc2.bias| 0.5684
```

What it shows, and what it does not. Part 1: the update size is near the predicted value while the loss is high, about four times that on the first step, and a quarter of it once the toy has fitted its data. It is a toy MLP, not a transformer, and one seed. Part 2: with both matrices at exact zero, neither ever gets a gradient, at full precision. So a block that starts that way stays stuck without any rounding. It does not show that bf16 rounding never stalls a weight; other papers in the folder show that it can.

## Appendix: the two CPU checks behind section 10 (H1 and H3)

Both import the detectors from `sdk/src` on `main` at `b11b2c6` and change nothing. Run as `python3 <script> /path/to/sdk/src`.

**H1: a slow rise in the largest attention logit.** One attention module, one probe every 1,000 steps, the logit multiplied by a fixed factor at each probe from 10 until it passes 10,000.

```python
"""Does attention_logit_growth fire on a slow, steady rise? Uses the detector on main, unchanged."""
import sys
sys.path.insert(0, sys.argv[1])
from tensorquill import evidence as ev
fn = ev.detect_attention_incidents if hasattr(ev, "detect_attention_incidents") else None
if fn is None:
    fn = [getattr(ev, n) for n in dir(ev) if "attention" in n.lower() and callable(getattr(ev, n)) and n.startswith(("detect", "_detect", "attention"))]
    print("candidates", [f.__name__ for f in fn]); fn = fn[0]
def run(growth, start=10.0, stop=10000.0):
    recs, v, step = [], start, 999
    while v < stop:
        recs.append({"step": step, "probe": {"attention": {"blk5.attn": {"entropy_share": [0.5, 0.5], "max_logit": [v, v * 0.9]}}}})
        v *= growth; step += 1000
    out = fn(recs)
    hits = [i for i in out if i.get("kind") == "attention_logit_growth"]
    return len(recs), v / growth, hits
for g in (1.05, 1.10, 1.20, 1.25, 1.27, 1.30, 1.50, 2.0):
    n, last, hits = run(g)
    print(f"growth x{g:.2f} per probe | {n:3d} probes | logit 10 -> {last:8.0f} | incidents: {len(hits)}" + (f" first at step {hits[0]['step']} value {hits[0]['modules'][0]['value']:.0f}" if hits else ""))
```

```text
growth x1.05 per probe | 142 probes | logit 10 ->     9721 | incidents: 0
growth x1.10 per probe |  73 probes | logit 10 ->     9556 | incidents: 0
growth x1.20 per probe |  38 probes | logit 10 ->     8506 | incidents: 0
growth x1.25 per probe |  31 probes | logit 10 ->     8078 | incidents: 0
growth x1.27 per probe |  29 probes | logit 10 ->     8063 | incidents: 1 first at step 5999 value 33
growth x1.30 per probe |  27 probes | logit 10 ->     9173 | incidents: 1 first at step 5999 value 37
growth x1.50 per probe |  18 probes | logit 10 ->     9853 | incidents: 1 first at step 3999 value 34
growth x2.00 per probe |  10 probes | logit 10 ->     5120 | incidents: 1 first at step 2999 value 40
```

The line sits where it should: the baseline is the median of five rising values, which is the value three probes back, and 1.26 cubed is 2.

**H3: one corrupted step under Adam.** A gradient fault (one tensor's gradient times 1e6) and a state fault (the same tensor's second moment times 1e12, gradient untouched), each at step 700 of 1,500, each compared with the same run without the fault.

```python
"""What Adam's update does after one corrupted step, and what two rules on main say about it.
CPU, toy MLP, torch.optim.AdamW, no weight decay, no gradient clipping. Each faulted run is
compared with the same run without the fault (same seed, same data), step for step.
update_spike: the detector on main, unchanged. Weights-not-updating: only its condition is
emulated here (the tensor's float64 norm exactly equal at checks 50 steps apart)."""
import sys, math, torch
sys.path.insert(0, sys.argv[1])
from tensorquill.evidence import detect_update_spikes
AT, STEPS, LR, TARGET = 700, 1500, 1e-3, "2.weight"
def run(beta2, fault):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                              torch.nn.Tanh(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, beta2), weight_decay=0.0)
    w_true = torch.randn(32, 1)
    names = [n for n, _ in net.named_parameters()]
    p = dict(net.named_parameters())[TARGET]
    recs, series, norms = [], [], []
    for step in range(STEPS):
        x = torch.randn(64, 32); y = x @ w_true + 0.1 * torch.randn(64, 1)
        opt.zero_grad(); loss = ((net(x) - y) ** 2).mean(); loss.backward()
        if step == AT and fault == "grad":    # one tensor's gradient is wrong by 1e6
            p.grad.mul_(1e6)
        gnorm = math.sqrt(sum(float(q.grad.pow(2).sum()) for q in net.parameters()))
        before = [q.detach().clone() for q in net.parameters()]
        if step == AT and fault == "state":   # Adam's second moment of one tensor is wrong by 1e12; gradient untouched
            opt.state[p]["exp_avg_sq"].mul_(1e12)
        opt.step()
        rms = {n: float((q.detach() - b).pow(2).mean().sqrt()) for n, q, b in zip(names, net.parameters(), before)}
        recs.append({"step": step, "grad_norm": gnorm, "update_rms": rms})
        series.append(rms[TARGET] / LR)
        if (step + 1) % 50 == 0:
            norms.append(float(torch.linalg.vector_norm(p.detach().double())))
    return series, detect_update_spikes(recs), norms, float(loss)
for b2 in (0.999, 0.95):
    clean, hit0, _, loss0 = run(b2, "none")
    print(f"beta2={b2} no fault: update_spike {'step %d' % hit0['step'] if hit0 else 'none'}, final loss {loss0:.4f}")
    for f in ("grad", "state"):
        s, hit, norms, loss = run(b2, f)
        ratio = lambda k: s[AT + k] / clean[AT + k]
        back = next((k for k in range(1, STEPS - AT) if ratio(k) < 0.5), None)
        rec = next((k for k in range(back or 1, STEPS - AT) if ratio(k) > 0.5), None) if back else None
        same = sum(1 for a, b in zip(norms[AT // 50:], norms[AT // 50 + 1:]) if a == b)
        print(f"  fault={f:5s} update RMS/lr at the fault {s[AT]:.3g} (no fault {clean[AT]:.3g}); faulted/unfaulted at +1 {ratio(1):.2g}, +10 {ratio(10):.2g}, +100 {ratio(100):.2g}, +500 {ratio(500):.2g}"
              f" | under half from +{back}, over half again at +{rec} | update_spike: {'step %d' % hit['step'] if hit else 'none'}"
              f" | 50-step norm checks exactly equal after the fault: {same} of {len(norms) - AT // 50 - 1} | final loss {loss:.4f}")
```

```text
beta2=0.999 no fault: update_spike none, final loss 0.3619
  fault=grad  update RMS/lr at the fault 2.25 (no fault 0.13); faulted/unfaulted at +1 14, +10 5.3, +100 0.0021, +500 0.0037 | under half from +34, over half again at +None | update_spike: step 700 | 50-step norm checks exactly equal after the fault: 0 of 15 | final loss 0.3099
  fault=state update RMS/lr at the fault 3.46e-08 (no fault 0.13); faulted/unfaulted at +1 2.7e-07, +10 3.3e-07, +100 1.6e-07, +500 9.3e-07 | under half from +1, over half again at +None | update_spike: none | 50-step norm checks exactly equal after the fault: 0 of 15 | final loss 0.3498
beta2=0.95 no fault: update_spike none, final loss 0.1128
  fault=grad  update RMS/lr at the fault 0.447 (no fault 0.203); faulted/unfaulted at +1 2.1, +10 0.89, +100 0.0066, +500 0.96 | under half from +20, over half again at +391 | update_spike: step 700 | 50-step norm checks exactly equal after the fault: 0 of 15 | final loss 0.1172
  fault=state update RMS/lr at the fault 6.46e-08 (no fault 0.203); faulted/unfaulted at +1 3.5e-07, +10 4.5e-07, +100 1.6e-05, +500 0.54 | under half from +1, over half again at +498 | update_spike: none | 50-step norm checks exactly equal after the fault: 0 of 15 | final loss 0.1239
```

What it shows. `update_spike` catches the gradient fault at its step with either β2. After a gradient fault the tensor's update also falls far below normal for a long time (to 0.002 of normal at +100 with β2 = 0.999), because the huge gradient sits in the second moment: this is the slow outcome He et al. describe. The state fault skips the spike and goes straight to the frozen tensor, and neither rule reports it. The update at a gradient fault is bounded by Adam's own arithmetic, (1 − β1)/√(1 − β2) times the learning rate per element: 3.16 at β2 = 0.999 and 0.447 at β2 = 0.95, which is what the run gives.

What it does not show. Anything about a real model, a run with gradient clipping, or a fault smaller than these. The final loss of this toy barely moves, so it says nothing about how much such a fault costs.

## Appendix: the two CPU checks behind section 11 (M1 and M2)

Both import from `sdk/src` on `main` at `b11b2c6` and change nothing. Run as `python3 <script> /path/to/sdk/src`. Both are toy models on CPU.

**M1: one weight corrupted in memory.** After step 700 the largest element of one 64 by 64 tensor is multiplied by a factor. Two rules are asked: `detect_grad_spikes` and `detect_update_spikes`. No other rule is exercised; the toy has no watched block, so the activation rule never runs.

```python
"""One weight element is multiplied by k after step 700 (a fault in memory, not an update).
CPU, toy MLP with tanh and then with ReLU, AdamW, no weight decay, no clipping. Compared with the same run without the fault.
The two detectors are the ones on main, unchanged. Weight norms have no rule on main (trend only)."""
import sys, math, torch
sys.path.insert(0, sys.argv[1])
from tensorquill.evidence import detect_update_spikes, detect_grad_spikes
AT, STEPS, LR, TARGET = 700, 1500, 1e-3, "2.weight"
def run(k, act):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(32, 64), act(), torch.nn.Linear(64, 64),
                              act(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    w_true = torch.randn(32, 1)
    names = [n for n, _ in net.named_parameters()]
    p = dict(net.named_parameters())[TARGET]
    recs, losses, wn = [], [], []
    for step in range(STEPS):
        x = torch.randn(64, 32); y = x @ w_true + 0.1 * torch.randn(64, 1)
        opt.zero_grad(); loss = ((net(x) - y) ** 2).mean(); loss.backward()
        g = {n: float(q.grad.norm()) for n, q in net.named_parameters()}
        before = [q.detach().clone() for q in net.parameters()]
        opt.step()
        rms = {n: float((q.detach() - b).pow(2).mean().sqrt()) for n, q, b in zip(names, net.parameters(), before)}
        recs.append({"step": step, "grad_norm": math.sqrt(sum(v * v for v in g.values())), "grad_norms": g, "update_rms": rms})
        losses.append(float(loss.detach())); wn.append(float(p.detach().double().norm()))
        if step == AT and k:
            with torch.no_grad(): p.view(-1)[p.abs().argmax()] *= k   # the largest element of one tensor
    return recs, losses, wn
m = lambda xs: sum(xs) / len(xs)
hit = lambda d: d["step"] if d else "none"
for act in (torch.nn.Tanh, torch.nn.ReLU):
    base, l0, w0 = run(0, act)
    print(f"{act.__name__} no fault: grad_spike {hit(detect_grad_spikes(base))}, update_spike {hit(detect_update_spikes(base))}, mean loss last 100 {m(l0[-100:]):.4f}")
    for k in (2, 10, 100, 1e8):
        r, l, w = run(k, act)
        print(f"  x{k:g}: tensor norm {w[AT-1]:.3g} -> {w[AT+1]:.3g} | loss at +1 {l[AT+1]:.3g} (no fault {l0[AT+1]:.3g}) | grad_spike: {hit(detect_grad_spikes(r))} | update_spike: {hit(detect_update_spikes(r))}"
              f" | mean loss last 100 {m(l[-100:]):.4g} (no fault {m(l0[-100:]):.4g})")
```

Output:

```text
Tanh no fault: grad_spike none, update_spike none, mean loss last 100 0.2947
  x2: tensor norm 5.33 -> 5.34 | loss at +1 0.339 (no fault 0.338) | grad_spike: none | update_spike: none | mean loss last 100 0.2948 (no fault 0.2947)
  x10: tensor norm 5.33 -> 5.73 | loss at +1 0.352 (no fault 0.338) | grad_spike: none | update_spike: none | mean loss last 100 0.296 (no fault 0.2947)
  x100: tensor norm 5.33 -> 21.8 | loss at +1 0.362 (no fault 0.338) | grad_spike: none | update_spike: none | mean loss last 100 0.3016 (no fault 0.2947)
  x1e+08: tensor norm 5.33 -> 2.12e+07 | loss at +1 0.369 (no fault 0.338) | grad_spike: none | update_spike: none | mean loss last 100 0.3025 (no fault 0.2947)
ReLU no fault: grad_spike none, update_spike none, mean loss last 100 0.0411
  x2: tensor norm 6.16 -> 6.18 | loss at +1 0.097 (no fault 0.0962) | grad_spike: none | update_spike: none | mean loss last 100 0.04112 (no fault 0.04114)
  x10: tensor norm 6.16 -> 6.62 | loss at +1 0.111 (no fault 0.0962) | grad_spike: none | update_spike: none | mean loss last 100 0.04119 (no fault 0.04114)
  x100: tensor norm 6.16 -> 25 | loss at +1 1.23 (no fault 0.0962) | grad_spike: 701 | update_spike: 701 | mean loss last 100 0.04177 (no fault 0.04114)
  x1e+08: tensor norm 6.16 -> 2.43e+07 | loss at +1 1.09e+12 (no fault 0.0962) | grad_spike: 701 | update_spike: 701 | mean loss last 100 1.173e+04 (no fault 0.04114)
```

With tanh neither rule fires at any factor, and the faulted run ends with a loss 2.6% above the unfaulted one at a factor of 1e8. With ReLU both fire one step after the fault at factors of 100 and 1e8, and neither at 2 or 10.

**M2: one replica with 32-bit values of Adam's betas.** Two replicas each compute a gradient on their own batch, the gradients are averaged, and both apply the average, as data-parallel training does. The fingerprints and the tolerance are the ones on `main`.

```python
"""Two data-parallel 'replicas'. Each computes a gradient on its own batch at its own weights,
the two gradients are averaged, and both apply the same averaged gradient, as DDP does.
One differs only in Adam's betas: the float32 value
of each beta, as a stand-in for the Laguna report's case (rank 0 with FP64 beta factors, the
other ranks FP32). When does the fingerprint spread on main pass DRIFT_TOLERANCE (1e-6)?
A hash check fails as soon as any weight bit differs. CPU, toy MLP, float32 weights."""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from tensorquill.distributed import parameter_fingerprints, spread_from_fingerprints
from tensorquill.evidence import DRIFT_TOLERANCE
def make(betas, lr):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 1))
    return net, torch.optim.AdamW(net.parameters(), lr=lr, betas=betas, weight_decay=0.0)
b64 = (0.9, 0.999); b32 = tuple(float(np.float32(b)) for b in b64)
print("betas", b64, "against", b32, "| tolerance", DRIFT_TOLERANCE)
for lr in (1e-3, 1e-4):
    (na, oa), (nb, ob) = make(b64, lr), make(b32, lr)
    torch.manual_seed(1); w = torch.randn(32, 1)
    first_bit = first_tol = None; worst = 0.0
    for step in range(1, 20001):
        for n, o in ((na, oa), (nb, ob)):
            x = torch.randn(64, 32); y = x @ w + 0.1 * torch.randn(64, 1)
            o.zero_grad(); ((n(x) - y) ** 2).mean().backward()
        for p, q in zip(na.parameters(), nb.parameters()):   # the reduction: one averaged gradient for both
            g = (p.grad + q.grad) / 2; p.grad.copy_(g); q.grad.copy_(g)
        oa.step(); ob.step()
        if first_bit is None and any(not torch.equal(p, q) for p, q in zip(na.parameters(), nb.parameters())):
            first_bit = step
        if step % 50 == 0:
            fp = torch.stack([parameter_fingerprints(list(n.parameters())) for n in (na, nb)])
            s = float(spread_from_fingerprints(fp).max()); worst = max(worst, s)
            if first_tol is None and s > DRIFT_TOLERANCE: first_tol = step
    print(f"lr={lr:g}: first step with any differing weight bit {first_bit} | first 50-step check over the tolerance {first_tol} | largest spread seen in 20,000 steps {worst:.3g}")
```

Output:

```text
betas (0.9, 0.999) against (0.8999999761581421, 0.9990000128746033) | tolerance 1e-06
lr=0.001: first step with any differing weight bit 1 | first 50-step check over the tolerance 50 | largest spread seen in 20,000 steps 4.46e-05
lr=0.0001: first step with any differing weight bit 1 | first 50-step check over the tolerance 200 | largest spread seen in 20,000 steps 6.94e-05
```
