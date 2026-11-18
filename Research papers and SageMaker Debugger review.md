# Research papers and SageMaker Debugger review

Nov 17, 2026. A review of the 22 papers in `../researchpapers/`, of AWS's `smdebug` library in `../sagemaker-debugger/` and of its rule configurations in `../sagemaker-debugger-rulesconfig/` (cloned and added later the same day), read against TensorQuill as it is on `main` today. Nothing in the repository was changed. This file is new and untracked.

**How to read it.** Page numbers are PDF pages. "Measured" means the paper reports a number from an experiment. "Claims" means the paper says it without a number, or from one run. What I did not read is listed per paper at the end. TensorQuill facts come from the focused plan, the v3 plan, `sdk/README.md`, the failure list and the code under `sdk/src/tensorquill/`.

**The short version.**

- The papers support the product's premise. Five of them measure a run that fails while its loss looks healthy.
- They do not support four things TensorQuill says about itself. The largest is that silent data corruption is covered by the replica-drift check. It mostly is not.
- Reading `smdebug` turned up one defect in TensorQuill's capture: forward passes made during evaluation are added to the next training step's record. Reproduced on a toy model on CPU.
- No paper shows a failure that calls for breaking the freeze today. Three candidates for later come out stronger, and two new ones appear.

---

## 1. Ranked proposals

Buckets: **(a)** allowed now, no new incident kind. **(b)** a new kind, frozen, listed for when the freeze lifts. **(c)** outside the product.

| # | What to build or change | Source | Size | Bucket |
| --- | --- | --- | --- | --- |
| 1 | **Say what the replica-drift check can and cannot see of silent data corruption.** Change failure-list entry 2 from "same detector as 1" to a narrow statement (section 2, finding W1). Add one sentence to the `replica_drift` entry of `kind_checks` and to the agent instructions: equal replicas do not rule out corruption. | SCOUT pp. 1, 3; ByteRobust p. 5; Altenbernd et al. p. 9; LLM-PRISM p. 9; Ma et al. pp. 7–8 | Small | (a) claims |
| 2 | **Keep evaluation forward passes out of training-step records.** The router hook adds to its counts on every forward call and they are zeroed only when an optimizer step commits (`capture.py:262–282`). `session.phase("eval")` changes the heartbeat only (`capture.py:1329–1336`). So a validation pass between two steps lands in the next step's `moe` counts, and the layer hook folds it into that step's activation statistics (`capture.py:118–136`). Zero the buffers when the phase returns to `train`, do it in the Lightning and `Trainer` adapters, and add a test with an evaluation between two steps. **Reproduced on CPU** with a toy one-router model in a throwaway script outside the repository: every step routed 10 tokens, three evaluation batches of 10 ran inside `session.phase("eval")` between two steps, and the next step's record held 40 (`[0, 9, 15, 16]` against `[0, 3, 3, 4]`). Not checked: a compiled model, the adapters, and the layer statistics (the README's "Things to know" says the record keeps the last forward before the step, which would be the training one, so they may be unaffected). The README documents that a hook runs on every forward pass, under gradient accumulation; it does not mention evaluation, and I found no test for it. The probe pass already clears the buffers after its own forward (README, "Probe pass"), so the fix can reuse that path. A loop that never calls `session.phase` has the same problem and no phase change to hook the fix on. | `smdebug/core/modes.py`, `docs/api.md` "Modes", `smdebug/core/save_config.py` (a save configuration per mode) | Small to medium | (a) existing kinds |
| 3 | **Make replica drift name the rank by strict majority, with an "inconclusive" answer.** The plan's change compares each rank with the group's median. SCOUT votes on equal values, names a rank only when the healthy value holds a strict majority, and otherwise returns inconclusive "rather than naming an arbitrary culprit". Also record whether the fingerprints were bit-for-bit equal, beside the `1e-6` tolerance. | SCOUT pp. 6–7, 12 | Small to medium | (a) existing kind |
| 4 | **Correct four statements and two tables** (findings W2 to W6): the reason given for router-only capture, the kurtosis sentence, the TrainCheck figure and what it supports, the paper's title, and add SCOUT's `lm-resiliency` and SageMaker Debugger to the alternatives table. | 2606.28116 pp. 3, 5–6; FP8 paper p. 5; TrainCheck pp. 5, 10–11, 16; `lm-resiliency/README.md`; `sagemaker-debugger/README.md` | Small | (a) claims |
| 5 | **Run the detectors against public fault injectors and bug sets** (section 5). First: SCOUT's injector on a multi-rank job for result 1, and TrainCheck's 20 reproduced bugs to state honestly how many of them the 12 invariants catch. | `lm-resiliency/docs/fault_injection.md`; TrainCheck `docs/ae-eval-s5.1-silent-issue-detection.md` | Medium | (a) tests and claims |
| 6 | **Stall evidence: group the stopped ranks by stack and say which group is the odd one out.** `_describe_stuck` already puts ranks at the same place together in one sentence (`evidence.py:4335`). ByteRobust's step is to call the largest group the waiting majority and the rest the suspects. Put the groups and their sizes in the incident, and say so in the agent instructions. Second part: write an end marker when a local run closes, as `smdebug` does, so a finished file and a file that stopped growing do not read the same. | ByteRobust p. 8; SCOUT p. 10; `smdebug/core/access_layer/utils.py` (`END_OF_JOB_FILENAME = "training_job_end.ts"`) | Small | (a) evidence |
| 7 | **Threshold work on recorded and from-scratch runs.** (i) `activation_drift` compares with a module's first 50 records; on a model trained from scratch that baseline is the untrained network, and the papers predict the outlier ratio then grows far past 3× (finding W7; a toy run came to 2.6× of the 3× cutoff). Run it on nanoMoE or nanochat before result 2. (ii) Record a median-and-MAD score beside each ratio, so result 2 can compare rules without new captures. (iii) Decide whether single-step gradient spikes are "ordinary" (finding W8). | Massive Activations pp. 1–3; FP8 paper pp. 5–6; SPAM pp. 2–5; EnergyCal p. 5; TrainSDC pp. 5, 10 | Medium | (a) existing kinds |
| 8 | **Two more numbers from the probe pass, with no incident:** the language-model head's output-logit norm and mean, and the gap between its BF16 and FP32 logits on the same hidden states; and the per-token kurtosis of each block's output. Recorded the way `weight_trends` is: numbers only. | Chen et al. (quantization) pp. 2, 7–8; FP8 paper pp. 5–6 | Medium | (a) evidence |
| 9 | **A check for corruption that the replica-drift check cannot see.** Cheapest form: compare one per-layer activation statistic across data-parallel ranks at the same step and flag the rank far from the others' median (TrainSDC's residual gain guard). Strongest form: SCOUT's sampled replay of one layer on every replica with hashed outputs. | TrainSDC pp. 4–5, 10–11; SCOUT pp. 6–8, 12 | Medium (compare) to large (replay) | (b) new kind |
| 10 | **Output-logit growth in the language-model head** as a kind, once proposal 8 has numbers from real runs. None of the 32 kinds watches the head; `attention_logit_growth` is about attention. | Chen et al. (quantization) pp. 1–2, 5–8 | Medium | (b) new kind |

**Outside the product (c), noted and not proposed:** host-metric fault detection (Minder), automatic eviction, restart and checkpoint certification (ByteRobust, MegaScale, SCOUT's checkpoint gate), optimizer changes that remove spikes (SPAM, EnergyCal, SGD on the head), FP8 architectures (FOG), routing replay for RL (R3), routing control for agents (Li et al.), INT8 inference checks (SProbe), `smdebug`'s loss, overfit and class-imbalance rules, its batch-size and dataloader sizing hints, its profiler report and its stop-training action. An agent's diagnosis may name a published remedy; TensorQuill should not apply one.

### How the papers bear on the four candidates the plan already names

| Candidate | What the papers add | Standing after this review |
| --- | --- | --- |
| Per-token routing entropy | The only one with a direct study: 2606.28116 measures it on an 8B MoE under learning-rate and batch-size sweeps (pp. 5–6, 8). The paper gives no threshold, no lead time against the loss and no false-alarm rate. Healthy public checkpoints sit anywhere from 49% to 94% of the maximum (Table 1, p. 3), so only a self-baseline can work. It needs the router's logits; a router that returns indices cannot give it. | First. Measure its cost under `torch.compile` before anything else |
| Update size per parameter | Altenbernd et al. compute it from Adam's own state with no copy of the weights, flag a jump against the warm-up mean, and report about 1% cost and no false alarm on 21 clean runs of 60M to 1.3B models (pp. 6–9). Rotational Equilibrium gives a reference value for decayed weights under AdamW, `sqrt(2·lr·wd·(1−β1)/(1+β1))` per step (Table 1, p. 4). 2606.28116 warns that update size alone missed its fault: 104% of baseline while the spectrum was at 51% (p. 7). AWS's `WeightUpdateRatio` rule is configured to save the weights themselves every 500 steps (`ruleConfigs.json`): a full copy, and a ratio between saved copies, not per step. | Second. The v3 note that a magnitude "has no threshold to judge it by" is now out of date |
| Slow ranks | Confirms the plan: step time cannot find one (SCOUT p. 3, Lin et al. p. 5, MegaScale p. 12). Adds that most slowness is workload, not hardware: sequence-length imbalance, the last pipeline stage, Python garbage collection (Lin et al. pp. 9–12). AWS's `StepOutlier` profiler rule flags a step duration beyond 3 standard deviations after ignoring 10 outliers; its docstring does not say whether that is per worker. | Third, and as an observation. Record tokens per rank per step beside the collective timings |
| Hidden expert collapse | Weakest support here. One case report finds experts that all receive tokens and compute the same thing, output cosine above 0.99 (Sha, pp. 1, 12). No paper in the folder measures "an expert adds less and less". | Last. Needs expert outputs, which nothing captures today |

Proposal 9 belongs between the second and third rows if TensorQuill wants to keep saying it helps with silent data corruption.

---

## 2. Wrong or unsupported: claims and thresholds

Ordered by how much they matter.

**W1. "Silent data corruption on a faulty GPU: same detector as 1" is not supported.**
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

More important: the 18 come from invariants inferred from clean reference runs, at the level of API calls and variable updates (Table 2, p. 6). Most of the 20 bugs are of that kind: a missing `zero_grad`, a wrong autocast type, dropout left on in evaluation (Table 3, p. 11). By my reading, TensorQuill's own kinds could plausibly catch about five of the 20 (the three out-of-sync cases, and the two where the model stops updating). That is untested. The result supports the idea of invariants. It is not evidence for TensorQuill's 12. Two more points from the same paper help TensorQuill: TrainCheck's instrumentation "interferes with JIT compilation tools like torch.compile" (p. 16), and it leaves optimisation problems out of scope (p. 5), which is where TensorQuill's anomaly kinds sit.

**W5. The title.** The plan calls 2606.28116 "Mechanism-driven monitors for LLM training instability". The PDF in the folder (v3, Oct 18, 2026) is titled "Mechanism-Aligned Observability for Training Instability of Large Language Models". An earlier version may have carried the other title; I could not check.

**W6. "None of the three debuggers detects" multi-GPU failures leaves out two tools.**
The statement is true of the three debuggers named. But SCOUT ships as open source in `lm-resiliency` (BSD 3-Clause, cloned beside this repo). Its README says it localises hangs, stragglers and silent data corruption to a rank, with "less than 1% amortized overhead". The paper itself says throughput overhead was not measured (p. 13), so that README figure has no published measurement behind it. SCOUT does not look at layers, routing, attention or gradients, so it is not a per-layer debugger. It is still the closest tool on claim 1 and it is missing from the alternatives table. SageMaker Debugger is missing too (section 4).

**W7. `activation_drift` is likely to fire on a healthy large model trained from scratch. Predicted, not shown.**
The rule compares the recent median of `std`, or of `abs_max / std`, with the module's first 50 records, and reports a 3× move (`evidence.py:76`, `754–762`). It was calibrated on pretrained Granite and SmolLM2, where the outliers already exist: 740 to 840 times the standard deviation in healthy Granite. Massive activations are learned during pretraining (Massive Activations p. 2), reach thousands of times the median (pp. 1–3), and the FP8 paper shows kurtosis growing through healthy training (pp. 5–6, Fig. 4). At initialisation the ratio is a few units. So on a real from-scratch run the ratio should pass 3× its first 50 records as a matter of course.

I tried it on a toy and it did **not** fire: a 4-block, 96-wide character model trained from scratch on CPU for 3,000 steps with full capture and no fault (loss 5.65 to 1.23) reported no incident. The largest move from the first-50 baseline was 2.62× in `std` (last block) and 2.05× in `abs_max / std` (from about 3.5 to 6.3). That is under the cutoff, with far less margin than the 1.27× seen on pretrained Granite. A toy this small does not grow massive activations, so it does not settle the question. nanochat and nanoMoE, the candidates for result 2, are from-scratch runs and would.

**W8. Single-step gradient spikes may not be "ordinary".**
The plan found 277 to 473 single-step jumps of 5× in one parameter on each healthy Granite capture and built the three-in-a-row rule around them. SPAM reports the same thing in LLaMA and Pythia training, frequent short spikes with LayerNorm weights the worst (p. 3), which matches TensorQuill's captures. But SPAM also measured that zeroing those spikes improves final perplexity (p. 5), and that one spike stays in Adam's second moment for more than 200 steps in a simulation (p. 5). So "healthy" is the wrong word for those steps; "not reported" is the accurate one. SPAM's threshold was 50× on single elements, not 5× on a norm, so the numbers are not comparable.

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
| bf16 weights whose updates round away are a real failure | Sha p. 14 (Table 2); Chen et al. pp. 2, 7–8 ("BF16 true" fails first); FlashOptim p. 3 (1/256 relative bound) |
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
| 1 | **Mechanism-Aligned Observability for Training Instability of Large Language Models.** Huang, Wang, Fang, Huang, You, Wang, Wu, Zheng. 2026 (arXiv 2606.28116v3). Two identical copies in the folder | Measured on one 8B MoE (128 experts, top-6). An injected FlashAttention fault moved the spectrum of 500-step weight updates to 51% and 87% of baseline at step 15,000, with raw weights at 96–97% and update norm at 104%; the loss turned near step 19,000 (p. 7). Per-token routing entropy fell earlier and deeper at higher learning rates and smaller batches (p. 8). No threshold, no false-alarm rate, no code found | X, S | W2, W5; first freeze-lift candidate; update size alone is not enough |
| 2 | **Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks** (TrainCheck). Jiang, Zhou, Xu, Liu, Xu, Huang. 2025 (arXiv 2506.14813v1; OSDI '25) | Measured: 18 of 20 reproduced silent bugs caught within one iteration by inferred invariants; signal detectors on loss caught 2 (pp. 10–11). Under 2% false positives on 63 clean programs (pp. 13–14). Overhead usually under 2%, up to 1.6× on toy jobs (p. 15). Breaks under `torch.compile` (p. 16) | X, C | W4; proposal 5 |
| 3 | **SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training.** Zhuang Wang. 2026 (arXiv 2608.11034v1) | Names the faulty rank by strict majority among equivalent replicas, for hangs, stragglers and corruption. Measured on at most 16 A100s with injected faults: 344 of 344 numerical faults localised, 30 of 30 clean runs quiet (pp. 12–13). Overhead, false positives at scale and recovery time not measured (p. 13) | C, X, S | W1, W6; proposals 3, 5, 9 |
| 4 | **Understanding Silent Data Corruption in LLM Training.** Ma, Pei, Lausen, Karypis. 2025 (arXiv 2502.12340v1) | Measured on 15 real faulty nodes: gradient noise at most 5.1% of the true norm (p. 6); pretraining loss identical to the healthy node while weights drift from the healthy run (p. 7); fine-tuning spikes on some nodes, one ending at zero accuracy (p. 8). Checksummed matmul did not detect reliably (p. 8). One node, tensor parallel only | X, C | W1 |
| 5 | **Exploring Silent Data Corruption as a Reliability Challenge in LLM Training.** Altenbernd, Wiesner, Kao. 2026 (arXiv 2604.00726v1) | Measured by flipping bits in GPU matmul instructions on 60M to 1.3B models: only exponent bits matter (p. 3); backward faults cause loss bumps (p. 4); an infinite gradient norm clips to a zero update, and without clipping freezes Adam (p. 5). A detector on update size from Adam's state: no false alarm on clean runs, about 1% cost (pp. 8–9). Single GPU only | S | Update-size candidate; name the "non-finite then stalled" pattern in the agent instructions |
| 6 | **LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training.** Tyagi, Hukerikar, Saxena, Huang, Shirvani, Tung, Zhu. 2026 (arXiv 2604.10390v2) | Measured over 7,664 runs of GPT-2 small and medium: 57–76% of runs unaffected; FP8 runs either recover or crash (41.9% crash), FP16 degrades silently by up to about 600% perplexity (pp. 7, 9–10). Four failure modes, one of them loss-scale collapse (p. 7) | C, X | W1; supports `loss_scale_trouble` as a real fault path |
| 7 | **TrainSDC: Characterizing and Mitigating Silent Data Corruption in Large Language Model Training.** Xia, Xu, Yun, Lin, Liu, Li, Zhuo. 2026 (arXiv 2608.30769v1) | Measured: faults on the query/key path do lasting damage and "can escape both loss-based and activation-magnitude monitoring" (pp. 3–4). A guard that compares each rank's residual gain with the other ranks' median: 1 alarm in an 814-step fault-free trace on Llama 3.2-1B and none on Qwen3-0.6B; recall 43% to 86% on Llama and 41% to 78% on Qwen, by fault size; 1.65% and 6.76% overhead (pp. 7, 10–11) | S | Proposal 9; `activation_spike` cannot see query/key faults |
| 8 | **Massive Activations in Large Language Models.** Sun, Chen, Kolter, Liu. 2024 (arXiv 2402.17762v2; COLM 2024) | Measured on trained models: a handful of activations above 100 and about 1,000× the median, in fixed dimensions, at the first and delimiter tokens (pp. 1–3); zeroing four of them breaks LLaMA2-7B (p. 2) | C, X | W3, W7; proposal 7 |
| 9 | **Towards Fully FP8 GEMM LLM Training at Scale.** Hernández-Cano, Garbaya, Schlag, Jaggi. 2025 (NeurIPS 2025) | Measured: FP8 for every matmul in the block, 35–43% faster than BF16 Llama3 (p. 7), up to 43% at 8B (p. 2). One failed run where query/key/value kurtosis left its trend at 3B tokens, the gradient norm at 12B and the loss at 15B (p. 6). Larger models diverge later (Table 2, p. 8) | X, S | W3; proposal 8 |
| 10 | **Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks.** Kosson, Messmer, Jaggi. 2024 (arXiv 2305.17212v4; ICML 2024) | Derived and checked on ResNets and GPT-2: with weight decay a weight vector's per-step rotation settles at a value set by the hyperparameters; for AdamW `sqrt(2·lr·wd·(1−β1)/(1+β1))` (Table 1, p. 4). Not a failure study | S | A reference value for the update-size candidate |
| 11 | **Understanding Stragglers in Large Model Training Using What-if Analysis.** Lin, Jiang, Song, Zhao, Yu, Wang, Wang, Shi, Shi, Jia, Liu, Wang, Lin, Liu, Panda, Li. 2025 (arXiv 2505.05713v2) | Measured on five months of ByteDance jobs: 42.5% of jobs at least 10% slower; waste 7.8% at the median and 45% at the 99th percentile (pp. 2, 7–8). Causes are mostly workload imbalance, not a few bad machines (pp. 9–12) | C, S | Slow ranks as an observation; record tokens per rank |
| 12 | **Minder: Faulty Machine Detection for Large-scale Distributed Model Training.** Deng, Shi, Jiang, Zhang, Zhang, Zhang, Li, Song, Zhu, Liu, Li, Wang, Lin, Ye, Yu. 2025 (arXiv 2411.01791v2) | Measured in production: finds the faulty machine from host metrics in 3.6 s, precision 0.904, F1 0.893, on 150 confirmed faults (pp. 1, 8). Compares machines with each other and requires 4 minutes of persistence | C | None; host metrics are outside the product |
| 13 | **Robust LLM Training Infrastructure at ByteDance** (ByteRobust). Wan, Liu, Song, Wang, Zhang, Sheng and 29 others. 2025 (arXiv 2509.16293v4; SOSP '25) | Measured over three months: 38,236 explicit and 5,948 implicit failures; job hangs 9.9%, NaN 0.3% of incidents (pp. 2–3). Groups stack traces of all ranks; the dominant group is healthy, the rest are suspects (p. 8). 97% effective training time on 9,600 GPUs (p. 1) | C, S | Proposal 6 |
| 14 | **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs.** Jiang, Lin, Zhong and 29 others. 2024 (arXiv 2402.15627v1) | Measured: 55.2% utilisation on 12,288 GPUs (p. 1); over 90% of faults handled automatically (p. 11); about 0.5% of machines much slower, found by per-rank timing of code segments (p. 8) | C | None beyond the slow-rank design |
| 15 | **The Role of Quantization in Loss Spikes.** Chen, Somasundaram, Li, McLeish, Kirchenbauer, Goldstein, Goldblum. 2026 (arXiv 2610.38277v2) | Measured on 28M to 1.4B models at high learning rates: late, unrecoverable divergence follows growth of the output logits; the BF16-to-FP32 logit error grows by orders of magnitude first; an FP32 head stays stable across five seeds (pp. 2, 7–8) | S | Proposals 8 and 10 |
| 16 | **SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training.** Huang, Zhu, Jin, Liu, Wang, Liu. 2025 (ICLR 2025; arXiv 2501.06842v2) | Measured on 60M to 1B models: gradient spikes up to 1,000× typical, frequent, worst in LayerNorm weights (pp. 2–3); zeroing them improves perplexity (p. 5) | C, X | W8 |
| 17 | **EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training.** Masato Fujitake. 2026 (arXiv 2610.37062v1) | Measured on one 4B MoE, three seeds: loss-spike burden down 99.51%, episodes from 37.3 to 3.3 (p. 5). The author says the comparison does not isolate the cause (pp. 2, 5, 7). Defines a spike as a loss above `max(m+0.5, 1.2m, m+8σ)` over the last 101 losses (p. 5) | S | A robust spike definition for proposal 7 |
| 18 | **Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers** (R3). Ma, Zhang, Zhao, Song, Wang, Sui, Luo. 2026 (arXiv 2510.11370v3) | Measured: float noise flips a near-tied top-k choice, so the training and inference engines route differently; policy divergence 1.5e-3 for the MoE against 6.4e-4 for a dense model (pp. 4–5, 8) | N | RL is a frozen pack. One fact for `compare_runs`: two runs that should match can part first at a router |
| 19 | **Sparse Mixture-of-Experts Routing in Visual Diffusion Transformers: Diagnosis, Boundary Calibration and Evolutionary Roadmap from Routing Collapse to Selective Deadlock.** Haiying Sha. 2026 (arXiv 2605.19378v1) | A case report on one 5B video model: experts homogenised (cosine above 0.99), a third of layers fell to one expert and a larger auxiliary loss did not help (pp. 12–13); bf16 updates truncated to zero (p. 14). One author, one setup, not peer reviewed | C | Hidden-collapse candidate; confirms per-layer reporting |
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

Nothing was downloaded. "Local" means a clone already sits beside this repository.

| What | Where | Use for TensorQuill | Notes |
| --- | --- | --- | --- |
| SCOUT's fault injector: JSON campaigns of faults on weights, biases, optimizer state, tensors, processes and collectives, for PyTorch, TorchTitan, Megatron and DeepSpeed; its README describes a campaign of 21 failure types on one eight-GPU host | **Local:** `../lm-resiliency/docs/fault_injection.md`, `examples/torchrun/`; upstream `github.com/LMResiliency/lm-resiliency`; BSD 3-Clause | Result 1 (two-GPU seeded cases) and result 4 (real multi-rank captures for the agent benchmark). It gives planted faults with a known rank and step | I read the README and the first tables of the injection guide, not the code. It needs GPUs |
| TrainCheck's 20 reproduced silent bugs: reproduction scripts, pre-collected traces and curated invariants | `github.com/OrderLab/TrainCheck-Evaluation-Workloads` (named in the local `../TrainCheck/docs/ae-eval-s5.1-silent-issue-detection.md`); not cloned | Run TensorQuill on the reproduction scripts and publish how many it catches, beside TrainCheck's 18. This is the honest version of the claim in W4 | The local doc marks 12 of the 14 listed bugs as supported for artifact evaluation. Many are tiny models on old library versions |
| TrainCheck's overhead benchmarks | `github.com/OrderLab/TrainCheck-Benchmarks` (named in `../TrainCheck/docs/benchmarks.md`) | A fourth arm for result 3, the head-to-head under `torch.compile` | TrainCheck says it does not support compiled code (p. 16); that is itself a result |
| Loss-spike reproductions: small models at high learning rates that diverge late | `github.com/tingtang2/quantization-loss-spikes` (Chen et al., p. 1) | A from-scratch failure that no injected fault produces: test `attention_logit_growth`, `grad_spike`, and proposal 8's numbers. Models start at 28M parameters, so a CPU or one small GPU may do | Not looked at |
| Massive-activation analysis code | `github.com/locuslab/massive-activations` (p. 1) | Check which token positions and dimensions carry the outliers in Granite and SmolLM2, for proposal 7 | Not looked at |
| Rotational-equilibrium code | `github.com/epfml/REQ` (p. 1) | Reference update sizes, if the update-size candidate is built | Not looked at |
| FOG (FP8) training code | `github.com/anonymous4375934/FOG`, as printed on p. 7 | A failing FP8 run with kurtosis traces | An anonymous address from the review period; it may not resolve |
| Straggler analysis | `github.com/ByteDance-Seed/StragglerAnalysis` (Lin et al., p. 1) | Possibly traces with per-operation timings per rank | The paper calls it a project page. I did not confirm that the trace is public |
| SPAM | A GitHub link on p. 1, cut off in my text extraction | Spike-prone small LLaMA runs | Read the link from the PDF |

Not public as far as the pages I read say: Minder's 150 fault instances, ByteRobust's incident records, the faulty nodes of Ma et al., LLM-PRISM's injection engine, TrainSDC's code.

---

## 6. What I did not read or could not verify

I read the text of each PDF as extracted by `pdftotext`, which was already installed. For most papers I read the first pages in full and then pulled the sentences on method, results and limits by keyword from the remaining main pages. Figures were not viewed, so every number above comes from text or a table. One table (Rotational Equilibrium, Table 1) was checked on the rendered PDF page.

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
| **Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes.** Liu, Cao, Li, Zhou (Tsinghua, University of Tokyo). Preprint, v5 Nov 2026 | In float32 the gradient of the correct class rounds to exactly zero once its logit leads the others by `23·ln 2`, about 16 (p. 3). Classifier updates went from about 1e-5 to two modes at ±4e-4 at the spike (p. 7). Casting only the loss computation to float64 removes the spikes (p. 2) | A CPU-scale reproduction for proposals 8 and 10, and a second mechanism beside Chen et al. for output-logit trouble |
| **Understanding Silent Data Corruption in LLM Training** (the ACL 2025 version of row 4 in section 3). Ma, Pei, Lausen, Karypis | Same paper, published in the Proceedings of the 63rd Annual Meeting of the ACL, pages 20372–20394 | Confirms the venue. Its finding on checksums is that they missed corruption: they flagged errors on one node only |

None of the three ByteDance systems appears to be released; I found no code link in any of them.

### What changes in sections 1 and 2

- **W1 is stronger.** It rested on reasoning from four papers. Zheng et al. and Lei et al. now give production numbers for the same point: 3 of 18 incidents visible, and 25% coverage for loss and gradient norms.
- **W6 needs one more name.** OpGuard's wrappers "preserve the original execution mode (eager, torch.compile, CUDA Graphs)" (p. 8), and it reports at least 95% kernel coverage on a compiled pretraining pipeline (p. 14). Its overhead figures are not split by compilation, so TensorQuill can still be first to publish an overhead number under `torch.compile`. It cannot say that no other tool instruments compiled code.
- **W9 has three more supporters.** All three ByteDance papers compare replicas bit for bit under enforced determinism. OpGuard measured what a tolerance costs: without determinism controls and with a tolerance of `1e-3`, it placed the fault exactly in 7 of 20 cases (p. 13). It also reports that TTrace flagged 11 of its 20 cases and misses corruption smaller than its tolerance (p. 12).
- **Proposal 9 has two concrete designs.** Neither is SDCHunter as built, which is diagnosis by replay after an alert: half the data-parallel groups re-run the identical batch as the other half (pp. 3, 10–12).
  - *A shared probe batch.* TensorQuill's probe pass already runs one fixed batch; if every rank ran the same one, their block statistics could be compared across ranks. This is my reading, not a published method.
  - *Forward against recomputation.* AEGIS fingerprints an operator's output in the forward pass and again when the backward pass recomputes it, and treats any difference as corruption (p. 8). TensorQuill's hooks already see recomputation under activation checkpointing, and skip it.
  - AEGIS's third mechanism is out of reach: its checksum comes from the float32 accumulator inside the matmul kernel (p. 7).
  - Its 0.86% overhead is with sampling tuned to a budget. Unsampled, it averages 1.61% or 2.35% and peaks at 2.6% (p. 12).
- **The slow-rank candidate is half reachable.** Flight Recorder's per-collective durations can support the compute-slow rule, where the rank that spends least time in the collective is the slow one (CCL-D p. 7, Aegis p. 10). The communication-slow rule and the hardware-hang rule need send and receive counters inside the communication library, which CCL-D and Aegis both customise.
- **`collective_hang` should not assume one rule.** In CCL-D the rank whose counter did not advance is the root cause only for Not-Entered hangs, 11.8% of the total. For Inconsistent hangs "the non-hang ranks are the root cause", and for Hardware-Fault hangs it is the rank with the fewest send and receive counts (p. 7).
- **The 300-second silence default now has a source.** CCL-D sets its hang bound at five minutes because "97% of cases exceeding this limit cannot recover" (p. 8). Section 2 could only say the default was in the same range as other systems.
- **Section 5 gains two bug sets,** neither with a repository link I could find: TTrace's 11 to 14 Megatron-LM bugs, and OpGuard's 10 open-source issues from Megatron-LM, DeepSpeed and others (p. 12).

### Where the survey goes further than these papers

| Survey statement | What the paper says |
| --- | --- |
| "The rank with the lowest collective sequence number is the hang's root cause" (summary and finding 5) | True for 11.8% of CCL-D's hangs. See `collective_hang` above |
| K2 "should follow SDCHunter" | SDCHunter is replay after an alert, on a deterministic job. AEGIS is the online detector |
| "Any in-process checksum must accumulate in float32 or wider" (finding 2) | Correct as a constraint, but AEGIS meets it inside the kernel, where a hook cannot go |
| Ma et al. found "false positives on healthy nodes in bf16" (finding 2) | Ma et al. report misses. The false-positive description is AEGIS's, when citing them (p. 4) |
| K2 "must be tolerance-based" | The three ByteDance papers are bitwise, and OpGuard measured the loss from a tolerance |
| "No surveyed tool reports instrumentation overhead under torch.compile" | Holds narrowly. OpGuard runs under it and reports coverage there |
| Greyhound: "100% accuracy on computation fail-slows" | Six positive cases |
| Aegis's 0.8 and 1.5 cut-offs "should be verified in the paper" | Verified, in the direction the survey gives |
| OLMoE: all tokens go to one expert, "then a second" | The first half is in the text. The second is only in a figure I did not view |

Every number the survey quotes from these twelve papers matched the PDF.

### Not read or not verified

- Figures in all twelve.
- OLMoE beyond pages 1–22 and OLMo 2 beyond pages 1–16.
- The appendix of Liu et al., which holds the model size and seed the survey's seeded recipe quotes.
- OLMo 2's z-loss weight, which the survey notes is given two ways; the equation did not survive text extraction.
- Whether any of these systems has released code beyond what the PDFs say.

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

