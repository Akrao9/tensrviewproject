# TensorQuill failure list

Gate 0 asks for a list of training failures that loss curves alone would not explain, with how long each took to notice. This list starts from public reports, because there are no customer interviews. It is a research list, not the author's own experience: the last section is for the failures you have hit yourself, and those should outrank everything here.

Each entry says whether the loss curve shows the problem, how it was noticed and how long that took where a source says so, the signal TensorQuill would need, and what TensorQuill captures today. Researched on Nov 13, 2026; the sources are linked at the end.

## Summary

| # | Failure | Loss curve shows it? | TensorQuill today | Priority |
| --- | --- | --- | --- | --- |
| 1 | Replicated weights drift apart across GPUs | No | Built (`replica_every=50`), two-GPU NCCL run pending | High |
| 2 | Silent data corruption on a faulty GPU | Usually not | Same detector as 1 | High |
| 3 | MoE routing collapse: dead or overloaded experts | No, or late | Detected (top-k diversity drop and 50-step starved window) | High |
| 4 | Weights not updating at all | Flat or slow loss, cause unclear | Detected (50-step float64 weight check, 3 unchanged checks) | High |
| 5 | Loss spikes from gradient outliers | Yes, but not the cause | Detected (5× 50-step median on `clip_grad_norm` or `grad_norm`) | High |
| 6 | Bad state after resuming from a checkpoint | A spike or a bend at resume | Partly: `start_step=`, repeated-step detector, and per-step `lr` recorded | Medium |
| 7 | Attention logit growth and entropy collapse | Late | Captured by probe pass (`max_logit`, `entropy_share`, 64×64 map); no detector yet | Medium |
| 8 | Activation outliers, especially with FP8 | Late | Detected in full capture (5× spike in `abs_max`/`std`, 3× drift in `abs_max/std`, dead outputs) | Medium |
| 9 | NaN or Inf | Yes, eventually | Detected (activations, gradients, `clip_grad_norm`, and replica weights) | Medium |
| 10 | Mixed-precision loss scale collapse and skipped steps | Sometimes | Detected with `init(..., scaler=)`: `step_skipped` and `grad_scale` per step; 10 skips in a row or a scale below 1 | Medium |
| 11 | MoE token dropping | No | Not captured | Medium |
| 12 | Embedding or other weight norms shrinking or growing | No | Recorded every 50 steps in float64; no slow-drift detector yet | Low |
| 13 | Silent compiler or framework bugs | No, or slow convergence | Not captured | Low |
| 14 | Data problems: repeated text, bad batches, repeated data | Spikes, or nothing | Out of scope | Low |
| 15 | Hardware stops, hangs and slow GPUs | The job stops or slows | Out of scope | — |

## Details

### 1. Replicated weights drift apart across GPUs

**What happened.** In BLOOM-176B, DeepSpeed's BF16 optimizer did not clip gradients on tensor-parallel ranks above 0. LayerNorm weights that should be identical on every rank drifted apart. The chronicles say the bug did not affect the model while training; it would have broken the final merge of ranks, so the checkpoints were at risk.

**How and when it was noticed.** Found on Mar 24, 2022, 13 days after the run started on Mar 11, by a test that compared LayerNorm weights across ranks. The loss was not the signal. The bug predated the run (older 104B checkpoints had it too), so the true time undetected is longer. TrainCheck's write-ups give ten and fifteen days for the same incident.

**Signal.** Per-parameter fingerprints compared across the ranks that should hold the same weights.

**TensorQuill today.** `replica_spread` is built and passes two CPU processes with seeded drift and a sign flip. Not yet run on two GPUs.

### 2. Silent data corruption on a faulty GPU

**What happened.** A faulty GPU computes wrong values without raising an error. Meta attributed 6 of 419 unexpected interruptions in a 54-day Llama 3 405B snapshot to silent data corruption. A study of faulty nodes found pre-training loss stayed close to healthy nodes while parameters drifted gradually; fine-tuning sometimes produced spikes or, in one case, zero test accuracy. Faults that flip exponent or sign bits can cause NaNs, short spikes, or lasting parameter divergence.

**How it is noticed.** Production teams hash or compare weights across data-parallel replicas (Laguna) or compare outputs of replicas that saw the same batch (SDCHunter, which reports 40 incidents caught and says synthetic GEMM stress tests miss over 60% of faulty GPUs).

**Signal.** The same cross-replica fingerprints as entry 1, plus NaN and spike detection.

**TensorQuill today.** Same detector as entry 1. The default tolerance of 1e-6 is a guess and must be checked against healthy multi-GPU runs, where nondeterminism alone may exceed it.

### 3. MoE routing collapse

**What happened.** The router sends most tokens to a few experts and the rest get none. DeepSeek-V3's report says an unbalanced load leads to routing collapse. One team training roughly 50B-parameter MoE models reports that a large router z-loss coefficient drove catastrophic expert collapse in the last layer. The total loss can hide it, since the model still learns through the experts it uses.

**How it is noticed.** Per-expert token counts per layer. Practitioners track the gap between the busiest expert's load and the mean.

**Signal.** Expert counts per router per step, which TensorQuill already records.

**TensorQuill today.** Detected. A collapse seeded in Granite layer 12 was found at the right step, layer and experts. Experts starved over a 50-step window are also flagged: on Granite, 20 experts that came starved from pretraining, which per-step checks never report. Gap: the 35% overload threshold cannot be reached by a top-k router with k above 2, which was the gap until the top-k rule: for routers with k of 3 or more, a drop in the router's diversity below 0.6 of its recent median now flags the experts that took the extra tokens, without needing dead experts. On the healthy Granite capture it raises no false alarms.

### 4. Weights not updating at all

**What happened.** Gradients are computed but never applied: a wrong FSDP wrapping in Hugging Face Accelerate (issue 2665) left `optimizer.step()` changing nothing, and a multi-GPU wrapper bug did the same while the gradient norm looked stable. The symptom is a flat or slowly worsening loss with no obvious cause.

**How it is noticed.** TrainCheck checks that `step()` actually changes the parameters.

**Signal.** The size of each parameter's update per step, or its ratio to the weight's size.

**TensorQuill today.** Detected. Each parameter's float64 norm after the optimizer step is compared with the previous check (every 50 steps by default), and a parameter with a non-zero gradient that is exactly unchanged on 3 checks in a row is reported. Tested on a toy model with two parameters left out of the optimizer. A first version using float32 norms gave 147 false reports out of 218 parameters on Granite in bf16, because a float32 norm misses few-element bf16 changes. It also reports bf16 weights too large for their updates to register, such as RMSNorm weights near 1.0 under a 1e-5 learning rate. That is a real failure too: pure-bf16 training without float32 master weights loses those updates. On the RTX PRO 6000 Granite run it reported exactly the 49 bf16 RMSNorm weights and nothing else.

### 5. Loss spikes from gradient outliers

**What happened.** PaLM's fix for its loss spikes was to restart from about 100 steps before the spike and skip 200 to 500 batches. Replaying those batches from an earlier checkpoint did not spike, so the cause was the combination of data and model state. OPT-175B lowered the learning rate and restarted from earlier checkpoints after divergences. BLOOM had a spike to loss 5.1 at iteration 31,219 with a gradient norm of 960 and recovered within about 30 iterations. Papers trace spikes to rare extreme gradient-norm spikes and to Adam's small epsilon and high beta2.

**How it is noticed.** The loss spike itself, which shows that something went wrong but not where.

**Signal.** Per-parameter gradient norms, to show which layer the spike started in, and update norms.

**TensorQuill today.** Detected, in router-only mode too when the run clips gradients: the total norm returned by `clip_grad_norm_` (or passed to `record_grad_norm`, as from Megatron's optimizer) is recorded every step at no extra cost. A step at 5× the median of the last 50 recorded steps is a spike; the report ranks parameters by their own spike and names the one that started rising first. Tested on synthetic records only; the factor is untuned.

### 6. Bad state after resuming from a checkpoint

**What happened.** On resume, the learning-rate schedule restarts warmup, the data loader replays data already seen, or optimizer state does not match the weights. The symptom is a spike or a change in slope right after the restart. One practitioner write-up says spikes after resuming MoE training are usually router logit drift rather than load imbalance.

**How it is noticed.** Comparing the learning rate and step right before saving and right after loading.

**Signal.** Learning rate, global step and data position recorded per step, and a check that they continue across a restart.

**TensorQuill today.** Partly. `init(start_step=...)` continues the step numbers on resume, repeated step numbers are reported (a resume from an earlier checkpoint is recognised by its `attempt` and not reported; its records replace the ones it rolled back), and each step's learning rate is recorded. A check that the learning rate continues smoothly across a restart is not built.

### 7. Attention logit growth and entropy collapse

**What happened.** Attention logits grow until attention becomes nearly one-hot, and training oscillates or diverges. Apple showed low attention entropy goes with instability and that forcing it low causes divergence. The growth comes from the query and key norms; QK-layernorm helps but does not remove the learning-rate sensitivity at scale.

**How it is noticed.** Per-head attention entropy or the largest attention logit, before the loss diverges.

**Signal.** Attention entropy and maximum logits per head. Query and key output norms are a cheap stand-in.

**TensorQuill today.** Captured by the Tier 2 probe pass (`probe=`, `probe_every=1000`): for each `scaled_dot_product_attention` call, per-head `max_logit`, `entropy`, and `entropy_share` (each query row's entropy divided by the log of its visible key count, leaving out all-masked and non-finite rows), plus sample 0's 64×64 pooled attention map, plotted in Rerun's Probe tab. No detector threshold is set yet.

### 8. Activation outliers, especially with FP8

**What happened.** A few activation values become very large. FP8 training diverges because its narrow range cannot hold them, often late in training. Massive activations are defined as values above 100 and at least 1,000 times the median in their hidden state.

**How it is noticed.** Per-layer kurtosis or the ratio of the maximum to a typical value; sustained low kurtosis predicted stable FP8 training in one study.

**Signal.** A fourth-moment (kurtosis) statistic per layer, which is as cheap as the statistics TensorQuill already computes.

**TensorQuill today.** Detected in full capture, relative to each module's own history: a sudden jump in the largest value or spread (5× the recent median), and drift of the outlier ratio `abs_max / std` or of the spread (3× from the module's early level). Healthy Granite raises nothing, although its middle blocks already carry outliers 740–840 times the standard deviation. Kurtosis is not captured; `abs_max / std` tracks the same outliers without a second pass over each output.

### 9. NaN or Inf

**What happened.** A value becomes NaN or infinite and spreads. Silent data corruption is one cause.

**Signal and TensorQuill today.** Detected: the first step and the layers and gradients involved.

### 10. Mixed-precision loss scale collapse and skipped steps

**What happened.** In OPT-175B, divergences were reported to go with the dynamic loss scale collapsing toward zero and with spikes in the final layer's activation norms (from a secondary summary; the paper's section 2.5 is the place to confirm). With `GradScaler`, steps with overflowing gradients are skipped silently.

**Signal.** The loss scale and the count of skipped steps.

**TensorQuill today.** Detected when the run passes its scaler, `init(..., scaler=scaler)` (Nov 14, 2026): each record carries `step_skipped` and `grad_scale`, skipped steps are counted, and `loss_scale_trouble` reports 10 or more skips in a row or a loss scale below 1. Thresholds are first guesses, and it has not run on a real fp16 job.

### 11. MoE token dropping

**What happened.** With a capacity limit, tokens that overflow an expert skip it. MegaBlocks reports this hurts quality. The loss can stay low while many tokens are dropped.

**Signal.** Dropped tokens per layer and expert.

**TensorQuill today.** Not captured. Granite is dropless, so this matters only for capacity-limited MoE models.

### 12. Weight norms shrinking or growing

**What happened.** OLMo 2 stopped applying weight decay to token embeddings because shrinking embedding norms amplified early gradients through normalization and made training less stable. Parameter-norm growth is also tied to attention logit growth.

**Signal.** Per-parameter weight norms, which the optimizer hooks already have access to.

**TensorQuill today.** Recorded: weight norms per parameter after each recorded step, and the total in the Rerun recording. No detector for slow norm drift yet.

### 13. Silent compiler or framework bugs

**What happened.** A study reports 19.2% of high-priority PyTorch issues are wrong outputs from `torch.compile`, and cites OLMo failing to converge because of one. The Hugging Face gradient-accumulation bug of October 2024 normalized the loss wrongly for most language models, with a normal-looking loss curve; a related report reappeared in 2025.

**Signal.** Compare compiled and uncompiled outputs or gradients on a fixed batch.

**TensorQuill today.** Not captured. The planned probe pass already runs a fixed batch with the compiler off, so it could also compare against the compiled step.

### 14. Data problems

**What happened.** OLMo 2 found repeated n-gram sequences in the data caused loss spikes and filtered them out. PaLM's spikes depended on specific batches meeting a particular model state. Data loaders that restart from the beginning on resume repeat data.

**TensorQuill today.** Out of scope except through entry 6. Recording which batch or data position each step used would let a spike be traced to its data.

### 15. Hardware stops, hangs and slow GPUs

Meta logged 419 unexpected interruptions in 54 days of Llama 3 405B training, 58.7% of them GPU issues. BLOOM lost hours to GPU deaths, hangs at evaluation (about 18 hours once) and a slow GPU that cut throughput from 149 to 140 TFLOPs until found by binary search. These are cluster-monitoring problems; TensorQuill does not address them.

## What this means for the detectors

Built since Nov 13, 2026:

1. **Weight norms and stalled-weight detection** in float64 every 50 steps (`weights=50`, entries 4 and 12).
2. **Step continuation (`start_step=`), repeated-step detection and per-step learning rate** (entry 6).
3. **Gradient-spike detection** on `clip_grad_norm` or `grad_norm`, ranking parameters by their spike and naming the first to rise (entry 5).
4. **Top-k MoE diversity-drop overload and 50-step starved-expert detection** (entry 3).
5. **Activation spike, drift (`abs_max / std` and `std`) and dead-output detection** per repeated block in full capture (entry 8).
6. **Tier 2 probe pass** for 64-bin activation histograms and per-head attention `max_logit`, `entropy_share` and 64×64 maps (entry 7).

Remaining priorities in order of value for effort:

1. **Run the cross-GPU detector on two CUDA GPUs over NCCL** (`sdk/examples/replica_check.py`, entries 1 and 2).
2. **Run the dense-model outlier and probe benchmark on a GPU** (`sdk/examples/dense_outlier_check.py`, entries 7 and 8) and add attention-entropy / logit-growth detector rules once real failure traces are collected.
3. ~~Record `GradScaler` skipped steps and scale factor~~ (entry 10): done Nov 14, 2026; check the thresholds on a real fp16 run.

TrainCheck (OSDI '25) already catches entries 1 and 4 by checking invariants, such as "replicated weights match" and "`step()` changes the parameters". It is the closest prior work for these two entries and should be added to the competitor check.

## Your own failures

Fill this in from your own training runs. Each one you have actually hit should outrank the researched entries above.

| What went wrong | Model and setup | How you noticed | How long it took | What would have caught it sooner |
| --- | --- | --- | --- | --- |
| | | | | |

Questions that help:

- Which run did you have to throw away or restart, and why?
- Which problem did the loss curve hide, or show only late?
- What have you added print statements or hooks for while debugging?
- What do you worry about on your next MoE or multi-GPU run?

## Sources

- [BLOOM training chronicles, tr11-176B (backup)](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md)
- [TrainCheck: catching silent errors in DL training](https://arxiv.org/abs/2506.14813), and its [OSDI '25 paper](https://www.usenix.org/system/files/osdi25-jiang.pdf)
- [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)
- [Understanding Silent Data Corruption in LLM Training](https://arxiv.org/html/2502.12340v1)
- [Exploring Silent Data Corruption as a Reliability Challenge in LLM Training](https://arxiv.org/abs/2604.00726)
- [SDCHunter, OSDI '26](https://www.usenix.org/system/files/osdi26-zheng.pdf)
- [Laguna M.1/XS.2 Technical Report](https://arxiv.org/pdf/2605.27605)
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
- [Z-loss in LLMs: a story of expert collapse and specialization](https://alltoall.notion.site/z-loss-in-llms-a-story-of-expert-collapse-and-specialization)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311)
- [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068)
- [AdaGC: adaptive gradient clipping](https://arxiv.org/html/2502.11034v2) and [Adam epsilon and loss spikes](https://arxiv.org/pdf/2502.17055v3)
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/pdf/2309.14322)
- [Stabilizing Transformer Training by Preventing Attention Entropy Collapse](https://proceedings.mlr.press/v202/zhai23a.html)
- [Towards Fully FP8 GEMM LLM Training at Scale](https://arxiv.org/pdf/2505.20524)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/pdf/2211.15841)
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)
- [Fixing Gradient Accumulation (Hugging Face)](https://huggingface.co/blog/gradient_accumulation) and [transformers issue 40564](https://github.com/huggingface/transformers/issues/40564)
- [torch.compile correctness bugs study](https://arxiv.org/html/2604.08720v1)
- [Hugging Face Accelerate issue 2665](https://github.com/huggingface/accelerate/issues/2665)
- [Resuming training: optimizer and learning-rate state](https://lyceum.technology/magazine/checkpoint-resume-loss-spikes-fixing-optimizer-lr-state/)
