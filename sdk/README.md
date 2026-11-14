# TensorLens capture SDK

Skeleton of the capture SDK from the [v3 plan](../TensorLens%20v3%20—%20Product%20and%20Build%20Plan.md). It records what happens inside a PyTorch model during training, cheaply enough to leave on, and in a way that survives `torch.compile`.

```python
import tensorlens

model.to(device)
session = tensorlens.init(model, routers={"blocks.3.router": 128})  # before torch.compile
# Router load only, cheap enough to leave on. For debugging, add capture="full".
model = torch.compile(model)

for batch in loader:
    loss = model(batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()        # TensorLens counts the step here

session.close()
```

To add TensorLens to your own training job, follow [Run TensorLens on your own job](RUN_ON_YOUR_JOB.md).

Each optimizer step becomes one JSON line in `tensorlens-out/<run_id>.rank<rank>.jsonl`.

There are two capture modes. `capture="routers"`, the default when `routers` are given, records only MoE router load (`moe`, `moe_stats`, `moe_top_k`) and the learning rate: this is the always-on mode. `capture="full"`, the default for models without routers, adds block activation statistics, gradient norms on every step and a weight check every 50 steps: this is the debugging mode. `layers`, `grads` and `weights` override either mode. See [Capture-mode decision](#capture-mode-decision) for the measurements behind it.

- **`layers`**: mean, standard deviation, absolute maximum and zero fraction of each watched module's output. By default the watched modules are the model's repeated blocks, such as each transformer decoder layer: the largest `ModuleList` of composite modules of one class, the unit PyTorch's FSDP `transformer_auto_wrap_policy` wraps. A model without such a list falls back to every leaf module with parameters, which `layers="all"` also selects; for Granite that is 171 modules against 24 blocks.
- **`grad_norms`** and **`grad_norm`**: the L2 norm of each parameter's gradient and their total.
- **`weight_norms`** and **`weights_unchanged`**: each parameter's L2 norm after the optimizer step, and which parameters are exactly unchanged since the previous check. Checked every 50 steps by default (`weights=N` for every N steps, `weights=False` to turn it off); a check also records that step's gradient norms.
- **`schema_version`** (1) and **`ts`**: the record format's version, and when the step ended, in Unix seconds.
- **`lr`**: each parameter group's learning rate after the step.
- **`step_skipped`** and **`grad_scale`** with `init(..., scaler=)`: whether the loss scaler skipped this optimizer step on Inf or NaN gradients, and the loss scale used.
- **`clip_grad_norm`**: the total gradient norm before clipping, in any capture mode, when your training clips gradients. TensorLens records what `torch.nn.utils.clip_grad_norm_` returns for this model's parameters while the session is open, which also covers Lightning's `gradient_clip_val` (Lightning calls that function and discards the result). It costs nothing extra: the norm is computed anyway and stays on the GPU until the next flush. Code that imported the function before `init`, or computes the norm elsewhere, can call `session.record_grad_norm(norm)`; with Megatron, pass the `grad_norm` that `optimizer.step()` returns. `clip_norm=False` turns the automatic recording off.
- **`moe`**, **`moe_stats`** and **`moe_top_k`**: tokens routed to each expert, plus router entropy, diversity (`exp(H)/N`), and dead-expert count/fraction, for every router you name (works whether the router outputs integer indices or floating-point logits with `Router(n_experts, top_k=...)`). Set `top_k` for top-k routers; the detectors need it.
- **`scalars`**: values you pass to `session.log`, such as `session.log({"train/loss": loss})`. A one-element tensor is copied on its device and moved to the CPU with the rest of the step data, so logging a CUDA loss adds no GPU sync, as Lightning keeps logged values on the device until it writes them. Call it before `optimizer.step()`, because with automatic step counting the step ends there. Pass the same values your tracker logs: TensorLens does not capture the loss on its own, since patching `Tensor.backward` would miss Megatron (which calls `torch.autograd.backward` directly) and record the scaled loss under `GradScaler`. Every incident in the evidence carries the logged values at its step with the median of the previous 50, and the Rerun export plots them under a Logged tab.

## What is built

| Part | State |
| --- | --- |
| Forward hooks into preallocated on-device buffers, marked static | Built, tested on CPU (including AOTAutograd `static_input_indices` check) |
| On-device ring buffer, moved to the CPU every `every` steps, written by a worker thread | Built, tested |
| Step counting through PyTorch's global optimizer hooks, limited to this model's optimizer | Built, tested |
| Gradient norms grouped by device and dtype (`torch._foreach_norm`, with `DTensor.to_local()` and fallback) | Built, tested |
| MoE expert load (a compare-and-sum per router, accumulated on the device) and router entropy / diversity (`moe_stats`) | Built, tested |
| Hugging Face `Trainer` callback (`tensorlens.huggingface.TensorLensCallback`): run ID from the tracker on the main process broadcast to every rank, steps from `state.global_step`, and the loss and pre-clipping gradient norm per optimizer step, taken by wrapping `Accelerator.backward` and `Accelerator.clip_grad_norm_` while a session is open. Works with `torch_compile=True` | CPU tests on one process with Transformers 5.23.0 / Accelerate 1.15.0 and their development versions. Not run on a GPU, on several ranks, with mixed precision, or with TRL |
| PyTorch Lightning callback (`tensorlens.lightning.TensorLensCallback`): run ID from the trainer's logger broadcast from rank 0, steps from `trainer.global_step`, the back-propagated loss per optimizer step, and the `16-mixed` scaler | Built, tested on CPU with Lightning 2.6.5 on one process and two (`ddp_spawn`) |
| Probe pass (`probe=`): activation histograms per block and attention logit, entropy and maps per layer, on a fixed batch every `probe_every` steps | Built, tested on CPU, under `torch.compile` with no graph breaks or recompiles, and on a small randomly initialized Hugging Face Granite MoE; not run on a GPU |
| Logged scalars (`session.log`) attached to incidents and plotted in Rerun | Built, tested on CPU and Apple MPS; not run on a GPU |
| Cross-GPU parameter fingerprints and replica spread (`parameter_fingerprints`, `replica_spread`) | Built, unit-tested on drift and sign-flip cases |
| Detectors and Rerun exporter (`build_evidence`, `export_rerun`, `python -m tensorlens`) | Built, tested on MoE collapse (top-k aware), starved experts, gradient spikes, weights not updating, repeated steps, NaN/Inf, and replica drift |
| Run ID from `run_id=`, `MLFLOW_RUN_ID`, the active MLflow run or the active W&B run; otherwise torchrun's or SLURM's job ID, or a generated `run-<UTC time>-<6 hex>` broadcast from rank 0, so separate untracked runs never append to the same file | Built, tested with stand-in tracker modules and two Gloo processes |
| No graph breaks and no recompiles under `torch.compile` | Tested with `eager`, `aot_eager`, and AOTAutograd static-input checks on CPU; on CUDA with Inductor and CUDA graphs on a toy MoE, and with Inductor and CUDA graphs off on Granite |

## What is not built or not verified

- **Arrow Flight and the Rust ingest server.** Records go to a JSON Lines file, and Python detectors/exporters turn that file into `evidence.json` and `.rrd`.
- **Replay** of a flagged batch (Tier 3). The probe pass (Tier 2) is built but has not run on a GPU.
- **Multi-rank reporting.** Every rank writes its own file.
- **Detector thresholds are untuned.** The gradient-spike factor (5× the median of the last 50 recorded steps), the top-k diversity drop (0.6 of its recent median), the starved-expert ratio and the stalled-weight check have been checked on synthetic cases and, for the MoE rules, on one Granite capture. Weight norms, learning rates and the spike and stalled-weight detectors have run on the Granite captures from the A100 and RTX PRO 6000 runs below, but not against a planted failure of their own kind on a GPU; the steps they flagged there are unconfirmed.
- **CUDA graphs on Granite.** Its `grouped_mm` expert path fails during CUDA-graph recording on PyTorch 2.14.1+cu130, so `mode="reduce-overhead"` is verified only on the toy model.

## Granite MoE GPU benchmark

The Colab run of [the Granite benchmark](examples/moe_dogfood.py) at commit `5d38288` used `ibm-granite/granite-3.0-1b-a400m-base` on WikiText-2 with BF16 weights, PyTorch `2.14.1+cu130`, and one NVIDIA RTX PRO 6000 Blackwell Server Edition. It ran with Inductor and CUDA graphs disabled. Each comparison used 50 warmup steps and three timed rounds of 200 steps per configuration.

| Reference configuration | Measured configuration | Reference median | Measured median | Added time |
| --- | --- | ---: | ---: | ---: |
| Activations and routers | Activations, routers, and gradient norms | 68.746 ms/step | 70.534 ms/step | +1.789 ms/step (+2.60%) |
| Capture off | Router-only (`layers=(), grads=False`) | 64.886 ms/step | 65.394 ms/step | +0.507 ms/step (+0.78%) |

The first row measures the incremental cost of gradient norms with activation and router capture already enabled. Comparing across stages estimates about **+5.95%** for activations and routers and **+8.70%** for full capture against capture off. A direct full-capture comparison with capture off under the same benchmark conditions remains pending.

Router-only capture passed the **under 1%** target on this run. Its three paired round differences were **+0.427, +0.510, and +0.469 ms/step** (**+0.66%, +0.79%, and +0.72%**, calculated from the rounded log times). Repeatability across sessions remains unverified. The benchmark verified the expected assignment totals across 650 recorded steps and all 24 routers in each capture configuration. The one hook-related recompile shown in the log occurred during warmup for the router-only model.

The healthy run's full-capture `granite.jsonl` exported to `granite.evidence.json` and `granite.rrd`, with no routing or numerical incidents reported.

## Seeded Granite collapse on CUDA

The Colab run with `--collapse-layer 12` passed on the RTX PRO 6000 with PyTorch `2.14.1+cu130`, Inductor, and CUDA graphs disabled. It verified assignment totals across 650 steps and 24 routers and found the first incident at **step 450 in layer 12**, with diversity **0.25**. Experts **1–8** each received **12.5%** of assignments, and experts **9–32** received none. The exporter wrote `granite-router-only.evidence.json` and `granite-router-only.rrd`.

Router-only capture added **+0.62%** in the seeded run. Both arms ran the fault hook, and the final timed round included the forced routing. The healthy run above supplies the overhead gate measurement. The gradient-norm cost in this run was **+1.78%**, against **+2.60%** in the healthy run; one round without gradient norms took 78.078 ms against about 68–69 ms for the others, so treat that cost as roughly 2–3% until it is measured more often.

### Viewer check

`granite-router-only.rrd` opened in the stock Rerun 0.43.2 viewer (headless, screenshots at steps 400 and 450). At step 450, layer 12 showed 8 amber and 24 red cells, the layer 12 expert table listed experts 1–8 at 12.50% and the rest at 0.00% with their change from step 449, and layer 12's diversity dropped to 0.25.

The check found a mismatch between the heatmap and the detector. Experts with no tokens in a sample were always red and labelled Dead, including transient gaps in healthy layers that the detector deliberately ignores; one appeared at step 400 and three outside layer 12 at step 450. Commit `40ce26a` colors those experts near-black (`IDLE_COLOR`), labels them Idle, keeps red for experts in the detector's dead list, and prints 12.5% instead of rounding it to 12%. The fix was then confirmed on the real capture by exporting a later run's `granite-router-only.jsonl` with the current SDK.

### Starved experts

That export showed 1,303 zero-token cells across 650 steps, mostly from a few experts that were almost never used. Layer 5 expert 25 had no tokens in 514 of 650 steps and an average share of 0.005%, against 3.125% for an even share; layer 8 expert 21 had none in 356 steps. These experts came that way in the pretrained model. The per-step detector never flags them, because one empty expert in a single step is ordinary noise.

`build_evidence` now flags an expert as starved when its share over the last 50 samples (`starve_window`) is below 10% of an even share (`starve_ratio`). It waits until an even share would have received at least 100 assignments in the window, and dead or overloaded flags take precedence. Starved experts are violet in the heatmap, `Starved` in the expert table and listed in the summary, worst first. Other zero-token experts are now gray rather than near-black, which was hard to tell from dark blue.

On the Granite capture it flags 20 experts in 15 layers, from 0.018% to 0.312% of assignments; 14 are flagged at step 49, the first full window. The 10% ratio is a starting point: about a third of those sit just under it, so a lower ratio would list only the most extreme few. The collapse result is unchanged.

## Detectors added for the failure list

From the [failure list](../TensorLens%20failure%20list.md), entries 3–6 and 12:

- **Top-k overload.** For routers with `top_k` of 3 or more, the 35% share threshold cannot fire (no expert can exceed 1/k), and a fixed lower threshold does not work either: on the healthy Granite capture some experts are picked by 85–100% of tokens in layers 2, 3, 6, 9 and 14, and with two documents per batch single experts often jump to over half the tokens for one step. A rule based on jumps in each expert's own share flagged 656 healthy samples. The rule used instead requires the router's diversity to fall below 0.6 of its median over the previous 50 samples (healthy Granite never went below 0.82; the seeded collapse fell to 0.29), and then flags the experts whose fraction of tokens rose by at least 25 points. On the Granite capture with `--top-k 8` it flags only layer 12 from step 450. Each sample records the rule and threshold it used. Use `python -m tensorlens --top-k 8` for captures made before `moe_top_k` was recorded.
- **Weights not updating.** Every 50 steps by default, the capture computes every parameter's norm in float64 and compares it with the previous check; `weights_unchanged` lists the parameters whose norm is exactly the same. A parameter with a finite non-zero gradient that is unchanged on 3 checks in a row (150 steps by default) is reported, for example because it was left out of the optimizer. Steps with a non-finite gradient norm are skipped, since mixed-precision loss scaling skips those on purpose. The first version compared float32 norms, and on the A100 run below it reported 147 of 218 parameters from step 1: a bf16 weight under a 1e-5 AdamW step changes in only a few elements per step, which a float32 norm often misses (`model.layers.7.self_attn.k_proj.weight` kept the same float32 norm for 128 steps while changing). Float64 norms resolve those changes. bf16 RMSNorm weights near 1.0 are still reported, correctly: their step size in bf16 is about 0.008, so a 1e-5 update rounds away, and in a CPU simulation such a weight was exactly unchanged on 273 of 650 steps. Each check reads every weight once, with no copy, accumulated in float64; on Apple MPS, which has no float64, the check is off. Captures made before `weights_unchanged` existed are not checked. The check runs every 50 steps rather than every step because the read alone costs about 1.5 ms for Granite's 2.6 GB of bf16 weights on the RTX PRO 6000, about 2% of a step, whatever arithmetic is used; with float64 on that GPU it measured about 3.7 ms. wandb.watch (every 1,000 batches by default) and TrainCheck (`sampling_interval`) sample their weight checks for the same reason. At every 50 steps the average cost is under 0.1 ms per step, not yet measured on a GPU.
- **Gradient spikes.** A step whose total gradient norm is at least 5× the median of the previous 50 recorded steps. The report ranks parameters by their own spike and names the one that started rising first.
- **Repeated steps.** Step numbers written twice by the same rank, as after a resume without `start_step`.
- **Activation spikes, drift and dead outputs** (full capture). Each watched module is compared with its own history, because trained LLMs carry large outliers by design: in healthy Granite the largest value in middle blocks is 740–840 times the standard deviation, so a fixed outlier threshold would fire at step 0. A spike is `abs_max` or `std` at 5× its median over the previous 50 records; drift is the recent median of `std`, or of the outlier ratio `abs_max / std`, 3× above or below the module's first 50 records; dead outputs are a zero fraction 0.25 above its baseline. On the healthy Granite capture (24 blocks, 650 steps) one-step jumps peaked at 2.2× and drift at 1.27×, and nothing was reported. A weight blown up 50× in one block of a test model was reported at that step. Kurtosis, the outlier measure used in FP8 training research, is not captured: `abs_max / std` tracks the same outliers from statistics already recorded, while kurtosis would need a second pass over each module's output inside the compiled graph.
- **Learning rate** is recorded per step and logged to the Rerun recording with the total weight norm.

## A100 run with the new detectors

Colab, NVIDIA A100-SXM4-80GB, PyTorch `2.14.1+cu130`, Inductor with CUDA graphs off, commit `9f88e50`, `--collapse-layer 12`. Steps take about 149 ms here against about 65 ms on the RTX PRO 6000, so the two runs' overheads are not directly comparable.

| Comparison | Reference median | Measured median | Added time |
| --- | ---: | ---: | ---: |
| Activations and routers → plus gradient and weight norms | 160.798 ms/step | 165.656 ms/step | +4.858 ms/step (+3.02%) |
| Capture off → router-only | 148.946 ms/step | 150.926 ms/step | +1.980 ms/step (+1.33%) |

- **Router-only missed the 1% target on the A100.** All three paired rounds were over it: +1.20%, +1.06% and +1.33%. The added time is about four times the RTX PRO 6000's (+0.51 ms) although the hooks do the same work, so per-step launch overhead on the host may matter more here than GPU time. Not investigated. Both arms include the fault hook, and the last round includes the forced collapse.
- **Gradient and weight norms** cost +3.02% over full activation and router capture. One round of that arm took 178.3 ms against 162.8 and 165.7 for the others.
- **Detection.** The router-only capture found the collapse at step 450 in layer 12, with experts 1–8 overloaded and 9–32 dead, and 20 starved experts. This run recorded `moe_top_k`, so the top-k rule ran without `--top-k`. The recording opened in the stock Rerun viewer with the current colors and legend.
- **Full capture (`granite.jsonl`)** recorded weight norms for all 218 parameters, the learning rate and `moe_top_k` on all 650 steps. Gradient spikes were flagged at steps 136, 141 and 645, at 5.2× the recent median, just over the threshold, starting in RMSNorm weights (`model.layers.6.input_layernorm.weight` at 26× its own median). That is not yet known to be a real problem. This capture predates the float64 weight check, so it is not checked for stalled weights; it is the run that showed the float32 check's 147 false reports.

## RTX PRO 6000 run with the float64 weight check

Colab, RTX PRO 6000 Blackwell Server Edition, PyTorch `2.14.1+cu130`, Inductor with CUDA graphs off, commit `fb4c2b7`, `--collapse-layer 12`.

| Comparison | Reference median | Measured median | Added time |
| --- | ---: | ---: | ---: |
| Activations and routers → plus gradient and float64 weight norms | 67.977 ms/step | 73.503 ms/step | +5.527 ms/step (+8.13%) |
| Capture off → router-only | 65.582 ms/step | 65.598 ms/step | +0.015 ms/step (+0.02%) |

- **The stalled-weight check reported the 49 RMSNorm weights and nothing else.** All 24 `input_layernorm`, all 24 `post_attention_layernorm` and the final `norm` were exactly unchanged on 3 or more recorded steps in a row, from step 3; no other parameter was ever exactly unchanged. This is the pure-bf16 stagnation described above, and the expected result.
- **The float64 weight norms are expensive on this GPU.** Gradient norms alone cost +2.60% (+1.79 ms) here at `5d38288`; with float64 weight norms the same arm costs +8.13% (+5.53 ms), so the weight check adds roughly 3.7 ms per recorded step. This GPU (GB202) runs float64 at 1/64 of its float32 rate, which likely explains the cost; the A100 runs float64 at half its float32 rate, but its run used float32 norms, so float64 there was not measured. The check now runs every 50 steps by default instead of on every step that records gradient norms.
- **Router-only capture** was within noise of capture off: paired rounds −0.50, +0.46 and −0.004 ms. With the earlier +0.78% on this GPU and +1.06–1.33% on the A100, the 1% target holds here and not on the A100.
- **Gradient spikes at steps 141 and 645**, starting in `model.layers.11.self_attn.k_proj.weight` (22× its median). The A100 run flagged 136, 141 and 645. The same steps on two GPUs point to the data at those steps, since both runs read the same batches in the same order.
- **The collapse** was found at step 450 in layer 12 with the same 20 starved experts.

## RTX PRO 6000 run with block-level capture (Stage 3)

Colab, RTX PRO 6000 Blackwell Server Edition, PyTorch `2.14.1+cu130`, Inductor with CUDA graphs off, commit `344a1e2`, no seeded collapse.

| Comparison | Reference median | Measured median | Added time |
| --- | ---: | ---: | ---: |
| Block activations and routers → plus gradient norms every step (weights every 50) | 66.168 ms/step | 67.659 ms/step | +1.491 ms/step (+2.25%) |
| Capture off → router-only | 65.646 ms/step | 66.003 ms/step | +0.357 ms/step (+0.54%) |
| Capture off → full capture with defaults | 64.967 ms/step | 67.663 ms/step | +2.697 ms/step (+4.15%) |

- **Full capture with defaults costs +4.15%, over the 1% target.** All three paired rounds agree: +2.60, +2.74 and +2.69 ms. From the three stages, the parts are roughly 0.36 ms for routers, 0.85 ms for the 24 block activation hooks and 1.5 ms for gradient norms on every step. Activation capture of every leaf module cost about 3.9 ms on this GPU before, so blocks cut it about 4.5 times; gradient norms are now the largest part.
- **Router-only passed again** at +0.54% (paired rounds −0.32, +0.36, +0.34 ms).
- **The 50-step weight check is cheap**: Stage 1's gradient arm cost +1.49 ms, about what gradient norms alone cost before float64 weight norms were added (+1.79 ms at `5d38288`).
- **Detectors on `granite-full.jsonl`**: 24 block statistics per record and weight checks at steps 49, 99, …, 649. No MoE incident, as expected without a seeded collapse; the same 20 starved experts. 28 parameters were reported as not updating from step 199, all of them RMSNorm weights; with 50 steps between checks, some RMSNorm weights change in a few near-zero elements and are no longer exactly unchanged. Gradient spikes at steps 136 and one more, again starting in `model.layers.6.input_layernorm.weight`.

## Capture-mode decision

Decided on Nov 13, 2026, after Stage 3: **router-only capture is the default for MoE models, and full capture is an opt-in debugging mode.**

- **Measured cost on Granite, RTX PRO 6000:** router-only +0.02% to +0.78% across three runs; full capture with block defaults +4.15%, of which about 1.5 ms is gradient norms on every step and 0.85 ms block activation statistics. On the A100, router-only was +1.06% to +1.33%.
- **Why not sample gradient norms to bring full capture down:** it would still be about 2%, because the block statistics run inside the compiled graph on every step and cannot be sampled without extra compiled variants; and most gradient spikes last one step, so sampling would miss them.
- **What comparable tools do:** detailed per-layer monitoring is sampled or opt-in. Composer's `ActivationMonitor` runs every N batches and warns that it adds overhead; wandb.watch defaults to every 1,000 batches and its docs say logging every step can significantly slow training; Lightning documents per-layer `grad_norm` as a debugging step. Always-on profilers aim for about 1% or less.
- **Why routers are the right always-on signal for MoE models:** a 2026 paper on preemptive detection of training instability argues that loss, gradient norms and weight norms are late indicators and proposes router-level monitors such as routing entropy, which router-only capture records.
- **What router-only gives up by default:** layer statistics, per-parameter gradient norms, stalled weights and non-finite gradient detection. Turn on `capture="full"` while investigating.
- **Gradient spikes stay on in router-only mode when the run clips gradients.** Training loops that clip compute the total gradient norm on every step; TensorLens records the value `clip_grad_norm_` returns (`clip_grad_norm`) and the spike detector uses it, without naming a parameter. PyTorch's `clip_grad_norm_` returns the pre-clipping norm as a tensor, Lightning's `Precision.clip_grad_by_norm` discards it, and Megatron's optimizer returns it from `step()`, which is why both automatic recording and `record_grad_norm` exist. Tested with Lightning 2.6.5's clipping path; not yet run on a GPU.

Sources: [Composer ActivationMonitor](https://mosaicml-composer.readthedocs-hosted.com/en/stable/_modules/composer/callbacks/activation_monitor.html), [wandb.watch](https://docs.wandb.ai/support/models/articles/how-do-i-log-gradients-and-model-weights-with-wandb-watch), [Pixie continuous profiling overhead](https://blog.px.dev/cpu-profiling-3), [Grafana continuous profiling costs](https://grafana.com/blog/continuous-profiling-in-production-a-real-world-example-to-measure-benefits-and-costs), [Mechanism-driven monitors for LLM training instability](https://arxiv.org/pdf/2606.28116), Lightning's `docs/source-pytorch/debug/debugging_intermediate.rst`.

## Where the A100 router-only overhead goes

**Pending a GPU run.** Router-only capture added about 2 ms per step on the A100 against 0.36–0.51 ms on the RTX PRO 6000 for the same hooks. A100 steps also take about 149 ms against 65 ms, although the model is small for either GPU, so both runs may be limited by the CPU launching kernels rather than by the GPU. Granite's `grouped_mm` expert path also copies expert offsets to the CPU, which syncs the CPU with the GPU and stops the CPU from running ahead; if that happens on every layer, every added launch adds to the step time.

[`examples/profile_overhead.py`](examples/profile_overhead.py) measures this. It times capture off and router-only like the benchmark, then profiles one 50-step window of each with `torch.profiler` and reports, per step, GPU busy and idle time, kernels, kernel launches, CPU syncs and the time spent waiting in them, the kernels capture added, and whether the added time is GPU work or host time. In Colab on the A100, from the repository root:

```sh
cd sdk
python -m pip install -e '.[rerun]' transformers huggingface_hub pyarrow
python examples/profile_overhead.py --out runs/profile-a100
```

Add `--capture full` to profile full capture, and `--trace` for a Chrome trace of each arm (open it at `chrome://tracing` or ui.perfetto.dev; it can be large). The same command on the RTX PRO 6000 gives the comparison. `--toy` checks the script on CPU without downloads.

## Dense model with a seeded activation outlier

**Pending a GPU run.** Everything so far ran on Granite, an MoE model. [`examples/dense_outlier_check.py`](examples/dense_outlier_check.py) checks the dense-model path: it trains SmolLM2-360M (Llama architecture, grouped-query attention) on WikiText-2 with capture off and with full capture plus the probe, in alternating 50-step chunks, and from step 250 scales two hidden channels of the middle block by 100, a seeded massive activation. It reports:

- full-capture overhead on a dense model, timed per step with CUDA events, and what a flush step and a probe step add, with the probe's average cost at every 100 steps and at the default 1,000;
- `PASS` if the first activation spike is at step 250 and includes the seeded block;
- the seeded block's probe histogram before and after, with a PNG of both on one shared axis, and the attention of the next block, the first to read the outlier;
- graph breaks, and a Rerun recording.

In Colab, from the repository root:

```sh
cd sdk
python -m pip install -e '.[rerun]' transformers huggingface_hub pyarrow matplotlib
python examples/dense_outlier_check.py --out runs/dense-outlier
```

On CPU with a tiny random Llama (`--toy --backend aot_eager`, PyTorch 2.10, transformers 5.5.4) it found the spike at step 250 in the seeded block, the probe's largest value in that block grew 77× between the probes before and after, and there were no graph breaks. CPU timings say nothing about a GPU.


**Pending.** The two-process CPU/Gloo check passed. The Colab runtime exposed one NVIDIA RTX PRO 6000, so the CUDA/NCCL check was skipped.

To resume, use a single machine with two visible NVIDIA GPUs and a CUDA-enabled PyTorch build. Run from the repository root. The check generates its own data and needs no Colab artifacts.

```sh
set -e
cd sdk
python -c 'import torch; assert torch.cuda.device_count() >= 2, "Two visible CUDA GPUs are required"'
python -m pip install -e .
python -m torch.distributed.run --standalone --nproc-per-node=2 examples/replica_check.py
```

Mark this complete when both NCCL ranks report `PASS`. Each rank checks healthy replicas at steps 0 and 2, injected drift at step 1, and a sign flip at step 3.

The check passes `replica_every=1`. In a training job the drift check runs every 50 steps by default (`replica_every=`), as Megatron's `check_weight_hash_across_dp_replicas_interval` does: each check reads every weight in float64 and gathers 16 bytes per parameter tensor from every rank, and a float64 pass over Granite's weights cost about 3.7 ms on the RTX PRO 6000. Until Nov 14, 2026 it ran on every step.

## Probe pass

`tensorlens.init(model, probe={"input_ids": fixed_batch}, probe_every=1000)` runs one fixed batch through the model every 1,000 steps (wandb.watch's default for histograms) and adds a `probe` field to that step's record:

- **`histograms`**: a 64-bin histogram of each repeated block's output (`lo`, `hi`, `counts`, and `nonfinite` for NaN and Inf values left out).
- **`attention`**: for each `scaled_dot_product_attention` call, named after the block it ran in, per head: the largest finite pre-softmax logit (`max_logit`), the mean entropy in nats (`entropy`), and `entropy_share`, each query row's entropy divided by the log of that row's own visible key count, so uniform attention is 1.0 under a causal mask too. Rows are kept apart by kind, so padding and numerical failures do not read as concentrated attention: `masked_rows` (every key masked, as for padded queries) and `invalid_rows` (a NaN or +Inf score among the visible keys) are counted over batch × heads × queries (`rows`) and left out of the entropy, as are rows with a single visible key. Also sample 0's attention map over its valid rows, averaged over heads and pooled to at most 64 × 64 (keys summed and query rows averaged per bin, so each row sums to 1; rows with no valid query are 0).
- **`error`** instead of the above when the probe raised. The step itself is always written: the probe runs after the step is complete, and a failure is reported as a `RuntimeWarning`.

`probe` can be a tensor (`model(batch)`), a mapping of keyword inputs (`model(**batch)`), or a callable `fn(model)`. How it runs:

- **After the step's statistics are recorded.** The capture hooks fire during the probe's forward pass too, so their buffers are cleared again afterwards; the probe never shows up in the step's activations or router counts.
- **Without the compiler**, under `torch.compiler.set_stance("force_eager")`, in eval mode and without gradients. Each module's own training flag is restored afterwards (not `model.train()`, which would unfreeze submodules kept in eval mode), and the random number generator state is restored, so training sees the same random numbers as without the probe.
- **No recompiles.** The probe's hooks are registered once at `init` and do nothing unless a probe is running, which only happens without the compiler. Adding and removing hooks around each probe would recompile the model.
- **Attention is recomputed alongside the real call.** Fused attention kernels never store the attention map, so during the probe `torch.nn.functional.scaled_dot_product_attention` is wrapped: the original computes the output, and the wrapper recomputes `softmax(QKᵀ)` from the query and key a block of query rows at a time (at most 2^26 scores, 256 MB, per block). Grouped-query attention (`enable_gqa`, as Hugging Face uses on CUDA without a mask) repeats the key heads to the query's head count. Hugging Face models loaded with `attn_implementation="flash_attention_2"` or `"flash_attention_3"` never call SDPA, so the probe switches them to `"sdpa"` and back. Code that imported `scaled_dot_product_attention` by name before the probe, or uses another attention kernel, is not seen; the record then says `no scaled_dot_product_attention call seen`.
- **Deterministic mode** is turned off during the probe, because `histc` and `index_add_` have no deterministic CUDA kernels; the probe's results do not feed training.
- **Cost** is one eager forward pass plus the attention recomputation and one sync per histogram, every `probe_every` steps; not yet measured on a GPU. A Granite probe record should be on the order of 2 MB of JSON (24 maps of 64 × 64), also not measured. The Granite benchmark takes `--probe-every N`.

The Rerun export adds a Probe tab: the largest attention logit and `entropy_share` per layer over training, the count of invalid attention rows when there are any, a histogram per block and an attention map per layer at each probe step. There is no detector on the probe data yet: thresholds for logit growth or entropy collapse need real failures first.

## Things to know

- **Mixed precision.** Pass the run's scaler: `init(..., scaler=scaler)`. `torch.amp.GradScaler` skips `optimizer.step()` when gradients overflow, so the optimizer hooks alone would miss those steps; with `scaler=`, the step ends at `scaler.update()`, which runs every iteration, and each record gets `step_skipped` and `grad_scale` (the loss scale used), read as tensors before `update()` resets them, with no added sync. Gradient spikes and non-finite gradients are not reported on skipped steps, where the scaler expects them, and `loss_scale_trouble` reports 10 or more skips in a row or a loss scale below 1 (first guesses; a few skips in a row at the start, while the scaler halves its initial 65,536, are normal). Without `scaler=`, pass `auto_step=False` and call `session.step(lr=...)` once per iteration.
- **Resuming.** Pass `start_step=` the step you resume from. Otherwise step numbers start at 0 again, and the evidence reports them as repeated steps.
- **Clipping.** Gradient norms are read just before `optimizer.step()`, which is after clipping. The spike detector therefore uses the pre-clipping `clip_grad_norm` for the whole run whenever the run clips, so a spike that clipping flattened is still found; a NaN or Inf `clip_grad_norm` is reported as non-finite. Call `session.record_grads()` before you clip to record the unclipped norms.
- **Gradient accumulation.** A hook runs on every forward pass; the record keeps the last one before the optimizer step.
- **Non-finite values** are written as the strings `"nan"`, `"inf"` and `"-inf"`, so the file stays standard JSON.
- **Several optimizers on one model** (for example Muon for matrices and AdamW for the rest) count as one step per training iteration: the first optimizer that ever steps leads, its steps end TensorLens steps, and gradient norms are read before it steps, so they cover every optimizer's parameters. The others' steps add no step. This holds however each optimizer clears its gradients, but assumes the lead steps once every iteration; when it does not, pass `auto_step=False` and call `session.step()` at the iteration boundary. Lightning's `global_step` counts every optimizer step instead; to match it, pass `auto_step=False` and call `session.step(step=trainer.global_step)`. Optimizers that subclass another and call `super().step()` fire PyTorch's step hooks twice, nested; only the outer call counts, and a step that raised does not stop later ones from counting.
- **`optimizer.step(closure)`**, as Lightning steps, runs forward and backward inside the step, so gradient norms are read after the step instead of before it.
- **Activation checkpointing** recomputes forward passes during backward; the capture hooks skip those recomputations, so router counts and activation statistics count each token once.
- **The cross-GPU drift check is a screening check.** A parameter's float64 norm and sum miss changes that keep both, such as two elements swapping places; Megatron's `check_param_hashes_across_dp_replicas` compares hashes for exact equality. A NaN or Inf weight on any replica is reported as drift with `non_finite`.
- **Tensor parallelism.** By default the drift check compares every trainable parameter across `replica_group`, which is right for data parallelism. Under tensor parallelism each rank holds a different shard of most weights, so pass `replica_params=`: a `(name, param) -> bool` filter, or the parameters or their names. With Megatron, which marks sharded weights with `tensor_model_parallel`, use `replica_params=lambda name, p: not getattr(p, "tensor_model_parallel", False)`.
- **The cross-GPU drift check is a collective**, so every rank must run it on the same steps or the others wait until the process group times out. It runs where `(step + 1) % replica_every == 0`, from the step number, so the ranks must agree on step numbers: the same `start_step`, or the same `step=` passed to `session.step`.
- **The writer queue is bounded** at 8 blocks of `every` steps, as TensorBoard's `SummaryWriter` bounds its queue (`max_queue=10`). If the sink falls behind, for example on a stalled network filesystem, training waits for it with a one-time `RuntimeWarning` rather than holding unwritten steps in memory; no step is dropped.
- **Register your own forward hooks before `tensorlens.init`.** On PyTorch 2.10 with a Hugging Face Llama, a hook registered with `prepend=True` on a block TensorLens also hooks crashes `torch.compile` while it builds guards (`IndexError: tuple index out of range`), with or without `skip_nnmodule_hook_guards`. The same hook registered normally, or prepended on a block TensorLens does not hook, works. Not checked on PyTorch 2.14.
- **Hooks stay on the model after `close()`**, because removing a hook from a compiled module triggers a recompile.
- **Full-capture cost.** Each watched module adds a few small kernels per step. Granite's measured ~6% for activation capture was with every leaf module (171); blocks are the default now (`--layers all` in the benchmark for the old selection). Full capture with defaults measured +4.15% against capture off on the RTX PRO 6000 (Stage 3 above).

## Develop

Use PyTorch 2.14.1 in the local development environment. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.14.1' -e './sdk[dev,rerun]'
```

For CUDA dogfooding, use the training job's PyTorch build and CUDA environment.

```sh
cd sdk
../.venv/bin/python -m pytest
../.venv/bin/python examples/toy_moe.py --rerun
```

CI ([ci.yml](../.github/workflows/ci.yml)) runs the tests on every push and pull request, on Python 3.11 and 3.12 with PyTorch 2.6, 2.10 and 2.14 (CPU builds), and once more with the Lightning, Hugging Face and Rerun extras installed. It also runs `gpu_check.py --skip-overhead`, which on CPU uses the `aot_eager` backend. The real GPU gate is a separate job, started by hand from the Actions tab with `gpu` ticked; it needs a self-hosted runner with a CUDA build of PyTorch, and none is registered yet.

On a CUDA machine, run the GPU check. It fails on any graph break, recompile or CUDA-graph skip, and then reports the time per step with and without capture:

```sh
python examples/gpu_check.py --dim 1024 --blocks 8 --batch 4096
```

Set the sizes close to the real job, because a small model makes the fixed per-step cost look large.
