# TensorQuill capture SDK

Skeleton of the capture SDK from the [v3 plan](../TensorQuill%20v3%20—%20Product%20and%20Build%20Plan.md). It records what happens inside a PyTorch model during training, cheaply enough to leave on, and in a way that survives `torch.compile`.

```python
import tensorquill

model.to(device)
session = tensorquill.init(model, routers={"blocks.3.router": 128})  # before torch.compile
# Router load only, cheap enough to leave on. For debugging, add capture="full".
model = torch.compile(model)

for batch in loader:
    loss = model(batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()        # TensorQuill counts the step here

session.close()
```

To add TensorQuill to your own training job, follow [Run TensorQuill on your own job](RUN_ON_YOUR_JOB.md).

Each optimizer step becomes one JSON line in `tensorquill-out/<run_id>.rank<rank>.jsonl`.

There are two capture modes. `capture="routers"`, the default when `routers` are given, records only MoE router load (`moe`, `moe_stats`, `moe_top_k`) and the learning rate: this is the always-on mode. `capture="full"`, the default for models without routers, adds block activation statistics, gradient norms on every step and a weight check every 50 steps: this is the debugging mode. `layers`, `grads` and `weights` override either mode. See [Capture-mode decision](#capture-mode-decision) for the measurements behind it.

- **`layers`**: mean, standard deviation, absolute maximum and zero fraction of each watched module's output. By default the watched modules are the model's repeated blocks, such as each transformer decoder layer: the largest `ModuleList` of composite modules of one class, the unit PyTorch's FSDP `transformer_auto_wrap_policy` wraps. A model without such a list falls back to every leaf module with parameters, which `layers="all"` also selects; for Granite that is 171 modules against 24 blocks.
- **`grad_norms`** and **`grad_norm`**: the L2 norm of each parameter's gradient and their total.
- **`weight_trends`** (in `evidence.json`, from `build_evidence`): each kind of parameter's latest norm relative to its first recorded norm, as numbers only.
- **`weight_norms`** and **`weights_unchanged`**: each parameter's L2 norm after the optimizer step, and which parameters' norms are exactly unchanged since the previous check, plus `optimizer_ownership` (whether an optimizer that stepped holds each trainable parameter, or why that is unknown). Checked every 50 steps by default (`weights=N` for every N steps, `weights=False` to turn it off); a check also records that step's gradient norms.
- **`schema_version`** (1) and **`ts`**: the record format's version, and when the step ended, in Unix seconds.
- **`attempt`**: when the `tensorquill.init` that wrote the record started, in Unix milliseconds. A job that resumes from an earlier checkpoint under the same run ID writes some step numbers again; the attempt tells the new records from the ones they replace.
- **`lr`**: each parameter group's learning rate after the step.
- **`step_skipped`** and **`grad_scale`** with `init(..., scaler=)`: whether the loss scaler skipped this optimizer step on Inf or NaN gradients, and the loss scale used.
- **`clip_grad_norm`**: the total gradient norm before clipping, in any capture mode, when your training clips gradients. TensorQuill records what `torch.nn.utils.clip_grad_norm_` returns for this model's parameters while the session is open, which also covers Lightning's `gradient_clip_val` (Lightning calls that function and discards the result). It costs nothing extra: the norm is computed anyway and stays on the GPU until the next flush. Code that imported the function before `init`, or computes the norm elsewhere, can call `session.record_grad_norm(norm)`; with Megatron, pass the `grad_norm` that `optimizer.step()` returns. `clip_norm=False` turns the automatic recording off.
- **`moe`**, **`moe_stats`** and **`moe_top_k`**: tokens routed to each expert, plus router entropy, diversity (`exp(H)/N`), and dead-expert count/fraction, for every router you name (works whether the router outputs integer indices or floating-point logits with `Router(n_experts, top_k=...)`). Set `top_k` for top-k routers; the detectors need it.
- **`scalars`**: values you pass to `session.log`, such as `session.log({"train/loss": loss})`. A one-element tensor is copied on its device and moved to the CPU with the rest of the step data, so logging a CUDA loss adds no GPU sync, as Lightning keeps logged values on the device until it writes them. Call it before `optimizer.step()`, because with automatic step counting the step ends there. Pass the same values your tracker logs: TensorQuill does not capture the loss on its own, since patching `Tensor.backward` would miss Megatron (which calls `torch.autograd.backward` directly) and record the scaled loss under `GradScaler`. Every incident in the evidence carries the logged values at its step with the median of the previous 50, and the Rerun export plots them under a Logged tab.

## What is built

| Part | State |
| --- | --- |
| Forward hooks into preallocated on-device buffers, marked static | Built, tested on CPU (including AOTAutograd `static_input_indices` check) |
| On-device ring buffer, moved to the CPU every `every` steps, written by a worker thread | Built, tested |
| Step counting through PyTorch's global optimizer hooks, limited to this model's optimizer | Built, tested |
| Gradient norms grouped by device and dtype (`torch._foreach_norm`, with `DTensor.to_local()` and fallback) | Built, tested |
| MoE expert load (a compare-and-sum per router, accumulated on the device) and router entropy / diversity (`moe_stats`) | Built, tested |
| Hugging Face `Trainer` callback (`tensorquill.huggingface.TensorQuillCallback`): run ID from the tracker on the main process broadcast to every rank, steps from `state.global_step`, and the loss and pre-clipping gradient norm per optimizer step, taken by watching the `Accelerator` while a session is open. The same mechanism is public for custom Accelerate loops as `tensorquill.accelerator.watch_accelerator(session)`. Works with `torch_compile=True` | CPU tests on one process, for `Trainer` and for a plain Accelerate loop, with Transformers 5.23.0 / Accelerate 1.15.0 and their development versions. Not run on a GPU, on several ranks, with mixed precision, or with TRL |
| A one-time warning when 10 optimizer steps run and none is on the model's parameters, as with DeepSpeed ZeRO stage 1 or 2 and Megatron-LM, where automatic step counting would otherwise write nothing | CPU tests, and a DeepSpeed 0.20.1 engine on CPU at stages 0, 1 and 2 |
| PyTorch Lightning callback (`tensorquill.lightning.TensorQuillCallback`): run ID from the trainer's logger broadcast from rank 0, steps from `trainer.global_step`, the back-propagated loss per optimizer step, and the `16-mixed` scaler | Built, tested on CPU with Lightning 2.6.5 on one process and two (`ddp_spawn`) |
| Probe pass (`probe=`): activation histograms per block and attention logit, entropy and maps per layer, on a fixed batch every `probe_every` steps | Built, tested on CPU, under `torch.compile` with no graph breaks or recompiles, and on a small randomly initialized Hugging Face Granite MoE; not run on a GPU |
| Logged scalars (`session.log`) attached to incidents and plotted in Rerun | Built, tested on CPU and Apple MPS; not run on a GPU |
| Opt-in streaming (`endpoint=` or `TENSORQUILL_ENDPOINT`, `tensorquill/flight.py`): a daemon thread tails the rank's `.jsonl` file from `<run_id>.rank<rank>.offset` and sends it over Arrow Flight `DoPut` | Built, tested on CPU against a stand-in server and the real `tensorquill-server` on loopback: torn lines, offset resume, duplicate resend, a refused batch, a server killed for 200 steps, and `grpcs://` with `--tls-cert`/`--tls-key` and `TENSORQUILL_TLS_CA`. Not run to a remote host or from a GPU job |
| Ingest server (`../server`, binary `tensorquill-server`): `DoPut`, `(run_id, rank, attempt, step)` deduplication, 128 MiB buffer quota, Zstd Parquet parts through `object_store`, Bearer token, TLS | Built; `cargo test` passes on a local-disk store and TLS passes end-to-end on loopback. S3 and GCS have never run |
| Detectors on the server's Parquet: `load_records` and `python -m tensorquill` read a run's parts directory, one rank's, or one part, and give the same evidence as the source `.jsonl`; `--watch` re-reads a growing file or directory every 2 seconds (`--interval`) and replaces the evidence by rename; `--rank` (`rank=` in `build_evidence` and `load_records`) picks a rank and reads only that rank's folder; `build_evidence` on a directory with several ranks and no `rank=` raises | Built, tested on CPU, also through the real server binary |
| Cross-GPU parameter fingerprints and replica spread (`parameter_fingerprints`, `replica_spread`) | Built, unit-tested on drift and sign-flip cases |
| Detectors and Rerun exporter (`build_evidence`, `export_rerun`, `python -m tensorquill`) | Built, tested on MoE collapse (top-k aware), starved experts, gradient spikes, weights not updating, repeated steps, NaN/Inf, and replica drift |
| Run ID from `run_id=`, `MLFLOW_RUN_ID`, the active MLflow run or the active W&B run; otherwise torchrun's or SLURM's job ID, or a generated `run-<UTC time>-<6 hex>` broadcast from rank 0, so separate untracked runs never append to the same file | Built, tested with stand-in tracker modules and two Gloo processes |
| No graph breaks and no recompiles under `torch.compile` | Tested with `eager`, `aot_eager`, and AOTAutograd static-input checks on CPU; on CUDA with Inductor and CUDA graphs on a toy MoE, and with Inductor and CUDA graphs off on Granite |

## What is not built or not verified

- **Detectors on the server's Parquet, limits.** The reader takes a local path, so parts in S3 or GCS cannot be read yet. A directory holding several ranks is read one rank at a time, and only that rank's `rank=` folder is opened. On a `.jsonl` file, or on a store whose server did not report (`--no-detect`, or `--recompute` here), `--watch` reruns the detectors over the whole run on every update, so an update takes time in proportion to the run's length: 0.4 s at 500 steps and 8.1 s at 10,000 on a Granite-shaped 64-rank store (see [Detector benchmark on 64 ranks](#detector-benchmark-on-64-ranks)). On a store the server is judging it reads the server's report and no part (see [Watching a run the server is judging](#watching-a-run-the-server-is-judging)).
- **Streaming, not verified:** S3 and GCS stores, several ranks streaming at once, a remote server and a GPU job. A token from the Python sender, TLS on loopback (`--tls-cert`/`--tls-key` with `TENSORQUILL_TLS_CA`), a non-zero rank, and forked `DataLoader` workers while the sender runs each pass an end-to-end test on macOS and on Linux in CI (as of `f30e639`). In a forked child the sender does nothing: the fork copies the object but not its thread, so `close()` there returns at once and the parent keeps streaming. No deadlock was seen in either fork test, but gRPC's behaviour across `fork` differs by platform, so a Linux machine with a CUDA job is still worth checking. The sender's memory bound during an outage is measured with `tracemalloc`, which does not see gRPC's own allocations.
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

From the [failure list](../TensorQuill%20failure%20list.md), entries 3–6 and 12:

- **Top-1 and top-2 overload.** An expert's assignment share must exceed the largest of the configured base threshold (35% by default), 1.5 times an even share, and an even share plus three standard errors. Dead experts can trigger an incident independently and never lower the overload threshold.
- **Top-3 and higher overload.** Once a router has ten unflagged samples, its diversity must fall below 0.6 times the median of its last 50 unflagged samples, and a flagged expert's selection rate must rise by at least 25 percentage points. Leaving flagged samples out of the baseline keeps a persistent collapse visible. During warm-up, a router that already has significant dead experts can attribute overload to experts above both twice an even assignment share and an even share plus three standard errors. Each sample records the rule it used. Use `python -m tensorquill --top-k 8` for captures made before `moe_top_k` was recorded.
- **Weights not updating.** Every 50 steps by default, the capture computes every parameter's norm in float64 and compares it with the previous check; `weights_unchanged` lists the parameters whose norm is exactly the same. A parameter with a finite non-zero gradient whose norm is unchanged on 3 checks in a row (150 steps by default) is reported as a warning (equal norms do not prove the weights are equal), for example because it was left out of the optimizer. Since Nov 15, 2026 each weight check also records `optimizer_ownership`, and the incident reports whether an optimizer that stepped holds the stalled parameters: `known`, or `unknown` with a reason when ownership cannot be established (`auto_step=False`, DeepSpeed ZeRO, Megatron, FSDP1). Steps with a non-finite gradient norm are skipped, since mixed-precision loss scaling skips those on purpose. The first version compared float32 norms, and on the A100 run below it reported 147 of 218 parameters from step 1: a bf16 weight under a 1e-5 AdamW step changes in only a few elements per step, which a float32 norm often misses (`model.layers.7.self_attn.k_proj.weight` kept the same float32 norm for 128 steps while changing). Float64 norms resolve those changes. bf16 RMSNorm weights near 1.0 are still reported, correctly: their step size in bf16 is about 0.008, so a 1e-5 update rounds away, and in a CPU simulation such a weight was exactly unchanged on 273 of 650 steps. Each check reads every weight once, with no copy, accumulated in float64; on Apple MPS, which has no float64, the check is off. Captures made before `weights_unchanged` existed are not checked. The check runs every 50 steps rather than every step because the read alone costs about 1.5 ms for Granite's 2.6 GB of bf16 weights on the RTX PRO 6000, about 2% of a step, whatever arithmetic is used; with float64 on that GPU it measured about 3.7 ms. wandb.watch (every 1,000 batches by default) and TrainCheck (`sampling_interval`) sample their weight checks for the same reason. At every 50 steps the average cost is under 0.1 ms per step, not yet measured on a GPU.
- **Gradient spikes.** A step whose total gradient norm is at least 5× the median of the previous 50 recorded steps. The report ranks parameters by their own spike and names the one that started rising first.
- **Repeated steps.** Step numbers written twice by the same rank where that is not a resume from a checkpoint: within one attempt, or by a later attempt that starts the numbering over, as after a resume without `start_step`.
- **Activation spikes, drift and dead outputs** (full capture). Each watched module is compared with its own history, because trained LLMs carry large outliers by design: in healthy Granite the largest value in middle blocks is 740–840 times the standard deviation, so a fixed outlier threshold would fire at step 0. A spike is `abs_max` or `std` at 5× its median over the previous 50 records, and the incident lists the first module in the model first, then the rest by ratio: a change in one block's output travels down the residual stream, so every later block shows it and the largest ratio is in the last of them (a fault seeded in block 16 of 32 reached blocks 16 to 31, largest in 31), while the first module is where it showed first; drift is the recent median of `std`, or of the outlier ratio `abs_max / std`, 3× above or below the module's first 50 records; dead outputs are a zero fraction 0.25 above its baseline. On the healthy Granite capture (24 blocks, 650 steps) one-step jumps peaked at 2.2× and drift at 1.27×, and nothing was reported. A weight blown up 50× in one block of a test model was reported at that step. It judges the block's output as a whole, so growth confined to a few ordinary channels is invisible once the output holds massive activations: in SmolLM2-360M's block 16 (A100, Nov 15, 2026) `abs_max` is about 20,000 and `std` 31 to 37, and two ordinary channels times 100 moved `std` 1.07 to 1.14 times and `abs_max` 0.95 to 1.07 times although the loss went from 2.3 to 6.9. Growth of the massive channels themselves, or of the whole block, is seen (both are seeded by `--fault`, below). Per-channel statistics would see the rest; they are not captured, and what they would cost is not measured. Kurtosis, the outlier measure used in FP8 training research, is not captured: `abs_max / std` tracks the same outliers from statistics already recorded, while kurtosis would need a second pass over each module's output inside the compiled graph.
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

- **Measured cost on Granite, RTX PRO 6000:** router-only +0.02% to +0.78% across three runs; full capture with block defaults +4.15%, of which about 1.5 ms is gradient norms on every step and 0.85 ms block activation statistics. On the A100, router-only was +1.06% to +1.33% at one commit and +0.41% to +0.95% in a later profiling run (see "Where the A100 router-only overhead goes").
- **Why not sample gradient norms to bring full capture down:** it would still be about 2%, because the block statistics run inside the compiled graph on every step and cannot be sampled without extra compiled variants; and most gradient spikes last one step, so sampling would miss them.
- **What comparable tools do:** detailed per-layer monitoring is sampled or opt-in. Composer's `ActivationMonitor` runs every N batches and warns that it adds overhead; wandb.watch defaults to every 1,000 batches and its docs say logging every step can significantly slow training; Lightning documents per-layer `grad_norm` as a debugging step. Always-on profilers aim for about 1% or less.
- **Why routers are the right always-on signal for MoE models:** a 2026 paper on preemptive detection of training instability argues that loss, gradient norms and weight norms are late indicators and proposes router-level monitors such as routing entropy, which router-only capture records.
- **What router-only gives up by default:** layer statistics, per-parameter gradient norms, stalled weights and non-finite gradient detection. Turn on `capture="full"` while investigating.
- **Gradient spikes stay on in router-only mode when the run clips gradients.** Training loops that clip compute the total gradient norm on every step; TensorQuill records the value `clip_grad_norm_` returns (`clip_grad_norm`) and the spike detector uses it, without naming a parameter. PyTorch's `clip_grad_norm_` returns the pre-clipping norm as a tensor, Lightning's `Precision.clip_grad_by_norm` discards it, and Megatron's optimizer returns it from `step()`, which is why both automatic recording and `record_grad_norm` exist. Tested with Lightning 2.6.5's clipping path; not yet run on a GPU.

Sources: [Composer ActivationMonitor](https://mosaicml-composer.readthedocs-hosted.com/en/stable/_modules/composer/callbacks/activation_monitor.html), [wandb.watch](https://docs.wandb.ai/support/models/articles/how-do-i-log-gradients-and-model-weights-with-wandb-watch), [Pixie continuous profiling overhead](https://blog.px.dev/cpu-profiling-3), [Grafana continuous profiling costs](https://grafana.com/blog/continuous-profiling-in-production-a-real-world-example-to-measure-benefits-and-costs), [Mechanism-driven monitors for LLM training instability](https://arxiv.org/pdf/2606.28116), Lightning's `docs/source-pytorch/debug/debugging_intermediate.rst`.

## Where the A100 router-only overhead goes

**Run on the A100 Nov 15, 2026; the result is at the end of this section.** Router-only capture added about 2 ms per step on the A100 against 0.36–0.51 ms on the RTX PRO 6000 for the same hooks. A100 steps also take about 149 ms against 65 ms, although the model is small for either GPU, so both runs may be limited by the CPU launching kernels rather than by the GPU. Granite's `grouped_mm` expert path also copies expert offsets to the CPU, which syncs the CPU with the GPU and stops the CPU from running ahead; if that happens on every layer, every added launch adds to the step time.

[`examples/profile_overhead.py`](examples/profile_overhead.py) measures this. It times capture off and router-only like the benchmark, then profiles one 50-step window of each with `torch.profiler` and reports, per step, GPU busy and idle time, kernels, kernel launches, CPU syncs and the time spent waiting in them, the kernels capture added, and whether the added time is GPU work or host time. In Colab on the A100, from the repository root:

```sh
cd sdk
python -m pip install -e '.[rerun]' transformers huggingface_hub pyarrow
python examples/profile_overhead.py --out runs/profile-a100
```

Add `--capture full` to profile full capture, and `--trace` for a Chrome trace of each arm (open it at `chrome://tracing` or ui.perfetto.dev; it can be large). The same command on the RTX PRO 6000 gives the comparison. `--toy` checks the script on CPU without downloads.

**Result** (Colab, A100-SXM4-80GB, PyTorch `2.14.1+cu130`, Inductor with CUDA graphs off, commit `d987509`, Granite). Three paired rounds of 100 steps after 50 warmup steps: capture off 145.28, 145.27 and 145.22 ms; `capture="routers"` 146.66, 146.44 and 145.82 ms. That is +0.95%, +0.80% and +0.41%, all under 1%, against +1.06% to +1.33% at `9f88e50` with steps of 149 ms. Why the earlier run was higher is not known: the code and the Colab host both changed. The profile says the added time is GPU work, not the host waiting: GPU busy rose 1.81 ms per step, kernel launches rose from 7,334 to 7,360, and CPU syncs did not change (144.04 against 144.06 per step; the 144 are in the model's own step, with capture off too). The largest added kernel, 24 times per step (once per layer) at about 40 µs each, is a fused `index_mul_sum` (the assignment counts), 968 µs per step in all. This is not the Gate 2 protocol (three paired rounds of 200 steps from `moe_dogfood.py`), so the Gate 2 overhead item is not ticked on it.

## Dense model with a seeded activation outlier

Everything before this ran on Granite, an MoE model; the A100 result is after the run command. [`examples/dense_outlier_check.py`](examples/dense_outlier_check.py) checks the dense-model path: it trains SmolLM2-360M (Llama architecture, grouped-query attention) on WikiText-2 with capture off and with full capture plus the probe, in alternating 50-step chunks, and from step 250 scales part of the middle block's output by 100 (`--fault`): by default `massive`, the two channels with the largest outputs on a calibration batch, the massive activations growing further; or `block`, the whole output; or `channels`, the two named by `--channels` (default 7 and 300), which a block summary cannot see on this model (the `Trainer` run below). It reports:

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

**A100 result** (Colab, A100-SXM4-80GB, PyTorch `2.14.1+cu130`, SmolLM2-360M in bf16, Inductor with CUDA graphs off, commit `d987509`, default `--fault massive`: channels 87 and 398 of block 16, times 100 from step 250):

- **`PASS`.** The first activation spike is at step 250 (`abs_max` 736× its recent median), with drift from step 254 (`std` 46× its early level). Both cover 16 modules, blocks 16 to 31: the fault travels down the residual stream. On this run the spike listed them by ratio, so it led with block 31 and the seeded block 16 came last of the 16; fixed the same day (the incident now leads with the first module in the model, block 16, and names block 31 as the largest, abs_max 736×). 0 graph breaks.
- **Overhead.** Capture off 49.45 ms per step; full capture plus probe 50.91 ms (+1.46 ms, +2.94%). A flush step (every 50) costs +4.39 ms over a plain step, and a probe step (with its flush) +169.9 ms over a flush step: 1.70 ms per step on average (3.44%) when probing every 100 steps, 0.17 ms per step (0.34%) at the default 1,000.
- **Probe.** The largest value in block 16 went from 21,632 (step 199) to 2,162,688 (step 299), 100.0×. In block 17, the first to read it, the attention's largest logit fell from 12.09 to 0.91 and its entropy share rose from 0.34 to 1.00, attention gone uniform (0 invalid rows). The histogram PNG shows two million values at zero and a handful out at ±1 to 2 million.
- **Also reported, not seeded.** 28 of 290 parameters had an unchanged float64 norm on 3 checks from step 199, the RMSNorm weights, which fits the frozen bf16 norm weights found on Granite (bf16 weights, learning rate 1e-5); an optimizer holds all of them.
- **Size.** `dense.jsonl` is 15.7 MB for 400 steps at full capture with 4 probes (about 39 KB a step). The Parquet size was not measured.
- **A warning that is not a fault.** `TensorQuill has recorded no steps: 10 optimizer steps ran, none on this model's parameters` prints at the start. The script's capture-off model steps first, so the first optimizer steps are not the captured model's; `profile_overhead.py` does the same.


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

The check passes `replica_every=1`. In a training job the drift check runs every 50 steps by default (`replica_every=`), as Megatron's `check_weight_hash_across_dp_replicas_interval` does: each check reads every weight in float64 and gathers 24 bytes per parameter tensor from every rank, and a float64 norm-and-sum pass over Granite's weights cost about 3.7 ms on the RTX PRO 6000 (measured before the projection was added). The seeded projection (Nov 15, 2026) took about seven times the norm and sum on CPU (3.7 s against 0.5 s for 24 bf16 4096×4096 tensors; peak extra memory about 36 MB, measured in a fresh process); its GPU cost is not measured. `replica_projection=False` drops it. Until Nov 14, 2026 it ran on every step.

## Probe pass

`tensorquill.init(model, probe={"input_ids": fixed_batch}, probe_every=1000)` runs one fixed batch through the model every 1,000 steps (wandb.watch's default for histograms) and adds a `probe` field to that step's record:

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

## Streaming to an ingest server

The local `.jsonl` file is always written and stays the source of truth. Streaming is off unless you set an endpoint, and then needs `pyarrow`:

```sh
pip install -e 'sdk[flight]'
cargo build --release --manifest-path server/Cargo.toml
server/target/release/tensorquill-server --listen 127.0.0.1:50051 --store file:///var/lib/tensorquill
```

**Evaluation and checkpoints are not hangs.** With `endpoint=`, a thread of its own sends the server a heartbeat every 15 seconds (`init(heartbeat_seconds=...)`, `None` turns it off) naming the rank's phase. Mark a stretch with no training steps with `with session.phase("eval"): ...` (or `session.set_phase("checkpoint")` and back to `"train"`); the Lightning callback marks validation and the Hugging Face callback marks evaluation and saves. While the phase is not `train` and heartbeats continue, the rank is `paused` and raises nothing. A rank that has not sent its first batch yet is `starting`, not stopped. A pause is bounded: heartbeats carry a work counter (steps plus `session.tick()`, which the callbacks call once per evaluation batch), a pause that keeps ticking is not limited, one with no sign of work is suspect after `--pause-seconds` (default 1800), and a committed training step always ends a pause. A quiet rank in `train` is `silent` (`suspected_hang`; `confirmed_hang` only when a Flight Recorder dump shows a stuck collective), `warning` after `--silent-seconds` and `critical` after `--critical-seconds` (default 900) or three of its own silence thresholds, if longer, so a slow rank still gets a warning first. `confirmed_hang` needs a complete Flight Recorder dump newer than the last recorded step. If the heartbeats stop too, the rank is `unresponsive`: the whole process is frozen or gone. The server records how long each pause lasted and `watchdog.json` carries `recommended_silent_seconds` (twice the longest pause); the summary says it. TensorQuill only reports: it never blocks or terminates training. All thresholds are uncalibrated starting values. A stuck evaluation that keeps its heartbeats going raises nothing, and a hung main thread keeps the heartbeats going too, so a heartbeat shows the process is alive, not that it is making progress.

**TrainCheck** (OSDI '25, a separate tool that checks a run against invariants inferred from a clean reference run) is read, not embedded: `python -m tensorquill <input> --traincheck <violations_summary.json or the traincheck-check results directory>` turns its violated invariants into one `traincheck_violation` incident beside TensorQuill's own. Its steps are its own iteration count, which match TensorQuill's only without gradient accumulation; running both tools in one process has not been tried, since each patches PyTorch's API.

**When a multi-GPU job hangs**, PyTorch's Flight Recorder already knows which collective is stuck; TensorQuill reads its dumps instead of tracing collectives. Set `TORCH_FR_BUFFER_SIZE` (formerly `TORCH_NCCL_TRACE_BUFFER_SIZE`; the default is 2000 entries), `TORCH_FR_DUMP_TEMP_FILE=/some/dir/fr_trace_rank_` (formerly `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`) and, with NCCL, `TORCH_NCCL_ENABLE_MONITORING=1` and `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, so each rank writes `<prefix><rank>` when a collective times out. Then `python -m tensorquill <input> --flight-recorder /some/dir` (add `--flight-recorder-prefix fr_trace_rank_` if PyTorch cannot infer it) runs PyTorch's analyzer over the dumps and reports the first stuck or mismatched collective as `collective_hang` or `collective_desync`. PyTorch 2.10 or later; checked only on dumps built in the analyzer's own test format. The dumps are pickles, so read only dumps your own job wrote. PyTorch's debug server (`torch.distributed.debug.start_debug_server`) is experimental and meant for trusted networks only, so TensorQuill does not start it.

The server also watches each run for ranks that stop making progress, using only what it receives, so it works when the training process is hung. It looks at each run every 10 seconds (`--watchdog-seconds`, 0 turns it off) and writes `runs/<run_id>/watchdog.json` when the verdict changed, and at least every third look (`refresh_seconds` in the file, 30 by default) while the run is unfinished, so that a reader can tell a server watching a stalled run from one that is gone; `python -m tensorquill <store>/runs/<run_id>` reads it and reports `job_stalled` (no rank progressing) or `rank_silent` (some ranks silent, lost or never connected). A rank is silent after `--silent-seconds` without a new step (default 300) or `--silent-factor` times its own usual time between batches (default 5), whichever is longer. Set it above your longest evaluation or checkpoint pause: the default is not calibrated, and a long pause looks the same as a hang. The server sees only ranks that connected since it started, and it names a rank as never connected only after `--silent-seconds` (ranks start a moment apart), so capture on every rank. It rewrites the file only when something changed and marks it `server_stopped` when it shuts down cleanly.

```python
session = tensorquill.init(model, routers=..., endpoint="grpc://127.0.0.1:50051")  # or TENSORQUILL_ENDPOINT
```

- **It does not touch the training thread.** A daemon thread reads complete lines from the file, at most 50 steps or 4 MiB at a time, every 50 steps or 2 seconds. A line the writer has not finished is never sent.
- **A server that is down, slow or full only delays the stream.** The sender keeps no backlog in memory: it sleeps (100 ms doubling to 30 s, with full jitter) and reads the file again from `<run_id>.rank<rank>.offset`. That file moves only when the server confirms the bytes are in a stored Parquet part, so it trails the `.jsonl` by up to 10 seconds or 500 steps, and a server killed with rows still in memory loses nothing.
- **Resending is harmless.** The server drops rows whose step is not above the last one it kept for that `(run_id, rank, attempt)`, and reads that mark from its Parquet files after a restart.
- **`session.close()` waits up to 5 seconds** for the server to store what is left, and otherwise prints the local path. `tensorquill.flight.send_file(path, endpoint, run_id, rank)` sends the rest later.
- **Errors in the setup raise at `init`:** an endpoint that is not `grpc://` or `grpcs://`, a missing `pyarrow`, an unreadable `TENSORQUILL_TLS_CA` file under `grpcs://`, or `endpoint=` with a sink that has no file. Nothing raises once training runs; `session._sink.sender.last_error` holds the last failure.
- **The server** writes `{store}/runs/{run_id}/rank={rank}/attempt={attempt}/part-{first step}-{last step}.parquet`, one row per step. It listens on loopback by default; any other address needs `TENSORQUILL_TOKEN` (the sender reads the same variable and sends it as a Bearer token) or `--allow-unauthenticated`, with optional TLS (`--tls-cert` and `--tls-key` on the server, `grpcs://` and `TENSORQUILL_TLS_CA` for a custom CA on the sender). On `SIGTERM` it writes what it holds and exits. It buffers at most 128 MiB across all ranks and refuses batches beyond that, which the sender retries.
- **Run IDs** may hold only letters, digits, `-`, `_` and `.` on the server, because the ID becomes a directory name.
- **A run resumed from an earlier step under the same run ID** is a new attempt. The server checks each row against its own attempt, so the steps written again are kept, in their own `attempt=` directory, and a resend of old lines after a lost offset file still stores nothing twice.
- **Columns.** Numbers and name-to-number maps are typed columns; `layers`, `moe_stats` and `probe` are JSON strings (`layers_json`, `moe_stats_json`, `probe_json`); anything else travels in `extra_json`, so a row converts back to exactly the record in the `.jsonl` file. The choices behind this are in the plan's Decisions section.

## Detector benchmark on 64 ranks

[`examples/benchmark_detectors.py`](examples/benchmark_detectors.py) times the Python detectors against the plan's budget of 250 ms per 50-step flush across 64 ranks (or one CPU core). It trains a Granite-shaped MoE (24 blocks, 32 experts, top-8, 219 parameters) for 105 steps under `tensorquill.init`, takes the record shapes from that capture, writes a healthy synthetic run of 64 ranks and 10,000 steps in the server's layout (50-step Zstd parts, 3 GB), and times `load_records`, `build_evidence` and one `--watch` update at 500, 2,000 and 10,000 steps, then profiles the slowest stage. `python examples/benchmark_detectors.py` runs all of it in about 6 minutes (384 s on the machine below, building the store included) with no GPU or server (`--quick`: 4 ranks, under a minute; `--json` keeps the numbers).

Measured on an Apple M1 Pro (8 cores, 16 GB, macOS, no GPU), Python 3.12.3, pyarrow 26.0.1, with other work on the machine (load average 3 to 6): the update is one flush of 50 steps from 64 ranks for rank 0, median of five runs, warm file cache.

| Run length | Poll | Detectors | Write | `--watch` update (worst) | Against 250 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 steps | 13 ms | 341 ms | 42 ms | 395 ms (402) | 1.6x |
| 2,000 steps | 15 ms | 1,450 ms | 167 ms | 1,631 ms (1,664) | 6.5x |
| 10,000 steps | 21 ms | 7,169 ms | 860 ms | 8,067 ms (8,156) | 32x |

The rebuild costs about 0.72 ms of detector time per step, so an update grows with the run, and one worker per rank over 64 ranks at 2,000 steps takes 105 s (3,200% of a core at Granite's 65 ms per step). A worker that kept the detectors' state between flushes would still pay one part read (13.6 ms) and 50 steps of detectors (35.9 ms) per rank: 3.2 s for 64 ranks, 12.7x the budget and 105% of a core, projected from the slope. The time is spread over per-step loops with rolling medians (activation checks 27%, gradient spikes 15%, the starved-expert window 15%, non-finite scan 13%, top-k overload 12% of a 2,000-step build), not one hot spot. Peak memory is 1.15 GB for one rank's 10,000 steps and 1.28 GB for the update. A 2-block, 8-expert, top-1 model (21 parameters) is 17 times cheaper per step: an update at 2,000 steps takes 100 ms, and the stateful floor for 64 ranks is 343 ms (1.4x, 17% of a core). The values are synthetic and healthy and the run is one process on one machine, so treat the figures as the shape of the cost, not as a forecast for a GPU job.

Two fixes came out of it, with identical output on 29 healthy and faulted runs, except for a store whose `rank=<n>` folder holds another rank's rows, which the server never writes. Reading `--rank N` from a directory of ranks opened every rank's parts and kept one in 64: the poll took 1.0 to 1.5 s, and the whole directory sat in memory at once (495 MB for 64 ranks of 100 steps, about 50 GB at 10,000 steps); it now opens only that rank's folder (poll 13 to 21 ms, 32 MB). The activation drift check now computes each module's baseline once, taking the 10,000-step build from 9.2 s to 7.2 s.

## Detectors in the server

`tensorquill-server` runs the detectors itself, in Rust: every kind that is read from a rank's step records (`server/src/detect.rs`), and the two that are read from the progress status (`server/src/watchdog.rs`). Python (`evidence.py`) stays the reference: new rules land there first, and the server must give the same answer.

| Where | Kinds |
| --- | --- |
| `runs/<run_id>/rank=<rank>/incidents.json` | `moe_imbalance`, `starved_experts`, `grad_spike`, `stalled_weights`, `repeated_steps`, `loss_scale_trouble`, `resume_lr_change`, `activation_spike`, `activation_drift`, `dead_activations`, `non_finite`, `replica_drift` |
| `incidents` in `runs/<run_id>/watchdog.json` | `job_stalled`, `rank_silent` |
| Python only | `collective_hang`, `collective_desync`, `traincheck_violation` |

The last three stay in Python because their sources never reach the server: PyTorch's Flight Recorder dumps are pickles on the training nodes, read by PyTorch's own analyzer, and TrainCheck writes its results where it ran.

- **What it writes.** After each part is stored and acknowledged, the server reads that part's rows and rewrites the rank's `incidents.json`: the incidents in the shape and order `build_evidence` gives them, `last_step` (detection has run through here), the kinds `checked`, `not_checked` (in the shape of the same list in `evidence.json`: the two sources the server does not have, and each resume whose learning rate could not be compared), the thresholds applied, `moe_flagged` (how many router samples had an overloaded or dead expert and the last step one did, which says whether an imbalance is still going on), and `rebuilds`. It holds no per-step samples: those stay in the Parquet parts, so the file stays a few KB however long the run is. `--no-detect` turns it off. The watchdog status carries its verdict as an `incidents` list as well, rewritten whenever the status changes, so a stall shows there although no part arrives. `python -m tensorquill --watch` reads both (see [Watching a run the server is judging](#watching-a-run-the-server-is-judging)); without `--watch` the command works the incidents out itself from the parts.
- **Only stored rows are judged**, so the file can always be rebuilt from the store. After a server restart, on every new attempt, and whenever an older attempt's rows arrive late, the rank's state is rebuilt from all of its stored parts with the same attempt rules as `load_records`. That replay also reads the rows a later attempt rolled back: `repeated_steps` and `resume_lr_change` compare a new attempt with what it replaced. The step numbers and learning rates it needs for that are held only while the replay runs. A detector error or panic is logged and never fails a flush.
- **How late it is.** Detection follows storage, and a part is written every `--flush-steps` steps (500 by default) or `--flush-seconds` (10), so by default an incident shows about 10 seconds after its step reached the server, at the latest.
- **Same answers as Python.** [`tests/test_server_detectors.py`](tests/test_server_detectors.py) (9 tests) streams records through the real binary and compares `incidents.json` with `build_evidence` on the same records: after 16 stretches of a 300-step run in which ten kinds each happen once; across a server restart in the middle of a 140-step collapse; after each of 12 stretches of a run with nine attempts (resumes at a step the run had, past the stored steps, with no learning rate, and two restarts of the numbering); on seven short runs for rules the long one does not reach; and on the seven Gate 1 fixtures. `test_flight_e2e.py` and `test_two_ranks_e2e.py` compare the status's `incidents` with `detect_watchdog_incidents` on a stalled job, a killed one, a lost rank and a rank that never connected. Twenty-two deliberate changes to the Rust rules each fail these tests. Steps, layers, experts and names match exactly and numbers to 1e-12.
- **Where the server differs, by design.** NaN and Inf are `null` in its files, which are strict JSON. A value that does not fit its column's type travels in `extra_json` and reads as absent. The server never stores a step twice within one attempt, so that case of `repeated_steps` cannot arise from its store. A stream cannot look ahead as Python does over a finished list; the places that matters (when clipping starts, a `moe_top_k` that changes in mid-run, and attempts) are described at the top of `detect.rs`.

Timed with `cargo run --release --example detect_store -- <store> <run_id> --bench` on the benchmark's own synthetic store (64 ranks, 2,000 steps, Granite's shape, healthy), Apple M1 Pro, one thread, median and worst of 40 flushes of 50 steps from every rank:

| Per flush of 64 ranks | Rust, twelve kinds | Python, all kinds |
| --- | ---: | ---: |
| Detectors | 56 ms (68) | 2,220 ms, projected with state kept |
| of which decoding `layers_json` and `moe_stats_json` | about 33 ms, measured apart | |
| Writing the 64 Parquet parts (already on the live path) | 138 to 145 ms | |
| Writing 64 `incidents.json` files | 13 ms (14) | |
| Reading the parts back | not needed live | 830 ms |

The first six kinds took 54 ms; the six added on Nov 15, 2026 cost about 2 ms more, not the 26 ms guessed from Python's proportions. That is 17.5 µs of detector time per step against Python's 693 µs. Detection, storage and the incidents files together are about 210 ms on one thread, under the 250 ms budget on this healthy run (about 290 ms, over it, if no shortcut applied: see below); the server spreads ranks over its threads, which this does not measure. Rebuilding one rank from 2,000 stored steps takes 75 ms, and all 64 at once 0.98 s.

What these numbers do not show:
- The store is healthy, and most of the speed comes from skipping a median when a window's lowest or highest value already rules a hit out, which is exact. Before those shortcuts the first six kinds took 136 ms per flush; take that as the cost when every check has to work its median out, as on a run that is failing everywhere.
- Two thirds of the detector time is decoding the two JSON columns. Typed columns for the per-layer statistics would remove most of it; that is a schema change and is not made.
- The store has a weight check and a replica check once in 50 steps and no loss scaler. A run that checks weights on every step pays for 219 name comparisons on each; that was not timed.
- The detectors' state is outside the server's memory quota and is kept until the server exits: 1.5 to 2 MB per rank at Granite's shape, measured under live streams below. A replay also holds every step number and learning rate of the rank while it runs (under 100 bytes a step, by its layout).
- A real model's values, a GPU job, and S3 or GCS are not measured.

### Under 64 live streams

[`server/examples/live_streams.rs`](../server/examples/live_streams.rs) opens one `DoPut` stream per rank against a running release server and replays the same store, one 50-step part per message, each rank waiting for its acknowledgement as the Python sender does (`--flush-steps 50`, so every message is one flush; all 64 ranks start together, so their flushes coincide, which is the worst case). Same machine (8 cores), with the client on it too, decoding Parquet and encoding Flight beside the server it times.

| 64 ranks, 2,000 steps each | Detectors on | `--no-detect` |
| --- | ---: | ---: |
| As fast as acknowledgements return | 3.4 to 4.0 s, 32,000 to 37,800 steps/s (18 runs) | 2.7 to 2.9 s, 44,700 to 48,000 steps/s (3 runs) |
| At 10 ms per step: sent to acknowledged, median / p99 / worst | 52 / 99 / 125 ms | 41 / 81 / 92 ms |
| At 65 ms per step (Granite's): the same | 54 / 100 / 132 ms | not run |
| Acknowledged to `incidents.json` covering the part, median / p99 / worst | 0.1 / 15 to 25 / 50 ms | |
| Server CPU for the run (2,560 flushes) | 14.3 s at 10 ms per step, 15.2 s at 65 ms | 10.1 s |
| Server peak memory | 360 to 415 MB | 260 to 300 MB |

- **Rate.** 64 ranks at Granite's 65 ms per step produce 985 steps a second; the server took 35 times that with the detectors on. At that pace it used 15.2 s of CPU in 130 s, 12% of one core.
- **The budget.** Every one of the 64 simultaneous flushes was stored and acknowledged within 132 ms, under 250 ms. As CPU it is over: 5.6 ms per rank and flush, 358 ms for a flush of all 64, of which the detectors and the incidents files are about 105 ms (1.6 ms per rank; 69 ms when timed alone on one thread above) and receiving, decoding and storing the rest (253 ms with `--no-detect`). The single-thread table above leaves out receiving and decoding the Flight messages.
- **Memory.** The detectors add roughly 100 MB for 64 ranks, 1.5 to 2 MB a rank, measured as the difference in peak memory between the two columns; the estimate from window sizes was 1 MB. It is outside the 128 MiB quota.
- **10,000 steps.** 64 ranks × 10,000 steps sent flat out took 18.3 s (35,000 steps/s, the same rate as at 2,000), acknowledgements 73 / 218 / 1,027 ms, the report 0.1 / 6.6 / 77 ms behind, peak memory 448 MB.
- **A restart with 10,000 steps stored.** Each rank's first part after the restart was stored and acknowledged in 100 to 170 ms at the median (442 worst, four runs). The server then rebuilt all 64 ranks from the store at once: the reports were current again 6.1 to 7.0 s later, and each rank's next part waited that long for its acknowledgement (6.2 to 7.1 s), because a rank's parts are taken one at a time. Nothing is lost while it waits: the sender keeps the lines in its file. The rebuild used 29 to 32 s of CPU over 8 cores, 0.5 s a rank, and peak memory was 400 to 535 MB.
- **A new attempt on all 64 ranks at once** (no restart) cost the same rebuild: 6.2 to 6.8 s (two runs).
- **Same answers.** After five restarts and three new attempts (10,800 stored steps, four attempts), the reports for the two ranks compared (5 and 40) equal `build_evidence` on the same store, one `resume_lr_change` each. A watcher's look at such a rank (216 part names listed, the report and the status read, the evidence written) took 2.0 ms (worst 4.3 of 50); `build_evidence` on it took 12.0 s.
- **One stream dropped once.** In the first of 19 flat-out runs with detectors on, made while the machine was still busy from a build, one rank's stream ended with a broken pipe 6 s in and the server logged nothing. It did not happen again and is not explained. A real sender reconnects and resumes from its offset file; this client does not. It was not the memory quota: a quota refusal reaches the client as `ResourceExhausted` ("ingest buffer full"), which a run with `--quota-mib 32` shows.
- **The quota at 64 ranks.** One 50-step part of this store is 1.4 MB as Arrow, so 64 ranks' parts arriving together hold 85 MiB of the default 128 MiB quota. With these record sizes about 90 ranks flushing at the same instant would fill it; the server then writes a rank's buffer early, and refuses the batch only if that is not enough. Not tried beyond 64.
- **Not covered:** more than 64 ranks, a second machine between client and server, TLS, S3 or GCS, a run where incidents happen (the store is healthy until the resumes), and rebuilds longer than 10,800 steps. By the slope, 64 ranks of 100,000 steps would take about a minute to rebuild, longer than the watcher's default 30-second wait; `--report-grace` raises it.

### Watching a run the server is judging

`python -m tensorquill <store>/runs/<run_id> --watch` takes its incidents from the server while it can: the rank's `incidents.json` for the twelve kinds read from step records, and the `incidents` list in `watchdog.json` for `job_stalled` and `rank_silent`. It reads no Parquet part then: a look lists the rank's part names and reads two small files, 2 ms with 216 parts (10,800 steps), where running the detectors here took 12 s. `--flight-recorder` and `--traincheck` are still read here and added, in the order `build_evidence` gives.

An empty `incidents` list means "no incidents" only in a report that covers what is stored. The report carries no clock, so the watcher compares it with the store: `read_server_report` takes the report's `attempt` and `last_step` and the newest stored part's, which are in that part's path (`attempt=<a>/part-<first>-<last>.parquet`), and calls the report:

| State | Meaning | What the watcher does |
| --- | --- | --- |
| `current` | The report ends exactly at the newest stored step of the newest attempt, covers all twelve kinds, and used the thresholds asked for | Uses it. "No incidents detected" is followed by "current through step N of attempt A (M steps judged)" |
| `behind` | The report stops before the newest part. Normal for a moment after every part (0.1 ms at the median, 77 ms at worst, measured), and for 6 to 7 s while a restarted server rebuilds 64 ranks of 10,000 steps | Waits, up to 30 seconds (`--report-grace`) since the report was last current; then the report is `stale` |
| `stale` | Still behind after that wait: the server was restarted with `--no-detect`, its detectors failed for this rank, or it died between storing a part and reporting on it | Reads the parts and runs the detectors here, and says the report stops at step X while the store holds step Y |
| `missing` | Parts, and no report: `--no-detect`, or a server from before its detectors | The same, at once if no report was ever seen |
| `unusable` | A report for another run or rank, from a server with fewer kinds, ahead of the store, or beside parts whose names carry no steps | The same, at once |
| `other_thresholds` | `--overload-threshold` or `--top-k` here differ from what the server applied | The same, at once |
| `unreadable` | The file could not be read or parsed on this look | Waits like `behind` |

Each evidence file says which happened under `detection` (`source` is `server` or `python`, with `state`, `reason`, and the steps and attempts compared), and the summary's last line says it in words. Evidence built from a report has no `samples`, `weight_trends`, `layer_count` or `router_names`: per-step data stays in the parts. `--recompute` ignores the report and gives them, at the cost above.

After a restart, specifically:

- **A clean stop or a kill, then a new server.** Every part the old server acknowledged was judged before it went on, so the report it left is still `current` for what is stored, and stays so until a rank sends again. The new server then rebuilds that rank from the store and the report moves on (`rebuilds` starts again at 1).
- **A kill between storing a part and reporting on it** leaves a report that is `behind`. A watcher that starts then works the incidents out itself; one that was already running waits its 30 seconds first. The next part the new server stores brings the report up to date.
- **The status file.** A killed server leaves its last `watchdog.json`, which may say `active`. A running server rewrites an unfinished run's status at least every `refresh_seconds` (30 by default) whether or not anything changed, so a stall that lasts an hour still has a fresh status. When the status claims a running server and is older than twice that, the summary says the status was last written N seconds ago and that nothing is watching the run for stalls; the same holds after a restart until a rank of the run reconnects, because the new server does not know the run yet. After a clean stop the status says `server_stopped`, and the summary says so. Neither is printed as a bare "No incidents detected."

Limits: the watcher judges freshness by the newest attempt only, so a part of an older attempt that arrived late and was not judged goes unnoticed; a NaN or Inf logged value prints as "not finite" (it is `null` in the server's file); `job_stalled` and `rank_silent` carry no logged scalars when they come from the status; and one rank is watched per command, as before. [`tests/test_server_reports.py`](tests/test_server_reports.py) (12 tests) produces each state with the real server. Twenty-two deliberate changes to the reader's rules each fail it.

## Things to know

- **Mixed precision.** Pass the run's scaler: `init(..., scaler=scaler)`. `torch.amp.GradScaler` skips `optimizer.step()` when gradients overflow, so the optimizer hooks alone would miss those steps; with `scaler=`, the step ends at `scaler.update()`, which runs every iteration, and each record gets `step_skipped` and `grad_scale` (the loss scale used), read as tensors before `update()` resets them, with no added sync. Gradient spikes and non-finite gradients are not reported on skipped steps, where the scaler expects them, and `loss_scale_trouble` reports 10 or more skips in a row or a loss scale below 1 (first guesses; a few skips in a row at the start, while the scaler halves its initial 65,536, are normal). Without `scaler=`, pass `auto_step=False` and call `session.step(lr=...)` once per iteration.
- **Resuming.** Pass `start_step=` the step you resume from. Otherwise step numbers start at 0 again, and the evidence reports them as repeated steps. When you resume from a checkpoint earlier than the last recorded step, the new attempt's records replace what earlier attempts recorded from that step on, in the evidence and in every reader; the file and the server's store keep both, and `load_records(..., all_attempts=True)` returns both.
- **Clipping.** Gradient norms are read just before `optimizer.step()`, which is after clipping. The spike detector therefore uses the pre-clipping `clip_grad_norm` for the whole run whenever the run clips, so a spike that clipping flattened is still found; a NaN or Inf `clip_grad_norm` is reported as non-finite. Call `session.record_grads()` before you clip to record the unclipped norms.
- **Gradient accumulation.** A hook runs on every forward pass; the record keeps the last one before the optimizer step.
- **Non-finite values** are written as the strings `"nan"`, `"inf"` and `"-inf"`, so the file stays standard JSON.
- **Several optimizers on one model** (for example Muon for matrices and AdamW for the rest) count as one step per training iteration: the first optimizer that ever steps leads, its steps end TensorQuill steps, and gradient norms are read before it steps, so they cover every optimizer's parameters. The others' steps add no step. This holds however each optimizer clears its gradients, but assumes the lead steps once every iteration; when it does not, pass `auto_step=False` and call `session.step()` at the iteration boundary. Lightning's `global_step` counts every optimizer step instead; to match it, pass `auto_step=False` and call `session.step(step=trainer.global_step)`. Optimizers that subclass another and call `super().step()` fire PyTorch's step hooks twice, nested; only the outer call counts, and a step that raised does not stop later ones from counting.
- **`optimizer.step(closure)`**, as Lightning steps, runs forward and backward inside the step, so gradient norms are read after the step instead of before it.
- **Activation checkpointing** recomputes forward passes during backward; the capture hooks skip those recomputations, so router counts and activation statistics count each token once.
- **The cross-GPU drift check is a screening check.** A parameter's float64 norm and sum miss changes that keep both, such as two elements swapping places. The seeded projection added Nov 15, 2026 catches that swap, but it is still a screen: a change that keeps all three values is possible, just far less likely. Megatron's `check_param_hashes_across_dp_replicas` compares hashes for exact equality. A NaN or Inf weight on any replica is reported as drift with `non_finite`.
- **Tensor parallelism.** By default the drift check compares every trainable parameter across `replica_group`, which is right for data parallelism. Under tensor parallelism each rank holds a different shard of most weights, so pass `replica_params=`: a `(name, param) -> bool` filter, or the parameters or their names. With Megatron, which marks sharded weights with `tensor_model_parallel`, use `replica_params=lambda name, p: not getattr(p, "tensor_model_parallel", False)`.
- **The cross-GPU drift check is a collective**, so every rank must run it on the same steps or the others wait until the process group times out. It runs where `(step + 1) % replica_every == 0`, from the step number, so the ranks must agree on step numbers: the same `start_step`, or the same `step=` passed to `session.step`.
- **The writer queue is bounded** at 8 blocks of `every` steps, as TensorBoard's `SummaryWriter` bounds its queue (`max_queue=10`). If the sink falls behind, for example on a stalled network filesystem, training waits for it with a one-time `RuntimeWarning` rather than holding unwritten steps in memory; no step is dropped.
- **Register your own forward hooks before `tensorquill.init`.** On PyTorch 2.10 with a Hugging Face Llama, a hook registered with `prepend=True` on a block TensorQuill also hooks crashes `torch.compile` while it builds guards (`IndexError: tuple index out of range`), with or without `skip_nnmodule_hook_guards`. The same hook registered normally, or prepended on a block TensorQuill does not hook, works. Not checked on PyTorch 2.14.
- **Hooks stay on the model after `close()`**, because removing a hook from a compiled module triggers a recompile.
- **Full-capture cost.** Each watched module adds a few small kernels per step. Granite's measured ~6% for activation capture was with every leaf module (171); blocks are the default now (`--layers all` in the benchmark for the old selection). Full capture with defaults measured +4.15% against capture off on the RTX PRO 6000 (Stage 3 above).

## Develop

Use PyTorch 2.14.1 in the local development environment. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'torch==2.14.1' -e './sdk[dev,rerun]'
```

That environment skips the Lightning, Hugging Face, Accelerate and streaming tests. Before committing a change to an adapter or to streaming, run the tests with every extra installed and the server built, as CI does: `pip install -e './sdk[dev,lightning,huggingface,rerun,flight]'` and `cargo build --manifest-path server/Cargo.toml`. A change once passed locally and failed there because only the project environment was used.

For CUDA dogfooding, use the training job's PyTorch build and CUDA environment.

```sh
cd sdk
../.venv/bin/python -m pytest
../.venv/bin/python examples/toy_moe.py --rerun
```

The ingest server needs a Rust toolchain. `test_flight.py` needs only `pyarrow` (`pip install -e './sdk[flight]'`); `test_flight_e2e.py` runs against the server binary and is skipped until it is built (or set `TENSORQUILL_SERVER_BIN`):

```sh
cargo test --manifest-path server/Cargo.toml
```

CI ([ci.yml](../.github/workflows/ci.yml)) runs the tests on every push and pull request, on Python 3.11 and 3.12 with PyTorch 2.6, 2.10 and 2.14 (CPU builds), and once more with the Lightning, Hugging Face and Rerun extras installed. It also runs `gpu_check.py --skip-overhead`, which on CPU uses the `aot_eager` backend. The real GPU gate is a separate job, started by hand from the Actions tab with `gpu` ticked; it needs a self-hosted runner with a CUDA build of PyTorch, and none is registered yet.

The `Trainer` callback has its own acceptance check, a dense model fine-tuned through `transformers.Trainer` with `torch_compile=True`, clipping and fp16, with the same seeded fault as `dense_outlier_check.py` (`--fault`, default `massive`). It exits non-zero unless every step, loss and clipping norm is recorded and agrees with what `Trainer` logs, the detector finds the outlier, and nothing recompiles:

```sh
pip install -e 'sdk[huggingface,flight]'
python examples/trainer_check.py --out runs/trainer-check
```

It passes with `--toy --backend aot_eager` on CPU. On a GPU (Colab, A100-SXM4-80GB, PyTorch 2.14.1+cu130, SmolLM2-360M, Inductor, fp16, clip 1; Nov 15, 2026) the first run trained 400 steps in 139 s and then crashed in the compile check, which counted Dynamo's backward-graph metrics as frames (fixed in `2ef2da4`). The rerun passed nine of ten checks: every step recorded and numbered as `Trainer` numbers them, loss and clipping norm on every step and equal to `Trainer`'s logged values, `grad_scale` on every step (4 steps skipped), 4 probe passes, 0 graph breaks, 1 compiled frame never recompiled. It **failed the detection check**: no `activation_spike` was found at step 250 in `model.layers.16`.

The cause is the seed, not a missed reading. The block's output had `abs_max` about 20,000 and `std` 31 to 37 at every step before the seed (the model's own massive activations, about 640 times the `std`), and scaling two channels of 960 by 100 moved the block's `std` by 1.07 to 1.14 times and its `abs_max` by 0.95 to 1.07 times, against a threshold of 5. The seed was real: the loss went from about 2.3 to 6.9 on step 250, and the `GradScaler` skipped steps 250 to 253 (scale 65,536 down to 8,192), so the gradient-norm detector first flagged step 254. The block summary (`std`, `abs_max`, `mean`, `zero_frac` of the whole output) cannot see growth confined to a few ordinary channels once the output holds large activations; it would see growth in the massive ones or a block-wide change. The toy Llama has 64 channels, so the same seed was a 13 to 42 times change there. Decided and changed the same day: the check now seeds the channels with the largest outputs by default (`--fault massive`), or the whole block (`--fault block`); `--fault channels` keeps the old seed, and a test builds this situation in miniature and runs the detector on it. The three faults pass on the CPU toy. Rerun on the A100 (same setup as above, commit `d987509`) with `--fault massive` (channels 87 and 398) and `--fault block`: **all ten checks passed in both**. The loss went from 2.32 to 8.46 with 0 steps skipped (`massive`), and to 13.27 with steps 250 to 252 skipped (`block`). The activation spike is at step 250 and the drift at step 254, each covering blocks 16 to 31, and the first gradient spike is at step 250 (`massive`) or 253 (`block`), first in `model.norm.weight`. On this run the spike listed modules by ratio, so it led with block 31, not the seeded 16; fixed the same day (it now leads with block 16 and names block 31 as the largest). The check still only requires the seeded block to be among the listed modules. `--fault channels` was not rerun on a GPU.

On a CUDA machine, run the GPU check. It fails on any graph break, recompile or CUDA-graph skip, and then reports the time per step with and without capture:

```sh
python examples/gpu_check.py --dim 1024 --blocks 8 --batch 4096
```

Set the sizes close to the real job, because a small model makes the fixed per-step cost look large.
