# TensorQuill capture SDK

The capture SDK, detectors and evidence exporter of the [v3 plan](../TensorQuill%20v3%20—%20Product%20and%20Build%20Plan.md). It records what happens inside a PyTorch model during training, cheaply enough to leave on, and in a way that survives `torch.compile`.

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
| Probe pass (`probe=`): activation histograms per block and attention logit, entropy and maps per layer, on a fixed batch every `probe_every` steps | Built, tested on CPU, under `torch.compile` with no graph breaks or recompiles, and on a small randomly initialized Hugging Face Granite MoE. Run on an A100 and an RTX PRO 6000 with SmolLM2-360M (see "Dense model with a seeded activation outlier"); not run on Granite on a GPU |
| Logged scalars (`session.log`) attached to incidents and plotted in Rerun | Built, tested on CPU and Apple MPS. On an A100 the `Trainer` callback's logged loss equalled `Trainer`'s own (`trainer_check.py`) |
| Opt-in streaming (`endpoint=` or `TENSORQUILL_ENDPOINT`, `tensorquill/flight.py`): a daemon thread tails the rank's `.jsonl` file from `<run_id>.rank<rank>.offset` and sends it over Arrow Flight `DoPut` | Built, tested on CPU against a stand-in server and the real `tensorquill-server` on loopback: torn lines, offset resume, duplicate resend, a refused batch, a server killed for 200 steps, and `grpcs://` with `--tls-cert`/`--tls-key` and `TENSORQUILL_TLS_CA`. Not run to a remote host or from a GPU job |
| Ingest server (`../server`, binary `tensorquill-server`): `DoPut`, `(run_id, rank, attempt, step)` deduplication, 128 MiB buffer quota, Zstd Parquet parts through `object_store`, Bearer token, TLS | Built; `cargo test` passes on a local-disk store and TLS passes end-to-end on loopback. S3 ran against an emulator (moto) through the real binary; GCS has never run |
| Detectors on the server's Parquet: `load_records` and `python -m tensorquill` read a run's parts directory, one rank's, or one part, and give the same evidence as the source `.jsonl`; `--watch` re-reads a growing file or directory every 2 seconds (`--interval`) and replaces the evidence by rename; `--rank` (`rank=` in `build_evidence` and `load_records`) picks a rank and reads only that rank's folder; `build_evidence` on a directory with several ranks and no `rank=` raises | Built, tested on CPU, also through the real server binary |
| Cross-GPU parameter fingerprints and replica spread (`parameter_fingerprints`, `replica_spread`) | Built, unit-tested on drift and sign-flip cases |
| Detectors and Rerun exporter (`build_evidence`, `export_rerun`, `python -m tensorquill`) | Built, tested on MoE collapse (top-k aware), starved experts, gradient spikes, weights not updating, repeated steps, NaN/Inf, and replica drift |
| Run ID from `run_id=`, `MLFLOW_RUN_ID`, the active MLflow run or the active W&B run; otherwise torchrun's or SLURM's job ID, or a generated `run-<UTC time>-<6 hex>` broadcast from rank 0, so separate untracked runs never append to the same file | Built, tested with stand-in tracker modules and two Gloo processes |
| The Rust detectors on a local file (`tensorquill-server detect`, `--detectors rust`, `build_evidence(detectors="rust")`): the incidents of the 25 per-rank kinds, the five threshold options, and the per-step MoE samples (`detect --samples`) | Built Nov 16, 2026, tested on macOS against the Python detectors on 64 record sets and against the golden files. Not the default, and no Python detector is deleted. Not run on Linux |
| What was checked (`kind_checks` in `evidence.json`, in the server's `incidents.json` and in `detect`'s output): `checked`, `not_configured` or `unavailable` for each of the 32 kinds, the count in the summary, and `degraded` only for `unavailable` | Built Nov 16, 2026, tested on macOS in Python and in Rust, the two held equal on 71 record sets. Also in the watcher's evidence from a server's report. Not in the all-ranks view. Not run on a real job |
| Compact evidence (`--compact`) and the trigger (`--watch --on-incident COMMAND`, `--webhook URL`): once for each new incident, invariants by default | Built Nov 16, 2026, tested on CPU (macOS), the store paths through the real server binary. No real agent or outside webhook has been driven by it |
| Read-only MCP server (`python -m tensorquill.mcp_server PATH`, four tools) and the explain-only [agent instructions](AGENT_INSTRUCTIONS.md) | Built Nov 16, 2026, tested with a scripted client on the seven Gate 1 fixtures. No live agent has used it |
| The Rust detectors from Python (`--detectors rust`, through `tensorquill-server detect`): one read of a file or store, `--watch` on a growing file or a store, and `--rank all`; `detect --run` for every rank of a run, with `duplicate_batches` across ranks; a reader that takes the thresholds a server applied; and the all-ranks view as one call, `evidence.all_ranks_evidence` | Built Nov 16, 2026, tested on CPU (macOS) with the debug binary: the same evidence as the Python detectors on 57 record sets and the seven Gate 1 fixtures, read once and written a stretch at a time. The Python detectors are still the default and none is deleted |
| No graph breaks and no recompiles under `torch.compile` | Tested with `eager`, `aot_eager`, and AOTAutograd static-input checks on CPU; on CUDA with Inductor and CUDA graphs on a toy MoE, and with Inductor and CUDA graphs off on Granite |

## What is not built or not verified

- **The agent loop has not met an agent.** The MCP server has been driven by a scripted client only, and the trigger by a local script and a local HTTP server. No agent session has been started by the trigger or has used the server.
- **Detectors on the server's Parquet, limits.** The reader takes a local path, so parts in S3 or GCS cannot be read yet. A directory holding several ranks is read one rank at a time, and only that rank's `rank=` folder is opened; `--rank all` reads every rank's report and no part (see "All ranks at once"). On a `.jsonl` file, or on a store whose server did not report (`--no-detect`, or `--recompute` here), `--watch` reruns the detectors over the whole run on every update, so an update takes time in proportion to the run's length: 0.4 s at 500 steps and 8.1 s at 10,000 on a Granite-shaped 64-rank store (see [Detector benchmark on 64 ranks](#detector-benchmark-on-64-ranks)). On a store the server is judging it reads the server's report and no part (see [Watching a run the server is judging](#watching-a-run-the-server-is-judging)).
- **Streaming, not verified:** GCS stores (S3 ran against an emulator only, see "Storing to S3 or GCS"), more than two ranks from real training processes at once (two pass on CPU, `tests/test_two_ranks_e2e.py`), a remote server and a GPU job. A token from the Python sender, TLS on loopback (`--tls-cert`/`--tls-key` with `TENSORQUILL_TLS_CA`), a non-zero rank, and forked `DataLoader` workers while the sender runs each pass an end-to-end test on macOS and on Linux in CI (as of `f30e639`). In a forked child the sender does nothing: the fork copies the object but not its thread, so `close()` there returns at once and the parent keeps streaming. No deadlock was seen in either fork test, but gRPC's behaviour across `fork` differs by platform, so a Linux machine with a CUDA job is still worth checking. The sender's memory bound during an outage is measured with `tracemalloc`, which does not see gRPC's own allocations.
- **What was checked, limits.** A kind counts as checked from the first record that carries its input, before its rule has the history to fire. The all-ranks view does not carry `kind_checks` yet (see "What was checked").
- **One Rust copy of the detectors is not finished.** The Rust path gives the incidents and the MoE samples, but the default is still the Python detectors and none is deleted. `--watch` on a local file and `--rank all` do not use the Rust path. The router names, sizes and experts per token in the evidence are still read from the records in Python.
- **Replay** of a flagged batch (Tier 3). The probe pass (Tier 2) has run on a GPU with SmolLM2-360M, not yet with Granite.
- **The Rust detectors are not the default, and the Python ones are all still here.** `--detectors rust` covers one read, `--watch` and `--rank all`, and `duplicate_batches` across ranks has a Rust path without a server (`detect --run`). The per-step MoE `samples` come from Rust on that path too. What is left before the Python copies can go is the switch of the default, and then deleting them. Not verified: Linux, any run longer than 10,000 steps, `detect --run` on more than four ranks or on a real multi-rank capture, and the release binary in the tests (they use the debug one; the release one was only timed). See "The Rust detectors on a local file".
- **Multi-rank reporting.** Every rank writes its own file. The watcher has an all-ranks view of the server's reports (`--rank all`); what stays per rank is the Python detectors, the per-step samples and `--rerun`.
- **Detector thresholds are untuned.** The gradient-spike factor (5× the median of the last 50 recorded steps, and for one parameter 3 steps in a row at 5× its own), the top-k diversity drop (0.6 of its recent median), the starved-expert ratio and the stalled-weight check have been checked on synthetic cases and, for the MoE rules, on one Granite capture. Weight norms, learning rates and the spike and stalled-weight detectors have run on the Granite captures from the A100 and RTX PRO 6000 runs below, but not against a planted failure of their own kind on a GPU; the steps they flagged there are unconfirmed.
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
- **Weights not updating.** Every 50 steps by default, the capture computes every parameter's norm in float64 and compares it with the previous check; `weights_unchanged` lists the parameters whose norm is exactly the same. A parameter with a finite non-zero gradient whose norm is unchanged on 3 checks in a row (150 steps by default) is reported as a warning (equal norms do not prove the weights are equal), for example because it was left out of the optimizer. Since Nov 15, 2026 each weight check also records `optimizer_ownership`, and the incident reports whether an optimizer that stepped holds the stalled parameters: `known`, or `unknown` with a reason when ownership cannot be established (`auto_step=False`, DeepSpeed ZeRO, Megatron, FSDP1). Since Nov 16, 2026 every parameter is judged on its own history: the incident is still dated at the first step anything stalled, and its `onsets` list each later step at which more parameters stalled, with their names (those that had changed at an earlier check first), how many of them had (`trained_before`) and the optimizer evidence for them, so a freeze that begins after the bf16 norm weights have stalled is named at its own step; `stalled_total` counts every stalled parameter and `weight_dtypes` (recorded at each weight check) says how many trainable parameters have each dtype. Steps with a non-finite gradient norm are skipped, since mixed-precision loss scaling skips those on purpose. The first version compared float32 norms, and on the A100 run below it reported 147 of 218 parameters from step 1: a bf16 weight under a 1e-5 AdamW step changes in only a few elements per step, which a float32 norm often misses (`model.layers.7.self_attn.k_proj.weight` kept the same float32 norm for 128 steps while changing). Float64 norms resolve those changes. bf16 RMSNorm weights near 1.0 are still reported, correctly: their step size in bf16 is about 0.008, so a 1e-5 update rounds away, and in a CPU simulation such a weight was exactly unchanged on 273 of 650 steps. Each check reads every weight once, with no copy, accumulated in float64; on Apple MPS, which has no float64, the check is off. Captures made before `weights_unchanged` existed are not checked. The check runs every 50 steps rather than every step because the read alone costs about 1.5 ms for Granite's 2.6 GB of bf16 weights on the RTX PRO 6000, about 2% of a step, whatever arithmetic is used; with float64 on that GPU it measured about 3.7 ms. wandb.watch (every 1,000 batches by default) and TrainCheck (`sampling_interval`) sample their weight checks for the same reason. At every 50 steps the average cost is under 0.1 ms per step, not yet measured on a GPU.
- **Gradient spikes.** Two rules, and the incident is the first either establishes. The total: a step whose total gradient norm is at least 5× the median of the previous 50 recorded steps; the report ranks parameters by their own spike and names the one that started rising first. One parameter (`"trigger": "parameter"`, since Nov 16, 2026): a parameter whose own norm stays at least 5× its own median for 3 recorded steps in a row, whatever the total does, dated at the first of those steps and known two recorded steps later. A small parameter cannot move the total (a `k_proj` holding 0.2% of it, times 100, moves it 1.02 times). Three in a row comes from the real captures: in six recorded Granite captures (650 steps, 218 parameters; the rule was chosen on two and checked on the other four) a single step at 5× happened 277 to 473 times and two in a row 2 to 15 times, never three, nor on the steps before the seed in three SmolLM2-360M captures, while each seeded fault held it from its seeded step for as long as the capture ran. Norms that are part of such an excursion stay out of the parameter's median. On the SmolLM2 block-8 capture, where the total reached 3.9× and nothing was reported before, it reports step 250 with `model.norm.weight` (62×) and `model.layers.8.input_layernorm.weight` (9×), the seeded block. A single-step jump in one small parameter is still not reported: it cannot be told from the healthy ones.
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

- **Measured cost on Granite, RTX PRO 6000:** router-only +0.02% to +0.78% across three runs; full capture with block defaults +4.15%, of which about 1.5 ms is gradient norms on every step and 0.85 ms block activation statistics. On the A100, router-only was +1.06% to +1.33% at one commit and +0.41% to +0.95% in a later profiling run (see "Where the A100 router-only overhead goes"). The gate protocol (`moe_dogfood.py`: three paired rounds of 200 steps after 50 warmup steps; commit `2b56eac`, Nov 15, 2026) read router-only +0.70% (64.97 to 65.42 ms per step; rounds +0.76%, +0.67% and +0.56%), full capture +4.11%, and gradient norms +3.42% (+2.25 ms) over full capture without them; the layer-12 collapse was found at step 450 with diversity 0.25 as before.
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

## Seeded gradient spike and stalled weights

**Seeded gradient spike and stalled weights.** `python examples/seeded_failures.py --case both --out runs/seeded` trains SmolLM2-360M on WikiText-2 with full capture and plants one failure in one parameter, the middle block's `self_attn.k_proj.weight`: its gradient multiplied by a factor from step 250 (a tensor hook, so no other gradient changes), or its learning rate set to 0 from step 100 (its own param group; the gradient stays finite and the optimizer still holds it). It exits 1 unless a `grad_spike` incident starts within 50 steps of the seed and lists the parameter, and a `stalled_weights` incident lists the parameter with an onset within 50 steps of the first step three weight checks could show the freeze (step 249). `--toy` runs a tiny random Llama on CPU (about 20 s) and is in CI.

It tests both gradient rules. One parameter's gradient ×100 moves the total only if it holds about 5% of the total norm (a `k_proj` on the toy holds 0.2%), so without `--factor` the script picks the factor that takes the total to twice the threshold, which tests the total rule; `--factor 100` leaves the total where it was and tests the one-parameter rule. With bfloat16 weights the RMSNorm weights stall from step 199, before the seeded freeze can show; the incident begins there and names the seeded parameter in its `onsets`, at step 249, as one that had changed before.

In Colab, after the usual clone and install, from `sdk/`:

```sh
# the gate command
python examples/seeded_failures.py --case both --out runs/seeded
# x100 on one k_proj: the total does not move, the parameter's own history shows it
python examples/seeded_failures.py --case grad_spike --factor 100 --out runs/seeded-x100
# fp32 weights, without the bf16 background
python examples/seeded_failures.py --case stalled_weights --dtype float32 --out runs/seeded-fp32
```

**Status.** Built Nov 15, 2026; the two detector findings it recorded were closed on Nov 16, 2026 by judging each parameter on its own history. On the CPU toy all of these pass: `--case both`, `--case grad_spike --factor 100` (reported at step 250, known at step 252) and `--case stalled_weights --dtype bfloat16` (the seeded parameter named at step 249 beside the norm weights from step 199); the first two are in CI. None has run on a GPU: the script's CUDA path, SmolLM2 and fused AdamW with a zero-lr group are untried, so expect to read the first run's output closely.

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

**RTX PRO 6000 result, block 8** (Colab, PyTorch `2.14.1+cu130`, commit `2b56eac`, `--layer 8`, the case Gate 2 names; the same channels 87 and 398): **`PASS`**. The spike is first in `model.layers.8` (`abs_max` 96× its recent median) with 23 more modules, the largest in `model.layers.31` (708×). The fault reached 24 blocks, more than the 20 an incident lists, and block 8 still leads with block 31 second; a list kept by ratio would have dropped block 8. Drift from step 254 (`std` 47×). Full capture added +1.18 ms (+3.72%, 31.79 to 32.97 ms per step); a probe step cost +69.8 ms over a flush step, 0.070 ms per step (0.22%) at the default 1,000 and 0.70 ms (2.19%) at every 100. The probe's largest value in block 8 went from 20,864 to 2,088,960 (100.1×), and the attention of block 9 went from entropy share 0.23 to 1.00 (largest logit 15.51 to 6.53). The same 28 of 290 norm weights were reported as not updating, as on the A100.


**Pending.** The two-process CPU/Gloo check passed. The Colab runtime exposed one NVIDIA RTX PRO 6000, so the CUDA/NCCL check was skipped.

To resume, use a single machine with two visible NVIDIA GPUs and a CUDA-enabled PyTorch build. Run from the repository root. The check generates its own data and needs no Colab artifacts.

```sh
set -e
cd sdk
python -c 'import torch; assert torch.cuda.device_count() >= 2, "Two visible CUDA GPUs are required"'
python -m pip install -e .
python -m torch.distributed.run --standalone --nproc-per-node=2 examples/replica_check.py
```

Add `--out runs/replica` to keep each rank's records as `runs/replica/replica-check.rank<N>.jsonl`; without it they stay in memory. Mark this complete when both NCCL ranks report `PASS`. Each rank checks healthy replicas at steps 0 and 2, injected drift at step 1, and a sign flip at step 3.

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

**Where a silent rank is stuck.** When a rank's work counter (steps plus `session.tick()`) has not moved for 60 seconds (`init(stack_seconds=...)`, `None` turns it off), each heartbeat also carries where the process's Python threads are: file, line and function names, the main thread first (up to 40 frames, 4 for each of up to 7 other threads, so under 10 KB a rank at the most), with TensorQuill's own threads left out and frames of TensorQuill's own code marked. No source text is read and no local values are taken. The server keeps the newest one in the rank's entry in `watchdog.json` as `stack` (dropped if the heartbeat is larger than 64 KB), also after the heartbeats stop and over a server restart, and clears it when a heartbeat arrives without one. `job_stalled` and `rank_silent` then list `stuck_at`: for each stopped rank the innermost frame, the nearest frame that is not a library's (`caller`), how long the main thread's stack had not changed (`unchanged_seconds`), and `in_tensorquill` when the call was reached through TensorQuill's own code, such as the replica check's `all_gather`. The server's own `incidents` in `watchdog.json` carry the same `stuck_at`; the whole stacks stay in the status's `ranks` and are not copied into incidents or `evidence.json`. The assessment stays `suspected_hang`: a stack says where Python was, not why. Limits: Python frames only, so a hang inside NCCL shows as the Python line that called the collective (PyTorch releases the interpreter lock around its collectives, so the heartbeat thread keeps running); a main thread hung while holding the interpreter lock stops the heartbeat thread, so the rank is `unresponsive` with no stack; verified with a process blocked in a Python wait, not yet with a real stuck collective on GPUs.

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
- **Columns.** Numbers and name-to-number maps are typed columns; `layers`, `moe_stats` and `probe` are JSON strings (`layers_json`, `moe_stats_json`, `probe_json`; `layers_json` keeps the model's key order, which the activation detectors use to say which block showed a change first, while the other two are written with sorted keys; Parquet parts sent by a version before Nov 15, 2026 have `layers` sorted by name, so "first" there is first by name); anything else travels in `extra_json`, so a row converts back to exactly the record in the `.jsonl` file. The choices behind this are in the plan's Decisions section.

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
| `runs/<run_id>/rank=<rank>/incidents.json` | `moe_imbalance`, `starved_experts`, `grad_spike`, `stalled_weights`, `repeated_steps`, `loss_scale_trouble`, `resume_lr_change`, `resume_data_mismatch`, `resume_incomplete`, `data_stalled`, `epoch_order_repeated`, `activation_spike`, `activation_drift`, `dead_activations`, `attention_collapse`, `attention_uniform`, `attention_logit_growth`, `non_finite`, `memory_growth`, `memory_pressure`, `allocator_retries`, `hardware_errors`, `gpu_throttled`, `input_starvation`, `replica_drift` (Nov 16, 2026: the data, checkpoint, attention and system kinds added) |
| `incidents` in `runs/<run_id>/watchdog.json` | `job_stalled`, `rank_silent`, `duplicate_batches` (cross-rank), plus the `run_state` |
| Python only | `collective_hang`, `collective_desync`, `traincheck_violation`, `memory_imbalance` (cross-rank, in `--rank all`) |

The first three Python-only kinds stay in Python because their sources never reach the server: PyTorch's Flight Recorder dumps are pickles on the training nodes, read by PyTorch's own analyzer, and TrainCheck writes its results where it ran. `memory_imbalance` compares ranks, and the server's detectors see one rank at a time; it reads the `memory` summary the server writes into each rank's `incidents.json`. The timings below are from Nov 15, 2026 with twelve kinds; they were repeated with all 25 per-rank kinds on Nov 16 (see [Re-measured with 25 kinds](#re-measured-with-25-kinds)).

- **What it writes.** After each part is stored and acknowledged, the server reads that part's rows and rewrites the rank's `incidents.json`: the incidents in the shape and order `build_evidence` gives them, `last_step` (detection has run through here), the kinds `checked`, `not_checked` (in the shape of the same list in `evidence.json`: the two sources the server does not have, and each resume whose learning rate could not be compared), `kind_checks` (one of three values for each of those kinds, see [What was checked](#what-was-checked)), the thresholds applied, `moe_flagged` (how many router samples had an overloaded or dead expert and the last step one did, which says whether an imbalance is still going on), and `rebuilds`. It holds no per-step samples: those stay in the Parquet parts, so the file stays a few KB however long the run is. `--no-detect` turns it off. The watchdog status carries its verdict as an `incidents` list as well, rewritten whenever the status changes, so a stall shows there although no part arrives. `python -m tensorquill --watch` reads both (see [Watching a run the server is judging](#watching-a-run-the-server-is-judging)); without `--watch` the command works the incidents out itself from the parts.
- **Only stored rows are judged**, so the file can always be rebuilt from the store. After a server restart, on every new attempt, and whenever an older attempt's rows arrive late, the rank's state is rebuilt from all of its stored parts with the same attempt rules as `load_records`. That replay also reads the rows a later attempt rolled back: `repeated_steps` and `resume_lr_change` compare a new attempt with what it replaced. The step numbers and learning rates it needs for that are held only while the replay runs. A detector error or panic is logged and never fails a flush.
- **How late it is.** Detection follows storage, and a part is written every `--flush-steps` steps (500 by default) or `--flush-seconds` (10), so by default an incident shows about 10 seconds after its step reached the server, at the latest.
- **Same answers as Python.** [`tests/test_server_detectors.py`](tests/test_server_detectors.py) (16 tests) streams records through the real binary and compares `incidents.json` with `build_evidence` on the same records: after 16 stretches of a 300-step run in which ten kinds each happen once; after 12 stretches of a 400-step run in which one small parameter's gradient spikes without moving the total and five parameters stall at three different steps, and across a restart inside that excursion; across a server restart in the middle of a 140-step collapse; after each of 12 stretches of a run with nine attempts (resumes at a step the run had, past the stored steps, with no learning rate, and two restarts of the numbering); on seven short runs for rules the long one does not reach; and on the seven Gate 1 fixtures. `test_flight_e2e.py` and `test_two_ranks_e2e.py` compare the status's `incidents` with `detect_watchdog_incidents` on a stalled job, a killed one, a lost rank and a rank that never connected. Twenty-two deliberate changes to the Rust rules each fail these tests, and so do nineteen more to the per-parameter rules added Nov 16, 2026 (one first survived, excursion norms entering the median, and the fixture was changed until it failed). Steps, layers, experts and names match exactly and numbers to 1e-12.
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

### Storing to S3 or GCS

```sh
tensorquill-server --listen 0.0.0.0:50051 --store s3://my-bucket/tensorquill
tensorquill-server --listen 0.0.0.0:50051 --store gs://my-bucket/tensorquill
```

Credentials come from the usual environment: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` and `AWS_REGION`, or the instance or pod role (EC2, ECS, EKS web identity); for GCS `GOOGLE_APPLICATION_CREDENTIALS` (a service-account file) or the instance's own account. Shared credential files and `AWS_PROFILE` are not read. For an S3-compatible service (MinIO, an emulator) set `AWS_ENDPOINT_URL`, plus `AWS_ALLOW_HTTP=true` if it is `http://`. The server needs `GetObject`, `PutObject` and `ListBucket` on the prefix and nothing else; it never deletes. Credentials are never printed. S3 has run against the `moto` emulator through the real binary; **GCS has never run** (the code compiles and its builder is unit-tested), and nothing has run against real AWS, TLS to S3 or real latency. `python -m tensorquill --watch` and `load_records` still read a local directory only.

The server reads the store when it starts (it lists `runs/` and reads each unfinished run's `watchdog.json`), so a wrong bucket or bad credentials stop it with the store's message instead of failing at the first batch. A failing store is retried for `--store-retry-seconds` (30); then the sender is told the store is unavailable and keeps its rows. A restarted server takes up the runs it was watching: a silent rank is still silent, a stalled run is still stalled, and the time the server was down is not counted against a rank. A run whose ranks have all been disconnected or finished for `--abandon-hours` (72) is given up. `cargo build --release --no-default-features` builds a local-disk-only binary (3.9 MB smaller).

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

### Re-measured with 25 kinds

Nov 16, 2026, same machine and examples, at load 6 to 22 from other work. The server from before that day's kinds ran beside the new one. Two stores were used: the benchmark store as it is, and a copy whose new columns are filled at realistic sizes (a data record every step, a checkpoint save and a 1.3 MB probe every 1,000 steps, a system reading every flush, data wait every step).

| Per flush of 64 ranks | Store as before | New columns filled |
| --- | ---: | ---: |
| Detectors, one thread, median (worst) | 61 to 63 ms (73), the same as the twelve-kind server | 66 ms; 146 to 150 ms (166) on the 1-in-20 flush that carries a probe |
| Live, 65 ms per step: sent to acknowledged, median / p99 / worst | 46 to 57 / 109 to 158 / 133 to 202 ms (3 runs) | 61 to 64 / 165 to 184 / 228 to 243 ms (3 runs); a fourth run: 72 / 248 / 620 ms |
| Server CPU and peak memory | 11% of a core, 364 to 379 MB | 12 to 13%, 652 to 704 MB |
| Restart with 10,000 steps stored: next part waits | 6.6 to 7.9 s | 7.1 to 7.9 s |

- **Where the time goes.** The thirteen new kinds cost 3 to 5 ms on an ordinary flush. A probe flush costs about 86 ms more, and about 95% of that is serde_json reading past the probe's 64×64 attention maps. The maps are 98% of the probe's bytes, and no rule reads them.
- **Storing a probe flush.** Writing its 64 parts takes 350 ms against 155 ms for an ordinary flush. The server already paid this before Nov 16; no earlier benchmark had a probe in it.
- **The budget.**
  - As detector time it holds: 66 ms on an ordinary flush, 150 ms on a probe flush.
  - As acknowledgement time under 64 live streams it no longer holds reliably once probes are on. One filled run of four reached 620 ms; that flush is unattributed, and a probe flush under load is likeliest.
  - The fix is to move the maps and histograms out of `probe_json` into a column the detectors do not parse, or out of the step record. That is a schema change and is not made yet.
- **Not measured:** the 10 ms pace, more than 64 ranks, a real Granite probe, a GPU, S3 or GCS.

### Watching a run the server is judging

`python -m tensorquill <store>/runs/<run_id> --watch` takes its incidents from the server while it can: the rank's `incidents.json` for the 25 kinds read from step records, and the `incidents` list in `watchdog.json` for `job_stalled` and `rank_silent`. It reads no Parquet part then: a look lists the rank's part names and reads two small files, 2 ms with 216 parts (10,800 steps), where running the detectors here took 12 s. `--flight-recorder` and `--traincheck` are still read here and added, in the order `build_evidence` gives.

An empty `incidents` list means "no incidents" only in a report that covers what is stored. The report carries no clock, so the watcher compares it with the store: `read_server_report` takes the report's `attempt` and `last_step` and the newest stored part's, which are in that part's path (`attempt=<a>/part-<first>-<last>.parquet`), and calls the report:

| State | Meaning | What the watcher does |
| --- | --- | --- |
| `current` | The report ends exactly at the newest stored step of the newest attempt, covers all 25 kinds, and no threshold given here differs from the one the server applied (see "Detector thresholds") | Uses it. "No incidents detected" is followed by "current through step N of attempt A (M steps judged)" |
| `behind` | The report stops before the newest part. Normal for a moment after every part (0.1 ms at the median, 77 ms at worst, measured), and for 6 to 7 s while a restarted server rebuilds 64 ranks of 10,000 steps | Waits, up to 30 seconds (`--report-grace`) since the report was last current; then the report is `stale` |
| `stale` | Still behind after that wait: the server was restarted with `--no-detect`, its detectors failed for this rank, or it died between storing a part and reporting on it | Reads the parts and runs the detectors here, and says the report stops at step X while the store holds step Y |
| `missing` | Parts, and no report: `--no-detect`, or a server from before its detectors | The same, at once if no report was ever seen |
| `unusable` | A report for another run or rank, from a server with fewer kinds, ahead of the store, or beside parts whose names carry no steps | The same, at once |
| `other_thresholds` | A threshold given here (`--overload-threshold`, `--drift-tolerance`, `--starve-window`, `--starve-ratio`, `--top-k`) differs from what the server applied (its own are set with `tensorquill-server --thresholds`). One not given here is taken as the server applied it | The same, at once, with the value given here and the server's values for the rest |
| `unreadable` | The file could not be read or parsed on this look | Waits like `behind` |

Each evidence file says which happened under `detection` (`source` is `server` or `python`, with `state`, `reason`, and the steps and attempts compared), and the summary's last line says it in words. Evidence built from a report has no `samples`, `weight_trends`, `layer_count` or `router_names`: per-step data stays in the parts. `--recompute` ignores the report and gives them, at the cost above.

After a restart, specifically:

- **A clean stop or a kill, then a new server.** Every part the old server acknowledged was judged before it went on, so the report it left is still `current` for what is stored, and stays so until a rank sends again. The new server then rebuilds that rank from the store and the report moves on (`rebuilds` starts again at 1).
- **A kill between storing a part and reporting on it** leaves a report that is `behind`. A watcher that starts then works the incidents out itself; one that was already running waits its 30 seconds first. The next part the new server stores brings the report up to date.
- **The status file.** A killed server leaves its last `watchdog.json`, which may say `active`. A running server rewrites an unfinished run's status at least every `refresh_seconds` (30 by default) whether or not anything changed, so a stall that lasts an hour still has a fresh status. When the status claims a running server and is older than twice that, the summary says the status was last written N seconds ago and that nothing is watching the run for stalls; the same holds after a restart until a rank of the run reconnects, because the new server does not know the run yet. After a clean stop the status says `server_stopped`, and the summary says so. Neither is printed as a bare "No incidents detected."

Limits: the watcher judges freshness by the newest attempt only, so a part of an older attempt that arrived late and was not judged goes unnoticed; a NaN or Inf logged value prints as "not finite" (it is `null` in the server's file); `job_stalled` and `rank_silent` carry no logged scalars when they come from the status; and one rank is watched per command unless `--rank all` is used. [`tests/test_server_reports.py`](tests/test_server_reports.py) (12 tests) produces each state with the real server. Twenty-two deliberate changes to the reader's rules each fail it.

**All ranks at once.** `python -m tensorquill <store>/runs/<run_id--rank all --watch` (and `watch(run_folder, rank="all")`) keeps one evidence file for a run with several `rank=N` folders, `<run_id>.all-ranks.evidence.json`, in a shape of its own (`"view": "all_ranks"`, `"view_version": 1`; no `schema_version`, `incidents` or `samples`, so the per-rank shape is never mistaken for it). A look reads each rank's `incidents.json` and newest part name, and the run's `watchdog.json`, and nothing else: 34 ms for 64 ranks of 10,000 steps, 102 ms at 50,000, on an 8-core laptop. A rank counts for or against incidents only while its report is `current`; `behind`, `stale`, `missing`, `unreadable`, `unusable`, `other_thresholds` and `nothing_stored` are *unknown*, with the reason, and never in the list of ranks with no incidents. Each rank has its own wait (`--report-grace`). Incidents with the same kind, step and named layers, experts, parameters or modules are one group naming the ranks ("0-63"); those that differ are listed with their ranks, and an incident on one rank is named with that rank alone. A group carries one incident (the first rank's) and says whether the others' numbers differ. The text summary gives what is common to all judged ranks, what only some show, the ranks with none, and the unknown ranks; stalls and silent ranks (`job_stalled`, `rank_silent`) are in `run_incidents` once for the run. Ranks are the `rank=N` folders, the status's ranks and `world_size`, so a rank that never stored anything is `nothing_stored`, and after a restart a rank that has not reconnected is still judged from the store. The file is replaced by rename only when a rank's report, a rank's state, the set of ranks or the status changed. No part is read and no detector runs here, so `--recompute` and `--rerun` are refused with `--rank all` (use `--rank N`), a threshold option given with it must be the one each rank's server applied (else that rank is `other_thresholds`), and `--detectors rust` changes nothing; without `--watch` it makes one look. The same look is one call in Python, `tensorquill.evidence.all_ranks_evidence(run_folder)`, which returns the evidence and writes nothing. Limits: a rank is called stale only after `--report-grace` and a blip inside it is not written; `stored_last_step` of an unknown rank is as of the last write; a rank is judged against its own store, so ranks at very different steps are all `current` (the spread is in `last_step` and in the status's `stream` entries and `rank_silent`); numbers differing between ranks are in each rank's own `incidents.json`, not copied.

### The Rust detectors on a local file

`tensorquill-server detect` runs the server's detectors over one rank's rows without a server or a store, and prints the report the server would write as that rank's `incidents.json`. Python calls it, so the 25 per-rank kinds can come from the one Rust implementation on a local `.jsonl` file too:

```sh
cargo build --release --manifest-path server/Cargo.toml
export TENSORQUILL_SERVER_BIN=$PWD/server/target/release/tensorquill-server
python -m tensorquill tensorquill-out/RUN.rank0.jsonl --detectors rust
```

In Python it is `tensorquill.build_evidence(records, detectors="rust")`, and `tensorquill.server_detect.server_report(records)` gives the report itself.

- **What it needs:** the binary, on `PATH` or named by `TENSORQUILL_SERVER_BIN`, and `pyarrow` (the `flight` extra). The records travel as an Arrow stream in the schema the sender uses.
- **What comes from Rust:** the incidents of the 25 per-rank kinds, with their scalars, the `not_checked` entries, what was checked for those 25 kinds (`kind_checks`), and the per-step MoE `samples`. The evidence then carries `"detectors": "rust"`.
- **What is still made in Python:** the router names and sizes, the weight trends, the run state, and the kinds read from other files (the server's status, TrainCheck, the Flight Recorder). No MoE rule runs in Python on this path.
- **The samples** are an option of the subcommand, `tensorquill-server detect --samples` (`server_report(records, samples=True)`): one more key in the report, `samples`, with one entry for every record and router as `build_evidence` lists them. Without the option the report is unchanged, and the server's own `incidents.json` never has them.
  - *Why not in `incidents.json`.* On 10,000 steps of 24 routers of 32 experts the report is 2.1 KB without samples and 96.9 MB with them, and the server would hold every sample in memory for each rank. The Parquet parts already hold the counts.
  - *Time, release binary, the same run, on a laptop under load:* 131 ms without samples and 324 ms with them. Through Python, with the records turned into an Arrow stream (1.8 s) and the output parsed, `server_report` takes 2.2 s and 3.5 s. `build_evidence` takes 5.6 s with the Python detectors and 3.5 s with `detectors="rust"`.
  - *Equal to Python's with no tolerance:* every field of every sample is the same value of the same type, a float the same double.
- **Thresholds:** the five options of `build_evidence` (`overload_threshold`, `drift_tolerance`, `starve_window`, `starve_ratio`, `top_k`) go to the Rust rules, and each has a flag on the command line (`--overload-threshold`, `--drift-tolerance`, `--starve-window`, `--starve-ratio`, `--top-k`). See "Detector thresholds" below.
- **Limits:**
  - The default is still `detectors="python"`.
  - A binary built from other thresholds than the SDK's, or one that lacks any of the 25 kinds, is refused.
- **Checked** in `tests/test_server_detect.py`: both paths give the same evidence on 57 record sets and on the seven Gate 1 fixtures. The record sets plant every one of the 25 kinds and every kind of `not_checked` entry, and include resumes and restarts written a stretch at a time and the edge cases of the server parity tests. Run by hand on eight recorded GPU captures (the seven of Nov 15, 2026 and an older Granite one; five hold 15,600 real MoE samples each), both paths give the same incidents, `not_checked`, `kind_checks` and run state, and the same samples with no tolerance.
  - The samples are compared field by field on those sets, at each of the 18 threshold option sets and with a `top_k` for one router. A difference is reported by step, router and field. The golden files (`tests/golden/`) hold the Rust samples to 1e-12, because a file may have been written on another platform.
- **Two differences, both pinned by a test:**
  - A step written twice within one attempt is reported by both, but Rust attaches the scalars logged at its second writing and Python those of its first. The server's own store never holds such rows.
  - Attempts out of order in the input are read oldest first, as a store holds them; Python reads a file in the order written.
- **Two more, in hand-made records only** (the SDK writes neither):
  - Python applies the last `moe_top_k` a run recorded to every step; Rust judges each step with the value known when it is read. This already held for the incidents and now holds for the samples' `overload_rule`.
  - A recorded `moe_stats` diversity that is not a finite number is used as it is by Python and replaced by the one worked out from the counts in Rust.

**Watching through Rust.** `python -m tensorquill run.jsonl --watch --detectors rust` (and `watch(path, detectors="rust")`) judges through `tensorquill-server detect` whenever the detectors run here: on every rebuild of a growing `.jsonl` file, and for a server's store whose report cannot be used (`--recompute`, or a report that is missing, stale or unusable).

- **The same evidence at every look.** Each of the 57 record sets and the seven Gate 1 fixtures is written to a file a quarter at a time under two watchers, one with each engine: at every look the incidents, `not_checked` and the run state are the same (numbers to 1e-12, a logged NaN as `null`) and every other key is equal ([`tests/test_watch_rust.py`](tests/test_watch_rust.py), 77 tests).
- **`--compact` and the trigger are the same.** They act on the evidence after it is built. The compact file is equal but for `"detectors": "rust"`, a planted failure starts the command once with the same payload, and five of the trigger's own tests pass again unchanged with the option given.
- **A missing binary or `pyarrow` is said before the first look**, not at the first rebuild, and the command line exits with status 2 and the reason.
- **What a rebuild costs.** One rebuild over every record read so far, timed in its parts. Medians of five on an Apple M1 Pro (8 cores, macOS) shared with other work: the load average fell from 45 to 4 during the run, and the 1,000-step rows ran at the highest.

  | Records | Python detectors | Rust, release binary | Arrow conversion | `detect` | The rest, in Python | Rust, debug binary (`detect`) |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Granite-shaped (24 routers, 32 experts, top-8), 1,000 steps, 28 MB | 766 ms | 885 ms | 304 ms | 199 ms | 383 ms | 1,229 ms (548) |
  | The same, 10,000 steps, 285 MB | 8,140 ms | 8,944 ms | 2,806 ms | 1,798 ms | 4,340 ms | 12,801 ms (5,592) |
  | Light rows (the parity tests' records, 0.8 KB a step), 1,000 steps | 91 ms | 112 ms | 18 ms | 22 ms | 72 ms | 153 ms (61) |
  | The same, 10,000 steps, 8 MB | 1,043 ms | 1,154 ms | 176 ms | 89 ms | 890 ms | 1,519 ms (469) |

  **This table was measured before the samples came from Rust**, when Python still made them on the Rust path. Measured again on the merged tree with the samples from Rust (release binary, medians of five, load average 7, records from the bake-off's generator with 24 routers of 32 experts, top-8): 435 ms through Python and 401 ms through Rust at 1,000 steps, and 4,500 ms against 3,877 ms at 10,000 steps. So a rebuild through Rust is now 8% to 14% shorter. The parts were not timed again, and the sentences below describe the table.

  Through Rust a rebuild takes 10% to 23% longer than through Python with the release binary, and 46% to 68% longer with the debug one. At 10,000 Granite-shaped steps the judging itself is the smallest part (1.8 s); before it the records are converted to Arrow (2.8 s), and beside it Python still makes the per-step samples and the other keys of the evidence (4.3 s, not timed apart). Reading the whole 285 MB file once takes 2.3 s, which a poll does not pay again.
- **What is and is not read again.** The file is tailed from an offset, so only new lines are read from disk. Each rebuild still converts every record read so far to Arrow, starts a new `detect` that judges from the first row, and makes the per-step samples again (in Rust since the merge), so it grows with the run, as the Python detectors' rebuild does. To stop that, the converted batches would have to be kept between polls, and one `detect` process be kept alive and sent only the new rows. The Rust side already appends a plain continuation to its detectors for a live stream; neither is built for the watcher.

**All ranks.** `--rank all --detectors rust` is accepted and changes nothing. That view takes one source, a run folder in a server's store, and runs no detector: every rank's incidents are read from the report its server's Rust detectors wrote. So it needs no binary, and the evidence is equal with and without the option. (It needs `pyarrow` in one case only, as before: `--flight-recorder` finding a stuck collective opens the newest stored part to date the dump.) [`tests/test_all_ranks_rust.py`](tests/test_all_ranks_rust.py) runs every test of the all-ranks view a second time with the option given (36 tests).

**Every rank of a run: `detect --run`.** `duplicate_batches` compares the rows of several ranks, which `detect` on one rank cannot do. The server judges it for a live run and writes it to `watchdog.json`. Without a server:

```sh
tensorquill-server detect --run rank0.arrow rank1.arrow     # or one stream holding every rank, on standard input
```

- **How the ranks reach it.** With `--run`, `detect` takes any number of Arrow streams and groups the rows by their `rank` column, so it does not matter how the ranks are spread over the inputs. Within a rank the rows must be grouped by attempt, oldest first, as for one rank. Rows of a second run are refused.
- **What it prints.** One JSON document: `reports` (each rank's report, exactly what `detect` gives for that rank alone), `duplicate_batches` (the finding the server puts in a run's status, or `null`), `incidents` (the same as an incident with its class), `ranks`, and `ranks_compared` (the ranks that had a hashed batch to compare, so "no duplicates" can be told from "nothing to compare").
- **From Python.** `tensorquill.server_detect.server_run_report(records)` returns the document, and `server_duplicate_batches(records)` gives the incident or `None`, sending only each record's keys and `data`. Pass every attempt of every rank: Rust leaves out what a resume rolled back.
- **Checked** in [`tests/test_run_detect.py`](tests/test_run_detect.py) (73 tests) against Python's `detect_duplicate_batches` on the same records: the cases of the data-check tests, the four-rank run with a resume that the server test uses, a run longer than the 4,096-step window, a rank that stopped early, and each of the 57 parity record sets as two ranks that share every third step's batch.
- **One difference, pinned by a test:** a step written twice within one attempt is compared across ranks by its last writing in Rust and by both writings in Python.
- **Limits.** Without `--run`, `detect` reads one rank as before and refuses a second. Every rank's rows are held in memory at once. Not run on more than four ranks, and not timed.

### Detector thresholds

The Rust detectors take their options as one JSON object, `--thresholds`, on the server and on `detect`:

```sh
tensorquill-server --store /var/lib/tensorquill --thresholds '{"overload_threshold": 0.5, "top_k": 2}'
tensorquill-server detect --thresholds '{"starve_window": 20}' < rows.arrow
```

| Key | Default | What it sets |
| --- | --- | --- |
| `overload_threshold` | `0.35` | The base assignment-share threshold for top-1 and top-2 routers |
| `drift_tolerance` | `1e-6` | The relative spread between replicas above which a parameter has drifted |
| `starve_window` | `50` | How many samples of a router the starved check sums over |
| `starve_ratio` | `0.1` | An expert is starved below this fraction of an even share of that window |
| `top_k` | `null` | Experts chosen per token: `null` is what the run recorded, a number applies to every router, and an object gives a number per router name |

- **A key left out keeps its default**, so `{}` and no argument are the same.
- **Refused, with exit status 2 before the server listens or `detect` reads its input:** an unknown key, anything that is not a JSON object, a value of the wrong type, a negative number, and a `starve_window` or `top_k` below 1.
- **Only these five are options.** A report lists every other rule's constant under `thresholds` too, but those cannot be set.
- **Every report says what was applied.** `incidents.json` and the output of `detect` list the five values under `thresholds`, and a `replica_drift` incident carries the tolerance it was judged by.
- **A reader takes the thresholds the server applied.** A reader of a server's report (`--watch` on a store, `--rank all`, `read_server_report`) gives no threshold unless one is typed, and then takes the report with whatever the server applied. Only a threshold the reader gives itself that differs from the applied one makes the report `other_thresholds`. This rule is recorded as open to change.
  - **The command line has a flag for each of the five:** `--overload-threshold`, `--drift-tolerance`, `--starve-window`, `--starve-ratio` and `--top-k`. None has a value until it is typed. On a `.jsonl` file one left out is the default in the table above.
  - **When the one-rank watcher has to judge here,** it uses the value it was given and the server's values for the rest, so the incidents do not move with who judged them. A watcher that never saw the server's report uses the defaults, and so does a read without `--watch`, which does not look at the report.
  - **`--rank all` takes all five,** `--top-k` among them. A rank whose server applied another value than the one given is `other_thresholds` and listed as unknown; nothing is recomputed there.
  - **The evidence says what was applied.** For one rank of a store, `detection.thresholds` holds the five, whether the server judged or this process did. In the all-ranks view `thresholds` holds them. When the judged ranks were not all judged by the same values, which a server restarted with other thresholds leaves until each rank sends again, `thresholds_by_rank` lists every set with its ranks and the summary says so.
  - **`top_k` left out is "not given"** like the other four, so a reader cannot insist on "what the run recorded" against a server started with a `top_k`.
  - **Checked** in [`tests/test_reader_thresholds.py`](tests/test_reader_thresholds.py) (24 tests): taking, matching and refusing, each through `read_server_report`, the one-rank watcher, the all-ranks view and the command line, and a watcher that loses the server's report and judges here with the server's values.
- **From Python**, `build_evidence(..., detectors="rust")` sends its own options this way, and `server_report(records, thresholds={...})` takes the object directly. A value the Rust rules refuse raises `ValueError` with the server's reason.
- **Checked** in `tests/test_server_detect.py`: 17 option values and one combination of all five, each on the eight record sets that hold MoE counts or a replica spread, give the same evidence on both paths, and each changes at least one incident in Python, so an ignored option would fail. A per-router `top_k` is tested on five record sets against the value recorded on every row. A live server started with thresholds matches Python across three appended stretches and a restart, and a watcher asking for the same values or for none takes its report while one asking for another value recomputes.
- **Not re-timed:** the 64-rank detector benchmark has not been rerun with the options in place.

## Compact evidence and the trigger

Built Nov 16, 2026 in `compact.py` and `trigger.py`, with small calls from `watch` and the command line.

**Compact evidence.** For an MoE run the per-step `samples` are nearly the whole evidence file. `--compact` writes the file an agent reads: every incident and every other key as they are, and of `samples` only the steps near each MoE incident.

```sh
python -m tensorquill run.jsonl --compact
python -m tensorquill run.jsonl --watch --compact --compact-window 20
```

- **What is kept.** The samples within `--compact-window` recorded steps (default 5; 0 keeps the incident's own step alone) of each `moe_imbalance` or `starved_experts` incident, for every router of those steps.
- **Where it goes.** `<input>.evidence.compact.json`, written instead of the full file. The full file is the cost being avoided and can be rebuilt from the records; under its own name, a full file already on disk is never replaced by a smaller one.
- **What says it is compact.** A `compact` key lists the window, the incidents the windows are around, the steps kept, how many samples were left out, and how many samples were flagged and the last step one was.
- **What still works.** `incident_summary` and the `Run state:` line print the same text for both files. `expert_load_table` and `export_rerun` raise `compact.SamplesOmitted`, naming the kept steps, when asked for what was left out. `--compact --rerun` records from the full evidence in memory.
- **Measured** on the detector benchmark's Granite-shaped run (24 routers, 32 experts, top-8) at 10,000 steps with two MoE incidents planted:

  | File | Size |
  | --- | --- |
  | Full | 86.5 MB |
  | Compact, window 5 | 195 KB |
  | Compact, window 0 | 22 KB |
  | No samples at all | 3.9 KB |

  The 1 MB line holds up to a window of 28 steps.

  These were measured before the evidence said what was checked. `kind_checks` adds 3.0 KB to every file: measured again as compact JSON, the same run gives 197,570 bytes at a window of 5, 24,541 at 0 and 6,495 with no samples, and the 1 MB line still holds up to 28 steps.

**The trigger.** With `--watch`, `--on-incident COMMAND` starts a command and `--webhook URL` posts JSON, once for each new incident. Nothing is started or sent without one of the two.

```sh
python -m tensorquill run.jsonl --watch --compact --on-incident 'my-agent "Explain the $TENSORQUILL_INCIDENT_KIND incident in $TENSORQUILL_EVIDENCE"'
python -m tensorquill store/runs/r --rank all --watch --webhook https://example.com/hooks/tensorquill
```

- **New** means its rank, kind and step have not fired before. What fired is kept in `<evidence>.fired.json`, so a restarted watcher does not fire again; delete the file to start over.
- **What fires.**
  - Invariants only, by default.
  - `--fire-anomalies` adds the anomalies, whose thresholds are not tuned on other jobs yet.
  - `--fire-kind KIND` adds one kind by name, and is the only way an observation fires.
  - A stalled job (`job_stalled`) and a silent rank (`rank_silent`) are observations, so **a hang does not fire unless you name them**.
  - A change of run state never fires, and neither does a kind this version does not know.
- **The command** is one string run by the shell.
  - It gets the incident as JSON on standard input, and `TENSORQUILL_EVIDENCE`, `TENSORQUILL_RUN_ID` and `TENSORQUILL_INCIDENT_ID`, `_KIND`, `_CLASS`, `_STEP`, `_RANKS` and `_SUMMARY` in its environment.
  - It is started and not waited for, so a long agent run does not hold the watcher. The watcher never stops it.
- **The webhook** is one POST of `application/json` with the standard library, 10 s by default (`--webhook-timeout`). Redirects are not followed, and the URL is never printed or stored. The body carries the evidence file's absolute local path.
- **Failures** are printed and never stop the watcher. A command that cannot start or exits with a status other than 0, and a post that fails, is tried again on later looks, 4 tries in all.
- **Limits.**
  - An incident counts as fired once its command has started, so a command that fails while the watcher is down is not tried again.
  - The trigger does not turn on `--compact`: the file handed over is the one the watcher writes.
  - Two watchers must not share one evidence path; the state file is not locked.
  - In the all-ranks view an incident several ranks share fires once for those ranks, and once more for ranks that show it later.
- **Checked** in `tests/test_compact.py` (9 tests) and `tests/test_trigger.py` (13 tests): a planted failure in a growing file starts a command with no human step, exactly once across later looks and a restart; the webhook's body against a local HTTP server; a failing command and an unreachable webhook leave the watcher running.
- **Not checked:** a real agent as the command, an outside webhook, https, a proxy, Linux and Windows.

## Data progress and checkpoints

Built Nov 16, 2026, on CPU; no GPU run yet. Both are opt-in.

- **Data.** Iterate `session.watch_loader(loader)` instead of `loader` to record which samples each step drew (a 64-bit BLAKE2b of their indices, never the samples) and the sampler's epoch. For an `IterableDataset` pass `ids=lambda batch: batch["id"]`; `session.data(...)` records them by hand. Measured on CPU: 4 µs per batch of 32 indices, 48 µs per 1,024.
- **Checkpoints.** Call `session.checkpoint(state)` with the dict you pass to `torch.save`, and `session.checkpoint(state, event="load")` with the one you resume from, before the first step; pass `start_step=` as for any resume. The Lightning, `Trainer` and Accelerate adapters record checkpoints themselves; none of them turns the loader hook on.
- **What is reported.**
  - `epoch_order_repeated`: an epoch in the previous epoch's order (`set_epoch` not called).
  - `data_stalled`: steps advance without new data.
  - `duplicate_batches`: two ranks drew the same batch (in `watchdog.json`, compared across ranks).
  - `resume_data_mismatch`: a resume repeated or skipped data.
  - `resume_incomplete`: a resume did not restore a component (optimizer state, scheduler, scaler, RNG). The last two are observations, judged at a resume's first step as `resume_lr_change` is.
- `python examples/data_checkpoint_check.py` shows all of it on CPU.
- **Limits.**
  - A save made just after a flush is recorded with the next step and is lost if the process ends first.
  - Per-epoch schedulers trip the scheduler check (they count epochs).
  - Lightning saves no RNG state by default, so its resumes list `rng` missing.
  - The server compares duplicates only among ranks it has read since it started, within each rank's last 4,096 hashed steps.
  - Wrapping the loader before handing it to Lightning or the `Trainer` is untested.

## Attention rules and run state

Built Nov 16, 2026.

- **Attention rules.** With `probe=`, three kinds are read from the probe's records, in Python and in the server:
  - `attention_collapse`: a block's mean attention-entropy share at most half its own baseline.
  - `attention_uniform`: the share risen to 0.98 or more, at least 0.2 above its baseline (attention that had structure spread evenly over every key).
  - `attention_logit_growth`: the largest pre-softmax logit at least twice its baseline.

  A block's baseline is the median of its last five probes that met no rule. A change must hold on two probes in a row and is dated at the first. With the default `probe_every=1000` nothing can be reported before the 4,000th step, and `not_checked` says so until then.

  Calibrated on the recorded SmolLM2-360M seeds (the blocks after the seed go uniform, 0.23 to 1.00, or collapse, 0.30 to 0.05) and on small models trained from scratch on CPU. The logit rule's margin is thin: healthy training moved the largest logit 1.7×, and the rule needs 2.0×. Nothing about MoE models was calibrated (the Granite captures have no probe records). `probe_json` now keeps the model's module order.
- **Run state.** `evidence.json`, the all-ranks file and the server's `watchdog.json` carry `run_state`; the worst reason wins, and each reason is listed with a stable `code`:
  - `healthy`: every expected source is current and no incident is listed. A `.jsonl` read alone can be healthy: no server was watching it, so its stall kinds are not configured (see "What was checked").
  - `degraded`: a kind is unavailable (anything in `not_checked`, or no `watchdog.json` where a server stores the run or a status was named).
  - `unknown`: required evidence is stale or missing (no steps judged, a server report that is not current, a rank without a current report or not yet connected under `world_size`, an old status, a stopped server, a status this version does not know).
  - `unhealthy`: an invariant or anomaly is listed (or a kind this version does not know).

  Every incident carries a `class` (the table is `tensorquill.evidence.INCIDENT_CLASSES`, the same in the server):
  - `invariant`: should never happen, such as `non_finite`, `replica_drift`, `duplicate_batches`, `epoch_order_repeated` or `hardware_errors`.
  - `anomaly`: judged against the run's own baseline, such as spikes, drift, collapse, attention, memory growth or input starvation.
  - `observation`: context that claims no failure. These are `resume_lr_change`, `resume_data_mismatch`, `resume_incomplete`, `gpu_throttled`, `memory_imbalance`, `memory_pressure`, `allocator_retries`, `job_stalled` and `rank_silent`.

  Observations are listed under the reason `observations` and never make the state worse, so a run with only observations reads `healthy` with them named. A stall is an observation of the stream (a hang, a long pause and a dead sender look the same), so a `stalled` or `degraded` status makes the run `unknown`, not `unhealthy`. The CLI and `--watch` print the state as `Run state: ...`, and the summary adds a `By class:` line. Incidents are never closed, so a run stays unhealthy once an invariant or anomaly is found. A `watchdog.json` that is no longer being rewritten says nothing about now: read its age (`written_at_ms`, `refresh_seconds`) before its `run_state`.

## What was checked

Built Nov 16, 2026. "No incidents" means little without knowing what could have been found: a router-only capture cannot show stalled weights, and a run without `scaler=` has no loss-scale check. Every evidence file now gives each of the 32 kinds one of three values, under `kind_checks`:

| Value | Meaning | Run state |
| --- | --- | --- |
| `checked` | Records carried what the kind is judged from, and its rule ran over them. For a kind read from a file (the server's status, Flight Recorder dumps, TrainCheck's results), the file was given and read | Unchanged |
| `not_configured` | The capture was never set up to provide it: no record carries the input, or the file was not given | Unchanged |
| `unavailable` | It was asked for and could not be judged: a `not_checked` entry names the kind, or a server stores the run and its progress status is missing | `degraded` |

```json
"kind_checks": {
  "moe_imbalance": {"value": "checked", "records": 60, "needs": "moe: a router's expert counts"},
  "stalled_weights": {"value": "not_configured", "records": 0, "needs": "weights_unchanged with grad_norms (full capture)"},
  "attention_collapse": {"value": "unavailable", "records": 5, "needs": "probe with attention (probe=)",
                         "check": "probe", "reason": "1 of 6 probes failed, the first at step 30: RuntimeError: CUDA out of memory"},
  "job_stalled": {"value": "not_configured", "needs": "the server's progress status (watchdog.json)"}
}
```

- **The summary carries the count.** `python -m tensorquill` prints a line such as `3 of 32 kinds checked, 29 not configured`, with the unavailable kinds by name when there are any. `tensorquill.evidence.checked_counts(evidence)` gives the numbers and `kinds_by_value(evidence)` the names.
- **How a value is worked out.** For the 25 kinds read from one rank's step records, `records` counts the records that carried the kind's input, and `needs` names it (`tensorquill.evidence.KIND_INPUTS` and `INPUT_NEEDS`; `input_counts(records)` does the counting). "Carried" means the field holds something a rule reads: an empty `layers` does not count, and a probe that failed carries no attention.
  - A kind that a `not_checked` entry names is `unavailable`, with that entry's `check` and `reason`, however many records carried its input. One failed probe leaves its step unjudged, as one failed memory reading does.
  - Otherwise it is `checked` when any record carried the input, and `not_configured` when none did.
- **A failed probe is now listed.** A probe that raised is recorded as `{"error": ...}`. It used to be skipped in silence. It is now a `not_checked` entry with the check `probe`, the three attention kinds, how many probes failed and the first error, so the run reads `degraded`.
- **The other seven kinds.**
  - `job_stalled` and `rank_silent` are `checked` when the server's status was read, `unavailable` when one was expected and is missing, and `not_configured` for a local file that no server was watching. A status is expected when the source is a directory of the server's parts or one part, or when `watchdog=` names one.
  - `collective_hang`, `collective_desync` and `traincheck_violation` are `checked` when `flight_recorder=` or `traincheck=` was given and read, `unavailable` when it could not be read, and `not_configured` when it was not given.
  - `duplicate_batches` and `memory_imbalance` compare ranks, which one rank's evidence cannot: `not_configured`. The all-ranks view judges them.
- **A rule that changed: a local file can read `healthy`.** Before, any evidence built without a server's `watchdog.json` read `degraded` (reason `no_status`). Now a `.jsonl` file or a list of records that no server was watching has its stall kinds `not_configured` and no such reason. A server's store whose status is missing, or a named status that cannot be read, still reads `degraded` with `no_status`.
- **In the server's report too.** `incidents.json` and the output of `tensorquill-server detect` carry `kind_checks` for the 25 kinds the server has, made by the same counts. `checked` (the kinds this server can judge) and `not_checked` keep their names and shapes. `build_evidence(..., detectors="rust")` takes the 25 from the report and adds the other seven.
- **Checked** in `tests/test_kind_checks.py` (17 tests, Python only) and `tests/test_server_detect.py`: Python and Rust give the same value, count and reason for every kind on the 57 record sets, the seven Gate 1 fixtures and seven more sets (a router-only run, and probes that are all good, one failed, all failed, too few, or failed with an error that is not text). Every one of the 25 kinds is `checked` in some set and `not_configured` in another, and the 11 kinds a `not_checked` entry can name are each `unavailable` in one. The golden files hold `kind_checks` for both paths.
- **Limits.**
  - A count is of records that carried the input, not of rules that had enough history. A kind is `checked` from its first record, although `grad_spike` needs ten recorded steps of history and `starved_experts` a full window. Only the attention kinds say "too few to judge", as they did before.
  - Evidence built from the server's report by `--watch` carries `kind_checks`: the report's 25 and the reader's seven, with a status expected. A report from a server older than the key gives evidence without it. The all-ranks view does not carry `kind_checks` yet. The server's own `watchdog.json` run state does not read it either; it still degrades on `not_checked`, which gives the same state.
  - A resume in a run that recorded no learning rate reads `unavailable`, not `not_configured`, because `not_checked` has always listed it.

## GPU memory, NVML and data wait

Built Nov 16, 2026, tested with mocked CUDA and NVML; the GPU side waits for a Colab run.

- **Memory and NVML.** `tensorquill.init(model, system=True)` records, at each flush, the CUDA allocator's counters (`allocated`, `reserved`, `peak_allocated`, `alloc_retries`, `ooms`). With `pip install "tensorquill[nvml]"` it also records NVML's temperature, SM clock, power, ECC counts, throttle reasons and Xid events, read on the writer thread. Both go into `system` on the flush's last record. Reading them never waits for the GPU. A reading that cannot be taken is written once as unavailable and listed in `not_checked`. It is off by default until its GPU cost is measured.
- **Data wait.** `data_wait=True` records `data_wait` and `step_time` per step: the host time spent fetching batches, never the move to the device. It is on in the Hugging Face, Lightning and Accelerate adapters, at +0.5 µs per batch on CPU. For a loader that is not a PyTorch `DataLoader`, use `for batch in session.time_data(loader):` and move the batch to the device inside the loop.
- **What is reported.**
  - `memory_growth`: allocated memory rose at six readings in a row, by at least 5% and 64 MiB.
  - `memory_pressure`: reserved memory at 95% or more of the device total.
  - `allocator_retries`: allocator retries or caught OOMs.
  - `hardware_errors`: uncorrectable ECC or Xid events, as evidence about the node.
  - `gpu_throttled`: thermal or power-brake throttling, an observation; the power cap alone is never reported.
  - `input_starvation`: over 20 steps, the median share of step time spent waiting for data is 0.3 or more, or has risen 0.2 above the first window's.
  - `memory_imbalance`, in `--rank all` only: one rank's peak allocated memory is 1.25× the median and 1 GiB above it.
- **Limits.**
  - Only the session's device is read, so a model spread over several GPUs in one process gets one device's memory.
  - Throttling between flushes is missed.
  - A hung NVML call blocks the writer thread and, once its queue fills, training.
  - Only PyTorch's `DataLoader` iterator and `time_data` are timed.

## For a coding agent: a read-only MCP server

Built Nov 16, 2026. `python -m tensorquill.mcp_server PATH` serves one run's evidence to a coding agent over MCP (stdio). `PATH` is an `evidence.json`, a `.jsonl` capture, or a run folder in a server's store (`runs/<run_id>`). It needs no package beyond the SDK: the part of the protocol it uses (revision 2025-06-18) is written out in `mcp_server.py`.

It has four tools, and all four only read:

| Tool | Gives |
| --- | --- |
| `list_incidents` | The incidents with their kind, class, first step and what they name, plus the run state and what was not checked |
| `get_incident` | One incident in full |
| `get_series` | One layer's, parameter's, router's or logged scalar's values over a step range |
| `get_all_ranks_status` | For a run folder: which ranks are judged, unknown, clean or showing incidents |

- **It follows a live run.** The path is read again at every call.
- **Answers are bounded.** No reply is longer than 50 KB. A long series is thinned to 200 points with the stride reported, and the minimum, maximum and median are still taken over every recorded step.
- **For a run folder it runs no detector.** The incidents come from the server's own reports, and only `get_series` opens the Parquet parts, which needs `pyarrow`.
- **The page to give the agent** is [AGENT_INSTRUCTIONS.md](AGENT_INSTRUCTIONS.md): which tool to call first, how to name the step, the layer or parameter and the kind, what the three classes and "not checked" mean, and that it explains and takes no action on the run.
- **Checked** in `tests/test_mcp_server.py` (29 tests): a scripted client using only the four tools recovers the kind, the first step and the layer, parameter or module of all seven Gate 1 fixtures; the size cap on a 10,000-step run; a two-rank run folder, by hand and through the real server; every error case.
- **Not checked:** no live agent has used it.

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

The cause is the seed, not a missed reading. The block's output had `abs_max` about 20,000 and `std` 31 to 37 at every step before the seed (the model's own massive activations, about 640 times the `std`), and scaling two channels of 960 by 100 moved the block's `std` by 1.07 to 1.14 times and its `abs_max` by 0.95 to 1.07 times, against a threshold of 5. The seed was real: the loss went from about 2.3 to 6.9 on step 250, and the `GradScaler` skipped steps 250 to 253 (scale 65,536 down to 8,192), so the gradient-norm detector first flagged step 254. The block summary (`std`, `abs_max`, `mean`, `zero_frac` of the whole output) cannot see growth confined to a few ordinary channels once the output holds large activations; it would see growth in the massive ones or a block-wide change. The toy Llama has 64 channels, so the same seed was a 13 to 42 times change there. Decided and changed the same day: the check now seeds the channels with the largest outputs by default (`--fault massive`), or the whole block (`--fault block`); `--fault channels` keeps the old seed, and a test builds this situation in miniature and runs the detector on it. The three faults pass on the CPU toy. Rerun on the A100 (same setup as above, commit `d987509`) with `--fault massive` (channels 87 and 398) and `--fault block`: **all ten checks passed in both**. The loss went from 2.32 to 8.46 with 0 steps skipped (`massive`), and to 13.27 with steps 250 to 252 skipped (`block`). The activation spike is at step 250 and the drift at step 254, each covering blocks 16 to 31, and the first gradient spike is at step 250 (`massive`) or 253 (`block`), first in `model.norm.weight`. On this run the spike listed modules by ratio, so it led with block 31, not the seeded 16; fixed the same day (it now leads with block 16 and names block 31 as the largest). The check still only requires the seeded block to be among the listed modules. `--fault channels` was not rerun on a GPU. On an RTX PRO 6000 (commit `2b56eac`, with the check tightened to require the spike to lead with the seeded block) both passed again with the same losses (8.46 with 0 steps skipped, and 13.27 with 3 skipped); each led with block 16 and named block 31 as the largest (722× and 720×).

On a CUDA machine, run the GPU check. It fails on any graph break, recompile or CUDA-graph skip, and then reports the time per step with and without capture:

```sh
python examples/gpu_check.py --dim 1024 --blocks 8 --batch 4096
```

Set the sizes close to the real job, because a small model makes the fixed per-step cost look large.

## License

TensorQuill is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in TensorQuill, as defined in the Apache-2.0 license, is dual licensed as above, without any additional terms or conditions.

## No telemetry

The SDK sends no usage statistics, update checks or crash reports. Its only network traffic is the opt-in Arrow Flight sender, which is off unless you set `endpoint=` or `TENSORQUILL_ENDPOINT`, and which connects only to that address (checked with an audit hook that raises on any non-loopback socket use, on a clean install of the wheel running `toy_moe.py`). `tensorquill-server` listens on the address you give it and writes to the store you configure (local disk, or S3 or GCS if you choose one); a reading of its source and its dependency list found no analytics or crash-reporting code, but it has not been run under a network trace. Third-party tools you use alongside TensorQuill, such as the Rerun viewer, have their own settings.
