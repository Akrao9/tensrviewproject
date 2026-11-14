# TensorLens capture SDK

Skeleton of the capture SDK from the [v3 plan](../TensorLens%20v3%20—%20Product%20and%20Build%20Plan.md). It records what happens inside a PyTorch model during training, cheaply enough to leave on, and in a way that survives `torch.compile`.

```python
import tensorlens

model.to(device)
session = tensorlens.init(model, routers={"blocks.3.router": 128})  # before torch.compile
model = torch.compile(model)

for batch in loader:
    loss = model(batch).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()        # TensorLens counts the step here

session.close()
```

Each optimizer step becomes one JSON line in `tensorlens-out/<run_id>.rank<rank>.jsonl`:

- **`layers`**: mean, standard deviation, absolute maximum and zero fraction of each watched layer's output.
- **`grad_norms`** and **`grad_norm`**: the L2 norm of each parameter's gradient and their total.
- **`weight_norms`**: the L2 norm of each parameter after the optimizer step, on the steps that record gradient norms (`weights=False` turns it off).
- **`lr`**: each parameter group's learning rate after the step.
- **`moe`**, **`moe_stats`** and **`moe_top_k`**: tokens routed to each expert, plus router entropy, diversity (`exp(H)/N`), and dead-expert count/fraction, for every router you name (works whether the router outputs integer indices or floating-point logits with `Router(n_experts, top_k=...)`). Set `top_k` for top-k routers; the detectors need it.

## What is built

| Part | State |
| --- | --- |
| Forward hooks into preallocated on-device buffers, marked static | Built, tested on CPU (including AOTAutograd `static_input_indices` check) |
| On-device ring buffer, moved to the CPU every `every` steps, written by a worker thread | Built, tested |
| Step counting through PyTorch's global optimizer hooks, limited to this model's optimizer | Built, tested |
| Gradient norms grouped by device and dtype (`torch._foreach_norm`, with `DTensor.to_local()` and fallback) | Built, tested |
| MoE expert load (`scatter_add_`) and router entropy / diversity (`moe_stats`) | Built, tested |
| Cross-GPU parameter fingerprints and replica spread (`parameter_fingerprints`, `replica_spread`) | Built, unit-tested on drift and sign-flip cases |
| Detectors and Rerun exporter (`build_evidence`, `export_rerun`, `python -m tensorlens`) | Built, tested on MoE collapse (top-k aware), starved experts, gradient spikes, weights not updating, repeated steps, NaN/Inf, and replica drift |
| Run ID from `run_id=`, `MLFLOW_RUN_ID`, the active MLflow run or the active W&B run | Built, tested with stand-in tracker modules |
| No graph breaks and no recompiles under `torch.compile` | Tested with `eager`, `aot_eager`, and AOTAutograd static-input checks on CPU; on CUDA with Inductor and CUDA graphs on a toy MoE, and with Inductor and CUDA graphs off on Granite |

## What is not built or not verified

- **Arrow Flight and the Rust ingest server.** Records go to a JSON Lines file, and Python detectors/exporters turn that file into `evidence.json` and `.rrd`.
- **The probe pass** for histograms and attention (Tier 2) and replay (Tier 3).
- **Multi-rank reporting.** Every rank writes its own file.
- **A PyTorch Lightning callback.** Until then, pass `run_id=` yourself when the run was created through `MlflowClient`.
- **Detector thresholds are untuned.** The gradient-spike factor (5× the median of the last 50 recorded steps), the top-k diversity drop (0.6 of its recent median), the starved-expert ratio and the stalled-weight check have been checked on synthetic cases and, for the MoE rules, on one Granite capture. Weight norms, learning rates and the spike and stalled-weight detectors have not run on a GPU.
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
- **Weights not updating.** After each recorded step the capture computes every parameter's norm in float64 and compares it with the previous record; `weights_unchanged` lists the parameters whose norm is exactly the same. A parameter with a finite non-zero gradient that is unchanged on 3 recorded steps in a row is reported, for example because it was left out of the optimizer. Steps with a non-finite gradient norm are skipped, since mixed-precision loss scaling skips those on purpose. The first version compared float32 norms, and on the A100 run below it reported 147 of 218 parameters from step 1: a bf16 weight under a 1e-5 AdamW step changes in only a few elements per step, which a float32 norm often misses (`model.layers.7.self_attn.k_proj.weight` kept the same float32 norm for 128 steps while changing). Float64 norms resolve those changes. bf16 RMSNorm weights near 1.0 are still reported, correctly: their step size in bf16 is about 0.008, so a 1e-5 update rounds away, and in a CPU simulation such a weight was exactly unchanged on 273 of 650 steps. One read of the weights per recorded step, no copy, accumulated in float64; on Apple MPS, which has no float64, the check is off. Captures made before `weights_unchanged` existed are not checked. The extra read is new overhead, so gradient-norm costs measured before it are not comparable.
- **Gradient spikes.** A step whose total gradient norm is at least 5× the median of the previous 50 recorded steps. The report ranks parameters by their own spike and names the one that started rising first.
- **Repeated steps.** Step numbers written twice by the same rank, as after a resume without `start_step`.
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
- **The float64 weight norms are expensive on this GPU.** Gradient norms alone cost +2.60% (+1.79 ms) here at `5d38288`; with float64 weight norms the same arm costs +8.13% (+5.53 ms), so the weight check adds roughly 3.7 ms per recorded step. This GPU runs float64 arithmetic far slower than float32; the A100, with fast float64, paid less. The check needs a cheaper exact test before it can stay on by default.
- **Router-only capture** was within noise of capture off: paired rounds −0.50, +0.46 and −0.004 ms. With the earlier +0.78% on this GPU and +1.06–1.33% on the A100, the 1% target holds here and not on the A100.
- **Gradient spikes at steps 141 and 645**, starting in `model.layers.11.self_attn.k_proj.weight` (22× its median). The A100 run flagged 136, 141 and 645. The same steps on two GPUs point to the data at those steps, since both runs read the same batches in the same order.
- **The collapse** was found at step 450 in layer 12 with the same 20 starved experts.

## Two-GPU replica drift validation

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

## Things to know

- **Mixed precision.** `torch.amp.GradScaler` skips `optimizer.step()` when gradients overflow, so the automatic step count misses those steps. Pass `auto_step=False` and call `session.step(lr=...)` once per iteration.
- **Resuming.** Pass `start_step=` the step you resume from. Otherwise step numbers start at 0 again, and the evidence reports them as repeated steps.
- **Clipping.** Gradient norms are read just before `optimizer.step()`, which is after clipping. Call `session.record_grads()` before you clip to record the unclipped norms.
- **Gradient accumulation.** A hook runs on every forward pass; the record keeps the last one before the optimizer step.
- **Non-finite values** are written as the strings `"nan"`, `"inf"` and `"-inf"`, so the file stays standard JSON.
- **Hooks stay on the model after `close()`**, because removing a hook from a compiled module triggers a recompile.

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

On a CUDA machine, run the GPU check. It fails on any graph break, recompile or CUDA-graph skip, and then reports the time per step with and without capture:

```sh
python examples/gpu_check.py --dim 1024 --blocks 8 --batch 4096
```

Set the sizes close to the real job, because a small model makes the fixed per-step cost look large.
