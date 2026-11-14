# TensorLens v3 — Product and Build Plan

Nov 13, 2026 · @Ak B

TensorLens lets ML engineers and their coding agents see inside a PyTorch training run, so a failing run is caught and explained before it wastes days of GPU time. "TensorLens" is a placeholder name. This version replaces the larger v2 plan.

## Product

TensorLens owns the capture SDK, detectors, and the evidence they produce. Stock Rerun displays that evidence, and the team's coding agent uses it to investigate incidents.

- **Who it is for:** teams training large models on their own clusters with MLflow or W&B. Start with RL, robotics and mixture-of-experts teams, whose important signals are not loss curves.
- **The pitch:** add one line to your PyTorch script and see inside a failing run (activations, attention, MoE routing), with evidence an agent can read.
- **Against trackers:** MLflow and W&B show scalar curves, and their MCP servers let coding agents query those runs. This shows what happens inside the layers.
- **Against CoreWeave ARIA:** ARIA is an agent inside W&B that analyses experiments, proposes next runs and launches them with approval (public preview June 2026). It reasons over what W&B logged, mostly scalars, and is tied to W&B and CoreWeave. This captures layer internals and works with whichever tracker, agent and cloud the team already has.
- **Against Trainy Pluto:** Pluto is a Neptune replacement built for responsiveness with many runs, and its experimental MCP server lets an agent read per-layer panels in one call. It displays what the user logs; no capture SDK or failure detectors turned up. This captures the internals cheaply under `torch.compile` and FSDP and detects known failures.
- **Against debugging agents:** free add-ons and research agents such as RunSleuth read logs, loss curves and gradient norms. None found inspects activations, attention or MoE routing. This gives those agents a picture and the numbers behind it.

**The gap.** A search on Nov 13, 2026 found no product that captures layer internals (activations, attention, MoE routing, cross-GPU drift) and hands an agent the evidence to explain a failure. Public requests for such a tool were not found either, so demand is unproven; the closest signal is that CoreWeave and Trainy are both building agent access to training runs.

**Why now.** OpenAI bought Neptune, whose product logged thousands of per-layer metrics to debug unstable runs, and the hosted service shut down in March 2026. Teams that relied on it said the hard requirement was a tool that stays responsive with thousands of runs and heavy logging.

## How it works

Capture signals inside the training script and detect failures on the ingest server. Export the evidence as JSON for agents and as a recording and blueprint for engineers to explore in Rerun.

&#91;embedded content: How it works · 7 parts, 8 flows\]

Scalars keep going to the tracker the team already uses. Tensor summaries take their own path, so a slow tensor pipeline never blocks metric logging.

## Build and borrow

Start with stock Rerun and reuse the MoE recording and blueprint from the bake-off. The first product work is capturing those signals from a real training run and generating the detector evidence.

| Piece | Build or borrow | Implementation | Note |
| --- | --- | --- | --- |
| Capture SDK | Build | Python | One line to add. Tags the tensor stream with the MLflow run first; W&B as a later target. Built in `sdk/`; writes JSON Lines until the ingest server exists |
| Detectors | Build | Python for v1 | MoE routing imbalance, non-finite values and cross-GPU drift. Built in `sdk/src/tensorlens/evidence.py` and run on the JSON Lines files. Move to Rust only if the ingest server needs them in-process |
| Ingest server | Build | Rust | Receives Arrow data and stores Parquet. Not started |
| Evidence and Rerun export | Build | Viewer-neutral JSON, `.rrd`, Rerun blueprint | Export the incident summary, status heatmap, expert-load table, and training-step timeline. Built from the bake-off's exporter (`python -m tensorlens`) |
| Incident view | Conditional | Rust (`re_viewer`, egui) | Prototype one custom view inside Rerun if trials show that layer selection and table sorting materially improve diagnosis |
| MLflow importer | Build | Python (`mlflow`, `rerun-sdk`) | Writes runs logged without the SDK to `.rrd` with the same exporter, so they open in Rerun |
| Viewer and screenshots | Borrow | Rerun (MIT / Apache 2.0) | Stock desktop viewer, built-in views, timeline, headless screenshots, and viewer MCP |
| Run metadata | Borrow | MLflow or W&B | Params, lineage, artifacts |
| Diagnosis | Borrow | The team's own coding agent | Reads TensorLens JSON and inspects the recording through Rerun's viewer MCP |

Postponed from v2: your own VLM agent loop, the agent-written layout blueprints, live fixes to a running optimizer, and a browser client of your own.

## Capture SDK

Inside the compiled graph, hooks compute only cheap statistics of fixed size. Copies to the CPU, Arrow encoding and anything bigger happen outside the compiled step.

**Setup for the user:** `pip install` and `tensorlens.init(model)`, with the team's MLflow configuration left unchanged. `init` tags the tensor stream with the tracker's run ID, and the SDK records its own global step with every batch.

- **Run ID.** Take an explicit `run_id=` first, then `MLFLOW_RUN_ID` or `mlflow.active_run()`. `active_run()` is thread-local and is `None` whenever a run was created through `MlflowClient`. PyTorch Lightning's `MLFlowLogger` works that way and only creates the run on rank 0 (`@rank_zero_experiment`), so offer a Lightning callback that reads `trainer.logger.run_id` on rank 0 and broadcasts it to the other ranks.
- **Step.** An MLflow run has no current step; the step is passed with each `log_metric` call. Count optimizer steps with `register_optimizer_step_post_hook` (imported from `torch.optim.optimizer`, since `torch.optim` deletes the `optimizer` submodule attribute; that way `init(model)` needs no optimizer argument), or take `step=` from the caller or from `trainer.global_step` in the callback. The hook count misses steps that mixed precision skips; see the rules below.

Do not ship a tracking-store plugin with its own URI scheme (`tensorlens+https://…`). MLflow selects the plugin by scheme, but its proxied artifact store (`mlflow-artifacts:`) accepts only `http` and `https` tracking URIs, so `log_artifact` would fail for teams that use it. The model registry would also need its own plugin.

**Rules**

- Register forward hooks **before** `torch.compile` and never change them afterwards. With default settings, hooks added to a module that had none when it was compiled are silently ignored (`skip_nnmodule_hook_guards`). On a module that already had hooks, Dynamo guards the hook dictionary, so adding, removing or replacing one triggers a recompile.
- Hook bodies compute mean, std, absolute max and zero fraction per layer, and write into GPU buffers allocated once and marked with `torch._dynamo.mark_static_address`. No histograms, `.cpu()`, numpy or I/O inside a hook. With gradient accumulation or activation checkpointing, a hook runs more than once per optimizer step and keeps the last forward pass's values.
- No backward hooks. Read `p.grad` norms in an optimizer step pre-hook (`register_optimizer_step_pre_hook`, before `step()` or `zero_grad()`) with `torch._foreach_norm(..., dtype=torch.float32)`. Group gradients by `(device, dtype)` first, as `clip_grad_norm_` does, or a mixed-dtype list falls back to the slow per-tensor path; unwrap `DTensor`s with `.to_local()`. `_foreach_norm` is a private API, so test it on each supported PyTorch release and fall back to `torch.linalg.vector_norm` per tensor.
- The global optimizer hooks have two blind spots:
  - **Every optimizer.** They fire for every optimizer in the process, including ones for other models, so keep only the parameters of the model passed to `init`.
  - **Overflow steps.** With mixed precision, `GradScaler.step` skips `optimizer.step()` when gradients overflow, so neither hook fires. Those are the steps that matter most, and a step counter built on the post-hook falls behind the trainer's. Record skipped steps from the scaler's `found_inf` or from the training loop, and prefer the caller's or Lightning's step over the hook count when one is available.
- The pre-hook runs after any gradient clipping the user does, so it records clipped norms, which hide gradient explosions. Label Tier 0 gradient norms as post-clipping, or read them before clipping through the Lightning callback (`on_before_optimizer_step` runs before Lightning clips) or an explicit `tensorlens.record_grads()` call.
- Fixed shapes only inside compiled hooks: `scatter_add_` into `zeros(n_experts)` is fine; `bincount` and `nonzero` have data-dependent output shapes and are not.
- After each optimizer step, copy the hook buffers into row `step % N` of a GPU ring buffer, outside the graph. A step counter used inside the compiled region would force a recompile on every step.
- Every N steps, after `optimizer.step()`, move the float32 ring buffer to the CPU with one `.cpu()` call (plus a second `.cpu()` call for int64 MoE expert counts when routers are watched, since packing counts into float32 loses exactness above $2^{24}$ tokens). A worker thread builds the Arrow batch and sends it over Arrow Flight.
- Add pinned buffers and a side CUDA stream only if the overhead gate shows a stall.
- Histograms and attention come from a probe pass: every M steps, one fixed batch through the wrapped model under `no_grad` and `torch.compiler.set_stance("force_eager")` (PyTorch 2.6+). That runs the same FSDP- or TP-wrapped model without the compiler. On older PyTorch the fallback is `model._orig_mod`, which only unwraps an outer `torch.compile(model)`; it does not bypass blocks compiled one by one, as is common with FSDP2's `fully_shard`, and it is untested with sharded parameters. `no_grad` does not stop dropout, batch-norm updates or router noise, so snapshot each submodule's `training` flag, call `model.eval()` for the probe, and restore the saved flags afterwards (a blanket `model.train()` would unfreeze submodules kept in `eval()` and trip Dynamo's `training` guards). On CUDA, `torch.histc` rejects `bf16` and `fp16` tensors, syncs with the CPU when `min == max`, and errors under `torch.use_deterministic_algorithms(True)`, so cast probe activations to `float32` and pass explicit `min < max` bounds. To capture attention maps on the probed layers, temporarily interpose `F.scaled_dot_product_attention` during the probe so it computes explicit `softmax(QKᵀ)` instead of calling the fused kernel.
- Run the probe after the ring-buffer copy, because the hooks fire during the probe too and overwrite the step's statistics.
- With plain data parallelism, rank 0 reports the Tier 0 statistics. With pipeline or tensor parallelism, every rank that owns layers reports its own, or layers on other ranks go unseen.
- CI gate: `TORCH_LOGS="graph_breaks,recompiles,cudagraphs"` must show zero breaks and recompiles with the SDK on. With `mode="reduce-overhead"`, the log must also not contain "skipping cudagraphs due to mutated inputs". A hook's in-place write into a preallocated buffer is an input mutation unless the buffer is a module buffer or marked with `torch._dynamo.mark_static_address` (with `cudagraph_trees` enabled), which can otherwise turn CUDA graphs off while breaks and recompiles stay at zero. The skip is logged as a warning, so it prints without `cudagraphs`; adding it also shows the non-warning CUDA-graph messages.

**Data tiers**

| Tier | Data | Computed how | When | Approx. size |
| --- | --- | --- | --- | --- |
| 0 | Loss, grad norms, router entropy, per-layer activation mean, std, absolute max, zero fraction | Hooks inside the graph; `_foreach_norm` after backward | Computed every step, sent every N steps | \~1.3 KB per step for 80 layers |
| 1 | MoE expert load table, dead-expert fraction, diversity exp(H)/N | `scatter_add_` on top-k indices, summed across data-parallel ranks | Every N steps | \~30 KB for 60 layers × 128 experts |
| 1 | Cross-GPU spread: norm and sum per replicated parameter tensor | Per-rank float64 norm and sum, one `all_gather`, spread computed on rank 0 | Every N steps | 16 bytes per tensor per rank |
| 2 | Activation histograms (64 bins) per layer | Probe pass, `histc` per layer | Every M steps | \~20 KB for 80 layers |
| 2 | Attention maps, entropy and max logit for a few layers and heads | Probe pass with explicit softmax(QKᵀ); maps downsampled to 256×256 | Every M steps | \~0.5 MB per layer |
| 3 | Full activations for the layers a detector flagged | Replay the flagged batch from a ring buffer of batch IDs and seeds | Only when a detector fires | MB to GB, written to Parquet |

Attention needs the probe because FlashAttention and SDPA never store the attention map. The probe uses a fixed batch, so it compares cleanly across steps but does not show the live batch; the Tier 3 replay covers that. Sizes are estimates.

## W&B support

Support W&B from inside the customer's own training script. Hold off on anything that uses a customer's W&B key until a lawyer has read W&B's terms.

| Route | What it does | Risk |
| --- | --- | --- |
| Read the active run in the script | `tensorlens.init(model)` reads `wandb.run.id` once when a run is active and tags the tensor stream with it | Low. It uses the MIT-licensed client on the customer's machine and touches no W&B server |
| Wrap `wandb.log` | Keeps TensorLens steps aligned with the steps W&B records | Low, but it leans on client behaviour that can change between releases. Handle `commit=False`, explicit `step=`, and step metrics set with `define_metric` |
| Viewer reads W&B history with the customer's key, on their machine | Pulls scalars into the viewer next to the tensor views | Needs legal review. The terms restrict accessing the service to build a competing product |
| Your hosted service holds the customer's W&B key | Your cloud pulls their W&B data | Highest. The terms also restrict sharing access credentials with a third party |

Do not read `wandb.run.step` per training step. It waits on a round trip to the W&B background process, with a 30-second timeout, and raises an error in `mode="shared"` runs. It is also W&B's own log counter, not the training step: PyTorch Lightning's `WandbLogger` records the step as a `trainer/global_step` metric. Use the SDK's own global step and record the W&B step only as metadata.

The W&B Python client is MIT-licensed. The W&B server is proprietary, and its free local licence is for personal projects only, so never bundle it. If TensorLens ever ships a default metadata server, use MLflow, whose client and server are both Apache 2.0. The terms were read through a third-party summary, not the original text.

## Cross-GPU detector

This detector checks that weights meant to be identical copies on several GPUs stay identical, and fires on the first step they drift apart.

**Why it is in the plan.** In the BLOOM-176B run, gradient clipping was applied on only one worker in each tensor-parallel group. LayerNorm weights that should have matched drifted apart while the loss curve looked healthy. It was found 13 days into the run, by a test that compared the weights across ranks, and the bug predated the run (TrainCheck's write-ups say ten or fifteen days). A separate study of faulty hardware found the same pattern: parameters on an unhealthy node drift while the loss stays close to normal.

**How it works**

1. Every N steps, after `optimizer.step()`, each rank computes two fingerprints per parameter tensor: its L2 norm and its sum, both in float64.
2. The ranks that hold copies of the same weights share their fingerprints with one small `all_gather`.
3. One rank in that group computes the spread across ranks for each tensor, relative to the tensor's norm, and sends it to the ingest server.
4. The detector fires when either spread passes a tolerance. Start at 1e-6 and tune it on healthy runs.

It runs outside the compiled step and needs no hooks. The evidence it hands to the agent and the viewer is a table of rank by layer spread, with the first step at which each layer diverged.

```python
# Every N steps, after optimizer.step(), outside the compiled step.
# Run on two CPU processes with Gloo (sdk/examples/replica_check.py); not yet on two GPUs.
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

F64 = torch.float64
CHUNK = 1 << 24  # elements per float64 chunk: at most 128 MB extra memory


def _sum_f64(p):
    # p.sum(dtype=float64) on a bf16 tensor copies the whole tensor to float64 first.
    flat = p.reshape(-1)
    return torch.stack([c.sum(dtype=F64) for c in flat.split(CHUNK)]).sum()


def _norms_f64(tensors):
    # _foreach_norm's fused CUDA kernel requires each call's list to share one
    # device and dtype, and be contiguous; mixed dtypes fall back to per-tensor
    # vector_norm, which copies bf16 tensors to float64 first.
    norms = [None] * len(tensors)
    groups = {}
    for i, p in enumerate(tensors):
        if p.is_cuda and p.is_contiguous() and p.numel() > 0:
            groups.setdefault((p.device, p.dtype), []).append((i, p))
        else:
            flat = p.reshape(-1)
            norms[i] = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(c, dtype=F64) for c in flat.split(CHUNK)])
            )
    for group in groups.values():
        idxs, ts = zip(*group)
        for i, n in zip(idxs, torch._foreach_norm(ts, 2, dtype=F64)):
            norms[i] = n
    return torch.stack(norms)


@torch.no_grad()
def replica_spread(params, group):
    # params: tensors meant to be identical on every rank in `group`.
    # group: the replication group. Pass dist.group.WORLD only for plain DDP;
    # under tensor, pipeline or hybrid-FSDP parallelism WORLD compares the wrong ranks.
    if group is None:
        raise ValueError("pass the replication process group explicitly")
    local = [p.to_local() if isinstance(p, DTensor) else p for p in params]
    if not local:
        raise ValueError("no replicated parameters to check")
    norms = _norms_f64(local)
    sums = torch.stack([_sum_f64(p) for p in local])
    fp = torch.stack([norms, sums], dim=1)      # [n_params, 2]
    out = [torch.empty_like(fp) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, fp, group=group)       # 16 bytes per tensor per rank
    allfp = torch.stack(out)                    # [ranks, n_params, 2]
    spread = allfp.max(0).values - allfp.min(0).values
    scale = allfp[..., 0].mean(0).clamp_min(1e-12)   # mean norm per tensor
    return spread / scale[:, None]              # [n_params, 2]
```

The sum is there because a norm cannot see a sign flip: `[1, -1]` and `[1, 1]` have the same norm and different sums. Both fingerprints are float64, because a norm or sum in bf16 or fp16 is too coarse to show small drift.

On CUDA, `torch._foreach_norm(ts, 2, dtype=torch.float64)` reads bf16 and fp16 tensors directly in one fused kernel per `(device, dtype)` group; an ungrouped list with mixed dtypes (such as bf16 weights and fp32 LayerNorm weights) silently falls back to `vector_norm` per tensor. `vector_norm(p, dtype=torch.float64)` and `p.sum(dtype=torch.float64)` convert the whole tensor to float64 first, which briefly needs four times a bf16 tensor's memory, so sums and any non-fast-path norms are taken in chunks. `_foreach_norm` is private, as noted in the capture rules. The run time and memory cost are unmeasured.

**Where it applies**

| Setup | What is checked | Note |
| --- | --- | --- |
| DDP (data parallel) | Every parameter | Each rank holds a full copy, and the copies should stay in sync |
| Tensor parallelism | Only weights replicated across ranks, such as LayerNorm | The user or an integration has to say which ones; this is the BLOOM case |
| Hybrid FSDP (sharded inside a node, replicated across nodes) | Each rank's local shard, compared across the replication group | Pass that process group as `group`. FSDP2 and tensor-parallel parameters are `DTensor`s, so take `.to_local()` before fingerprinting, as the code does. FSDP1's flat parameters need another route. Both are untested |
| Fully sharded (FSDP full shard, ZeRO-3) | Nothing | Each rank holds a different shard, so there are no copies to compare |

**Limits**

- A swap of two values inside a tensor changes neither the norm nor the sum. A weighted sum with fixed pseudo-random weights would catch it, for one extra pass over the tensor. Add a full hash at checkpoint time if that matters.
- Buffers such as BatchNorm running statistics differ per rank by design and are left out.
- It does not catch faults that hit every rank alike, slow nodes, or crashes.
- The tolerance, the cost and the code are untested. Include a cross-GPU drift case and a sign-flip case in the seeded failures for Gate 2.

## Viewer and agent evidence

Use stock Rerun 0.43.2 for the first release. Pin the exporter SDK and Viewer together. The [MoE bake-off](experiments/moe-bakeoff/README.md) uses a blueprint and built-in views for the detector summary, status heatmap, per-expert load table, and diversity timeline. Keep the evidence schema independent of Rerun so engineers and agents can read the same numbers in JSON.

Use Rerun's headless Viewer for per-view PNGs and `rerun viewer-mcp` for screenshots and the widget tree. Give the agent the JSON evidence for the numbers. When exporting a screenshot for a requested run and step, record the actual sample step alongside the PNG and JSON. A view can show the most recent sample at or before the cursor. Verify software rendering in the target Linux environment.

If the trials show that sorting and a table that follows layer selection materially improve verification, a custom view is needed. In 0.43.2 the built-in dataframe view cannot sort. Rerun's sortable table widget is reachable only through `ViewerClient.send_table`, which opens a separate table page outside the recording and its blueprint, not beside the heatmap.

Spend up to one week prototyping those interactions in a single Rust `ViewClass`. Register it in a pinned Rerun app using the upstream `custom_view` example. Check that selection and scrubbing stay synchronized and that the same view can render headlessly.

- **Headless.** The example starts a windowed viewer with `re_viewer::run_native_app`, whose closure returns `Result<Box<dyn eframe::App>, _>`. `re_viewer::run_headless_app` takes a closure returning concrete `re_viewer::App` directly, plus an optional window size, and needs no main-thread token, so share a `create_app(cc) -> re_viewer::App` helper between the two branches as the `rerun` CLI does.
- **Command line.** The example has none of the stock `rerun` flags (`--headless`, `--window-size`, `.rrd` loading), so the prototype has to add the ones it needs.
- **Agent access.** `rerun viewer-mcp` runs as its own process and connects to a viewer's control port. The example's gRPC server serves that port, so the stock MCP server should drive the custom app as well; check it in the spike. Estimate the upgrade work before committing to it. Rune remains an option if a separate TensorLens app becomes necessary.

## Roadmap

Twelve weeks in five phases. Each phase ends in a gate, and the next phase starts only once the gate passes.

&#91;embedded content: 12-week roadmap · 5 phases, 5 gates\]

The embedded roadmap diagram predates this version; the steps below are the current roadmap. There are no customer interviews: the author is the first user, and TensorLens is validated on the author's own training runs, then through a free public release. Gates 1 and 2 restate criteria from elsewhere in this plan. The Gate 0, 3 and 4 criteria and the split into weeks are proposals. The one exception to the gate order is the SDK skeleton, which starts alongside the Gate 1 trial because the trial does not depend on it. If any gate after Gate 0 fails, spend up to one extra week, then cut scope. The gate numbers are targets, not measurements.

### Progress so far

Work ran ahead of the gate order: the SDK, detectors and exporter were built before Gate 0 and Gate 1. Details and raw numbers are in the [SDK README](sdk/README.md); the [failure list](TensorLens%20failure%20list.md) records what each detector covers. State as of commit `344a1e2`.

| Steps | State |
| --- | --- |
| 1–5, Gate 0 | Partly. The working dogfood run is IBM Granite 3.0 1B-A400M (MoE, 24 routers of 32 experts, top-8) trained on WikiText-2 in Colab, on one RTX PRO 6000 or one A100. It is a benchmark rather than one of the author's own jobs. A researched failure list of 15 public failure types exists; the author's own failures, tracker choice and multi-GPU setup are not written down, and the competitor re-check is not done (add TrainCheck to it) |
| 6–9, Gate 1 | Not started |
| 10, SDK skeleton | Done except the Lightning callback. `init(start_step=)` continues step numbers after a resume, and each step records the learning rate |
| 11, Tier 0 capture | Done except Arrow Flight; records go to JSON Lines. Activations are captured once per repeated block by default (24 decoder layers for Granite instead of 171 leaf modules; `layers="all"` keeps every leaf). Weight norms are checked in float64 every 50 steps (`weights=N`) |
| 12, Tier 1 capture | MoE load done and checked on Granite, with each router's top-k recorded. Cross-GPU fingerprints pass two CPU processes; the two-GPU NCCL run is pending |
| 13, CI gate | Partly. No graph breaks and no CUDA-graph skips on a toy MoE with Inductor and CUDA graphs. Granite runs with CUDA graphs off, because its `grouped_mm` expert path fails during CUDA-graph recording, so `reduce-overhead` is unverified on a real model. Granite shows one expected recompile per benchmark model with different hooks. Not wired into CI |
| 14, ingest server | Not started |
| 15, detectors | Done in Python. MoE imbalance (top-k aware), starved experts, weights not updating, gradient spikes (also from the clipping norm in router-only mode), activation spikes, drift and dead outputs, repeated steps, non-finite values and cross-GPU drift. On Granite: the seeded layer 12 collapse found at the right step, layer and experts on three GPU runs; 20 experts starved since pretraining; the 49 bf16 RMSNorm weights reported as not updating and no other parameter; gradient spikes at steps 141 and 645 on both GPUs |
| 16, probe pass | Not started |
| 17, Gate 2 | Partly. Router-only capture passes 1% on the RTX PRO 6000 (+0.78% and +0.02% on two runs) and fails on the A100 (+1.06% to +1.33% per round). Full capture with the new defaults (24 block hooks, gradient norms every step, weights every 50 steps) measured +4.15% against capture off on the RTX PRO 6000: about 0.36 ms routers, 0.85 ms block activations and 1.5 ms gradient norms. Blocks cut activation cost about 4.5 times; full capture still misses 1%, so it became an opt-in debugging mode and router-only the default for MoE models. Seeded drift and sign-flip cases pass only on CPU |
| 18, product exporter | Done early; the Granite recordings were checked in the stock Rerun viewer |
| 19–27 | Not started |

Findings along the way:

- **Fixed per-step thresholds failed on real data.** Healthy Granite has experts picked by 85–100% of tokens, and with two documents per batch single experts jump to over half the tokens for one step. Top-k overload therefore requires the router's diversity to drop below 0.6 of its recent median (healthy minimum 0.82; the seeded collapse 0.29). Experts nearly unused for hundreds of steps needed a windowed check, since one empty step is noise.
- **float32 norms cannot show whether bf16 weights changed.** A first stalled-weight check reported 147 of 218 parameters falsely; float64 norms fixed it but cost about 3.7 ms per step on the RTX PRO 6000. Reading the weights alone costs about 1.5 ms, so the check is sampled, as wandb.watch (1,000 batches) and TrainCheck (`sampling_interval`) do.
- **Pure-bf16 training freezes norm weights.** In the Granite benchmark's setup (bf16 weights, AdamW, lr 1e-5, no float32 master weights) all 49 RMSNorm weights never change: their bf16 step size near 1.0 is about 0.008. TensorLens now reports this.

Open gaps:

- **Full-capture overhead.** Stage 3 measured +4.15%. Gradient norms on every step are the largest part (1.5 ms); sampling them every 50 steps, as Lightning's `log_every_n_steps` and wandb.watch do, would leave about 1.2 ms (about 2%) but would also hide single-step gradient spikes. Activation hooks inside the compiled graph cannot be sampled without recompiles.
- **A100 router-only overhead.** About 2 ms per step against 0.5 ms on the RTX PRO 6000 for the same hooks; possibly host-side launch overhead. Not investigated.
- **Untuned thresholds.** The spike factor (5×), diversity drop (0.6), starved ratio (10% of an even share) and stall count (3 checks) rest on one model's data and synthetic cases.
- **Gate order.** Write the author's own failures and run on one of the author's jobs before adding more detectors, so detectors are tuned to real failures rather than seeded ones.

### Week 1: set up the dogfood runs (Gate 0)

1. Pick one or two real training jobs you run yourself, ideally an MoE model, multi-GPU, with `torch.compile`. These are the dogfood runs for every later gate.
2. Write down the failures you have hit before, or expect to hit, on those jobs: what went wrong, how long it took to notice, and what would have caught it sooner. These become the seeded failures and the answer key.
3. Record your own tracker (MLflow or W&B) and multi-GPU setup. That settles two open decisions below for v1.
4. Re-check the competitors named under Product (ARIA, Pluto, tracker MCP servers) for per-layer capture or detectors launched since Nov 13, 2026.
5. **Gate 0:** the dogfood runs are chosen, they can be instrumented, and the failure list holds at least one case that loss curves alone would not explain. If a competitor already covers that case, narrow or rethink before writing SDK code.

### Weeks 2–3: is a custom view needed? (Gate 1)

6. Generate unseen synthetic fixtures with the bake-off's `generate.py`, using different incident steps, layers and experts.
7. Pre-register timed questions that neither default screen answers, balanced between Rune's layer selection and Rerun's timeline.
8. Run three arms: stock Rerun with the bake-off blueprint, the Rune incident prototype, and the JSON evidence alone. Give the two viewer arms the same detector summary, status colors and default expert table. Counterbalance the order for engineers, and use fresh sessions with the same model and prompt for agents.
9. **Gate 1:** compare median time to a correct answer at matched accuracy, and agent tool calls. A 2× reduction from the custom interactions schedules the one-week custom view spike in weeks 8–10; otherwise stay on stock Rerun.
10. In parallel, start the SDK skeleton: `init`, run ID and step handling, the Lightning callback, and the optimizer hooks, including steps that mixed precision skips.

### Weeks 4–7: capture and detect on a real run (Gate 2)

11. Tier 0 capture: hook statistics into static buffers, the GPU ring buffer, one `.cpu()` copy every N steps, and Arrow Flight to ingest.
12. Tier 1 capture: the MoE load table and the cross-GPU fingerprints.
13. CI gate: zero graph breaks, zero recompiles, and no CUDA-graph skips under `reduce-overhead`.
14. Ingest server in Rust: Arrow in, Parquet out.
15. The first two detectors, MoE routing imbalance and cross-GPU drift, each writing the viewer-neutral JSON evidence.
16. The Tier 2 probe pass for histograms and attention; Tier 3 replay if time allows.
17. **Gate 2:** overhead under 1% on a dogfood run, a clean CI gate, and every seeded failure caught, including a cross-GPU drift case and a sign-flip case.

### Weeks 8–10: evidence for engineers and agents (Gate 3)

18. Turn the bake-off exporter into the product exporter: JSON from real runs to `.rrd` plus the blueprint.
19. Agent access: Rerun headless screenshots, `rerun viewer-mcp`, and the JSON, with the actual sample step recorded next to each PNG.
20. Verify software rendering in your Linux training environment.
21. The MLflow importer, only with three or more engineers.
22. The one-week custom view spike, only if Gate 1 called for it: headless rendering, MCP access, and an estimate of the upgrade cost.
23. **Gate 3 (proposed):** on seeded failures, an agent with TensorLens evidence finds the right cause more often than an agent with tracker data alone, and TensorLens has caught and explained at least one real problem on a dogfood run.

### Weeks 11–12: public release (Gate 4)

24. Release the SDK, detectors and exporter free under Apache 2.0 / MIT, with the MoE bake-off as a runnable demo.
25. Track outside use: installs, issues, and teams running it on their own clusters. Ask those teams whether they would pay for hosted monitoring.
26. Measure storage, rendering and model-call cost per monitored run on the dogfood runs.
27. **Gate 4 (proposed):** outside teams run TensorLens on real jobs and report problems it caught. Decide on the hosted tier and its price only after that, against the measured cost.

## Free and paid

Everything that runs on the user's own machine is free. The paid tier is the hosted monitoring.

- **Free (Apache 2.0 / MIT)** The capture SDK, detectors, and evidence exporter. Use the open-source Rerun viewer for inspection and screenshots.
- **Paid, $250–500 per seat per month (unvalidated):** hosted tensor storage, round-the-clock monitoring, alerts to Slack or PagerDuty, and approved fixes delivered as pull requests.
- **Self-hosted enterprise:** the same features inside the customer's own cloud account.

An SDK alone is hard to charge for, so the monitoring has to carry the revenue. Cost per monitored run (storage, rendering, any model calls) is not modelled yet and decides whether the price holds.

## Risks

The biggest risk is commercial, not technical: teams may value the tool and still not pay for monitoring.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Nobody pays for monitoring | No revenue | No interviews are planned, so dogfooding proves usefulness, not willingness to pay. Release free first, and ask the outside teams who adopt it whether they would pay before building the hosted tier |
| ARIA or Pluto adds layer capture and detectors | The gap closes, and the hosted tier competes with W&B or Trainy | Re-check both at Gate 0 and before the release; stay tracker- and cloud-neutral; lead with MoE, attention, cross-GPU drift and `torch.compile` support |
| Trackers add per-layer views | The wedge narrows | Stay tracker-neutral; lead with MoE, attention and torch.compile support that they lack |
| Rerun's extension interfaces change | A custom incident view may need regular updates | Pin the Viewer and SDK, test recording read-back on upgrades, and budget maintenance if the custom view ships |
| Hooks stall training or break `torch.compile` | Users switch the SDK off | Overhead gate under 1%; CI with compile on; zero graph breaks. Router-only capture, the default, meets the gate on the RTX PRO 6000 but not the A100; full capture (+4.15%) is opt-in for debugging |
| Agents misread the pictures | Wrong diagnoses, lost trust | Numbers travel as JSON; key values shown as text in the summary and table views beside each chart; score on seeded failures |
| Coding-agent clients handle tool images differently | The picture does not reach the model or the user | Test saved PNGs with the target agent clients and supply the same evidence in JSON |
| Tensor data leaves the cluster | Enterprise deals fall through | Self-hosted ingest; an option to send statistics only |
| W&B's terms restrict competing products and credential sharing | A W&B importer or hosted W&B sync breaches the customer's contract | Read only the active run inside the customer's script; legal review before anything that uses a customer's W&B key |

## Unproven and open decisions

The synthetic MoE bake-off verified recording read-back and headless rendering. Capture now runs on a real MoE model (Granite) on an RTX PRO 6000 and an A100, with router-only overhead under 1% on the first and over it on the second, a seeded collapse detected, and seven detectors checked on Granite or synthetic data. Multi-GPU capture, full-capture overhead with the new defaults, real failures on the author's own jobs, and diagnosis times remain unmeasured.

**Unproven**

- Whether sorting and layer selection improve diagnosis enough to justify a custom Rerun view.
- Whether the stock Rerun viewer stays smooth on a real 5,000-metric MLflow run.
- Whether full capture can get under 1% with block-level activations and a 50-step weight check. Router-only capture does on the RTX PRO 6000 and not on the A100; full capture was about 9% with every leaf module and per-step weight norms.
- Whether the detector thresholds hold on models other than Granite.
- Whether the hooks keep CUDA graphs on real models. They do on a toy MoE; Granite needs CUDA graphs off for reasons unrelated to TensorLens.
- The cross-GPU drift tolerance of 1e-6 on real multi-GPU runs.
- Whether a coding agent with TensorLens evidence and Rerun screenshots finds the right cause more often than one with tracker data alone.
- The data sizes in the tier table.
- Whether anyone besides the author wants this. A search on Nov 13, 2026 found no product that captures layer internals and explains failures, and no public requests for one. That does not prove none exists or that demand is absent.

**Decisions that are yours**

- [ ] Product name; "TensorLens" is a placeholder.
- [ ] Team size against 12 weeks. With fewer than three engineers, move the MLflow importer after launch.
- [ ] Who explains failures. v1 hands the evidence to the user's own coding agent. A built-in explainer, as ARIA has, is the agent loop postponed from v2 and would add scope.
- [ ] Whether W&B is the next tracker target after MLflow.
- [ ] Multi-GPU runs: which rank reports activations. Rank 0 is enough for plain data parallelism, and with pipeline or tensor parallelism every rank that owns layers reports; every rank still shares its small parameter fingerprints for the cross-GPU detector.
- [ ] Price, after the cost per monitored run is modelled.
- [x] Default capture level: decided Nov 13, 2026. Router-only is the always-on default for MoE models (+0.02% to +0.78% on the RTX PRO 6000; over 1% on the A100), and full capture (+4.15%) is an opt-in debugging mode, `capture="full"`. Rationale and sources in the SDK README's capture-mode decision.
- [ ] Whether detectors stay in Python after the Rust ingest server exists.

## Sources

- [TechInformed: OpenAI buying Neptune](https://techinformed.com/openai-buying-neptune-ai-in-all-stock-deal-what-it-means-for-ai-architecture/)
- [ZenML: Neptune shutdown and migration](https://www.zenml.io/blog/neptune-to-zenml)
- [Trainy: What Neptune got right](https://www.trainy.ai/blog/what-neptune-got-right)
- [CoreWeave blog: Introducing ARIA](https://www.coreweave.com/blog/introducing-coreweave-aria-ai-research-and-iteration-agent)
- [SiliconANGLE: CoreWeave debuts ARIA (June 29, 2026)](https://siliconangle.com/2026/06/29/coreweave-debuts-aria-agent-automate-ai-research-weights-biases/)
- [Trainy Pluto](https://www.trainy.ai/pluto)
- [Trainy Pluto MCP server](https://docs.trainy.ai/pluto/mcp)
- [W&B MCP server](https://docs.wandb.ai/platform/mcp-server)
- [MLflow MCP server](https://mlflow.org/docs/latest/genai/mcp/)
- [RunSleuth: LLM agent for PyTorch training-failure diagnosis](https://github.com/Kaiden118/RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification)
- [Mechanism-driven monitors for LLM training instability](https://arxiv.org/html/2606.28116)
- [AMD: Agentic diagnosis for LLM training at scale](https://rocm.blogs.amd.com/software-tools-optimization/maxtext-slurm-agentic-diagnosis/README.html)
- [Rerun terms: MIT / Apache 2.0 licensing](https://rerun.io/terms)
- [Rerun Rust viewer extensions](https://rerun.io/docs/howto/visualization/extend-ui)
- [Rerun headless and per-view screenshots](https://rerun.io/docs/howto/visualization/export-screenshots)
- [Rerun viewer MCP](https://rerun.io/docs/reference/viewer/mcp)
- [Rerun: How we sped up time series by 20-30x](https://rerun.io/blog/primary-query-caching)
- [MLflow plugins: tracking store entry points](https://mlflow.org/docs/latest/ml/plugins/)
- [PyTorch docs: torch.compile and nn.Module hooks](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_nn_module.html)
- [NVIDIA perf-torch-sync-free: removing host-device syncs](https://mcpservers.org/agent-skills/nvidia/perf-torch-sync-free)
- [NeMo AutoModel: MoE load balance metrics](https://docs.nvidia.com/nemo/automodel/latest/apidocs/nemo_automodel/nemo_automodel.components.moe.load_balance_metrics.html)
- [MLX issue #2590: no attention weights from fused SDPA](https://upd.dev/ml-explore/mlx/issues/2590)
- [Do MLLMs Really Understand the Charts?](https://arxiv.org/abs/2509.04457v2)

* [TrainCheck: catching silent errors in DL training (BLOOM case)](https://arxiv.org/html/2506.14813v1)
* [BLOOM training chronicles: the BF16Optimizer clipping bug](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md)
* [PyTorch docs: DDP design note](https://docs.pytorch.org/2.0/notes/ddp.html)
* [PyTorch forums: comparing DDP replicas with all\_gather](https://discuss.pytorch.org/t/loss-goes-down-on-gpu0-but-up-on-other-3-when-using-distributeddataparallel/81811)
* [Understanding Silent Data Corruption in LLM Training (summary)](https://liner.com/review/understanding-silent-data-corruption-in-llm-training)

- [FSDP hybrid sharding strategies](https://apxml.com/courses/distributed-training-pytorch-fsdp/chapter-4-multi-node-scaling-nccl-tuning/hybrid-sharding-strategies)
- [PyTorch docs: torch.compiler.set\_stance](https://pytorch.org/docs/2.6/generated/torch.compiler.set_stance.html)
- [W&B pricing: local server licence](https://wandb.ai/site/pricing)
- [wandb client repository (MIT)](https://github.com/wandb/wandb)
- [ConductAtlas: W&B terms, competitive products clause](https://conductatlas.com/platform/weights-biases/weights-biases-terms-of-service/provision/CA-P-048158/prohibition-on-building-competitive-products/)
- [ConductAtlas: W&B terms, credential sharing clause](https://conductatlas.com/platform/weights-biases/weights-biases-terms-of-service/provision/CA-P-048138/prohibition-on-unauthorized-user-access-sharing/)
