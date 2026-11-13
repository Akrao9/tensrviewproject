# TensorLens v3 — Product and Build Plan

Nov 13, 2026 · @Ak B

TensorLens lets ML engineers and their coding agents see inside a PyTorch training run, so a failing run is caught and explained before it wastes days of GPU time. "TensorLens" is a placeholder name. This version replaces the larger v2 plan.

## Product

TensorLens owns the capture SDK, detectors, and the evidence they produce. Stock Rerun displays that evidence, and the team's coding agent uses it to investigate incidents.

- **Who it is for:** teams training large models on their own clusters with MLflow or W&B. Start with RL, robotics and mixture-of-experts teams, whose important signals are not loss curves.
- **The pitch:** add one line to your PyTorch script and see inside a failing run (activations, attention, MoE routing), with evidence an agent can read.
- **Against trackers:** MLflow and W&B show scalar curves. This shows what happens inside the layers.
- **Against debugging agents:** free add-ons for coding agents read logs and loss curves. This gives those agents a picture and the numbers behind it.
- **Against CoreWeave ARIA:** ARIA lives inside W&B on CoreWeave. This works with whichever tracker, agent and cloud the team already has.

**Why now.** OpenAI bought Neptune, whose product logged thousands of per-layer metrics to debug unstable runs, and the hosted service shut down in March 2026. Teams that relied on it said the hard requirement was a tool that stays responsive with thousands of runs and heavy logging.

## How it works

Capture signals inside the training script and detect failures on the ingest server. Export the evidence as JSON for agents and as a recording and blueprint for engineers to explore in Rerun.

&#91;embedded content: How it works · 7 parts, 8 flows\]

Scalars keep going to the tracker the team already uses. Tensor summaries take their own path, so a slow tensor pipeline never blocks metric logging.

## Build and borrow

Start with stock Rerun and reuse the MoE recording and blueprint from the bake-off. The first product work is capturing those signals from a real training run and generating the detector evidence.

| Piece | Build or borrow | Implementation | Note |
| --- | --- | --- | --- |
| Capture SDK | Build | Python | One line to add. Tags the tensor stream with the active MLflow run first; W&B as a later target |
| Ingest server and detectors | Build | Rust | Receives Arrow data, stores Parquet, checks for known failure patterns |
| Evidence and Rerun export | Build | Viewer-neutral JSON, `.rrd`, Rerun blueprint | Export the incident summary, status heatmap, expert-load table, and training-step timeline. Reuse the bake-off's Python exporter as the starting point |
| Incident view | Conditional | Rust (`re_viewer`, egui) | Prototype one custom view inside Rerun if trials show that layer selection and table sorting materially improve diagnosis |
| MLflow importer | Build | Python (`mlflow`, `rerun-sdk`) | Writes runs logged without the SDK to `.rrd` with the same exporter, so they open in Rerun |
| Viewer and screenshots | Borrow | Rerun (MIT / Apache 2.0) | Stock desktop viewer, built-in views, timeline, headless screenshots, and viewer MCP |
| Run metadata | Borrow | MLflow or W&B | Params, lineage, artifacts |
| Diagnosis | Borrow | The team's own coding agent | Reads TensorLens JSON and inspects the recording through Rerun's viewer MCP |

Postponed from v2: your own VLM agent loop, the agent-written layout blueprints, live fixes to a running optimizer, and a browser client of your own.

## Capture SDK

Inside the compiled graph, hooks compute only cheap statistics of fixed size. Copies to the CPU, Arrow encoding and anything bigger happen outside the compiled step.

**Setup for the user:** `pip install` and `tensorlens.init(model)`, with the team's MLflow configuration left unchanged. `init` reads the active MLflow run (`mlflow.active_run()`) and the step, and tags the tensor stream with them, as the W&B route below does.

Do not ship a tracking-store plugin with its own URI scheme (`tensorlens+https://…`). MLflow selects the plugin by scheme, but its proxied artifact store (`mlflow-artifacts:`) accepts only `http` and `https` tracking URIs, so `log_artifact` would fail for teams that use it. The model registry would also need its own plugin.

**Rules**

- Register forward hooks **before** `torch.compile` and never change them afterwards.
- Hook bodies compute mean, std, absolute max and zero fraction per layer, and write into GPU buffers allocated once. No histograms, `.cpu()`, numpy or I/O inside a hook.
- No backward hooks. Read `p.grad` norms after `backward()` with `torch._foreach_norm`. It is a private API (`clip_grad_norm_` uses it), so test it on each supported PyTorch release and fall back to `torch.linalg.vector_norm` per tensor.
- Fixed shapes only: `histc` with fixed bins and `scatter_add_` into `zeros(n_experts)` are fine; `bincount` and `nonzero` are not.
- After each step, copy the hook buffers into row `step % N` of a GPU ring buffer, outside the graph. A step counter used inside the compiled region would force a recompile on every step.
- Every N steps, after `optimizer.step()`, move the ring buffer to the CPU with one `.cpu()` call. A worker thread builds the Arrow batch and sends it over Arrow Flight.
- Add pinned buffers and a side CUDA stream only if the overhead gate shows a stall.
- Histograms and attention come from a probe pass: every M steps, one fixed batch through the wrapped model under `no_grad` and `torch.compiler.set_stance("force_eager")` (PyTorch 2.6+). That runs the same FSDP- or TP-wrapped model without the compiler. On older PyTorch the fallback is `model._orig_mod`, which is untested with sharded parameters.
- Run the probe after the ring-buffer copy, because the hooks fire during the probe too and overwrite the step's statistics.
- With plain data parallelism, rank 0 reports the Tier 0 statistics. With pipeline or tensor parallelism, every rank that owns layers reports its own, or layers on other ranks go unseen.
- CI gate: `TORCH_LOGS="graph_breaks,recompiles"` must show zero breaks with the SDK on. With `mode="reduce-overhead"`, the log must also not contain "skipping cudagraphs due to mutated inputs". The hooks' in-place writes into preallocated buffers are input mutations, which can silently turn CUDA graphs off while breaks and recompiles stay at zero.
- Hooks changed after compiling are ignored, not recompiled: `torch._dynamo.config.skip_nnmodule_hook_guards` defaults to `True`, so the compiled graph keeps the hooks it saw first.

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
| Read the active run in the script | `tensorlens.init(model)` reads `wandb.run.id` and the step when a run is active, and tags the tensor stream with them | Low. It uses the MIT-licensed client on the customer's machine and touches no W&B server |
| Wrap `wandb.log` | Keeps TensorLens steps aligned with the steps W&B records | Low, but it leans on client behaviour that can change between releases |
| Viewer reads W&B history with the customer's key, on their machine | Pulls scalars into the viewer next to the tensor views | Needs legal review. The terms restrict accessing the service to build a competing product |
| Your hosted service holds the customer's W&B key | Your cloud pulls their W&B data | Highest. The terms also restrict sharing access credentials with a third party |

The W&B Python client is MIT-licensed. The W&B server is proprietary, and its free local licence is for personal projects only, so never bundle it. If TensorLens ever ships a default metadata server, use MLflow, whose client and server are both Apache 2.0. The terms were read through a third-party summary, not the original text.

## Cross-GPU detector

This detector checks that weights meant to be identical copies on several GPUs stay identical, and fires on the first step they drift apart.

**Why it is in the plan.** In the BLOOM-176B run, gradient clipping was applied on only one worker in each tensor-parallel group. LayerNorm weights that should have matched drifted apart while the loss curve looked healthy. It took 10 days to notice. A separate study of faulty hardware found the same pattern: parameters on an unhealthy node drift while the loss stays close to normal.

**How it works**

1. Every N steps, after `optimizer.step()`, each rank computes two fingerprints per parameter tensor: its L2 norm and its sum, both in float64.
2. The ranks that hold copies of the same weights share their fingerprints with one small `all_gather`.
3. One rank in that group computes the spread across ranks for each tensor, relative to the tensor's norm, and sends it to the ingest server.
4. The detector fires when either spread passes a tolerance. Start at 1e-6 and tune it on healthy runs.

It runs outside the compiled step and needs no hooks. The evidence it hands to the agent and the viewer is a table of rank by layer spread, with the first step at which each layer diverged.

```python
# Every N steps, after optimizer.step(), outside the compiled step. Not yet run.
import torch
import torch.distributed as dist

@torch.no_grad()
def replica_spread(params, group=None):
    # params: tensors meant to be identical on every rank in `group`
    f64 = torch.float64
    fp = torch.stack([
        torch.stack([torch.linalg.vector_norm(p, dtype=f64), p.sum(dtype=f64)])
        for p in params
    ])                                          # [n_params, 2]: norm, sum
    out = [torch.empty_like(fp) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, fp, group=group)       # 16 bytes per tensor per rank
    allfp = torch.stack(out)                    # [ranks, n_params, 2]
    spread = allfp.max(0).values - allfp.min(0).values
    scale = allfp[..., 0].mean(0).clamp_min(1e-12)   # mean norm per tensor
    return spread / scale[:, None]              # [n_params, 2]
```

The sum is there because a norm cannot see a sign flip: `[1, -1]` and `[1, 1]` have the same norm and different sums. For fp32 weights, `torch._foreach_norm(params)` is a faster way to get the norms. Keep the float64 path for bf16 and fp16 weights, where a norm in the weight's own precision is too coarse to show small drift. In at least one code path, `vector_norm(p, dtype=torch.float64)` converts the whole tensor to float64 first, so a large bf16 tensor briefly needs four times its own memory. Process large tensors in chunks if that matters; the cost is unmeasured.

**Where it applies**

| Setup | What is checked | Note |
| --- | --- | --- |
| DDP (data parallel) | Every parameter | Each rank holds a full copy, and the copies should stay in sync |
| Tensor parallelism | Only weights replicated across ranks, such as LayerNorm | The user or an integration has to say which ones; this is the BLOOM case |
| Hybrid FSDP (sharded inside a node, replicated across nodes) | Each rank's local shard, compared across the replication group | Pass that process group as `group`. How to reach the local shard depends on the FSDP version and is untested |
| Fully sharded (FSDP full shard, ZeRO-3) | Nothing | Each rank holds a different shard, so there are no copies to compare |

**Limits**

- A swap of two values inside a tensor changes neither the norm nor the sum. A weighted sum with fixed pseudo-random weights would catch it, for one extra pass over the tensor. Add a full hash at checkpoint time if that matters.
- Buffers such as BatchNorm running statistics differ per rank by design and are left out.
- It does not catch faults that hit every rank alike, slow nodes, or crashes.
- The tolerance, the cost and the code are untested. Include a cross-GPU drift case and a sign-flip case in the seeded failures for Gate 2.

## Viewer and agent evidence

Use stock Rerun 0.43.2 for the first release. Pin the exporter SDK and Viewer together. The [MoE bake-off](experiments/moe-bakeoff/README.md) uses a blueprint and built-in views for the detector summary, status heatmap, per-expert load table, and diversity timeline. Keep the evidence schema independent of Rerun so engineers and agents can read the same numbers in JSON.

Use Rerun's headless Viewer for per-view PNGs and `rerun viewer-mcp` for screenshots and the widget tree. Give the agent the JSON evidence for the numbers. When exporting a screenshot for a requested run and step, record the actual sample step alongside the PNG and JSON. A view can show the most recent sample at or before the cursor. Verify software rendering in the target Linux environment.

If the trials show that sorting and a table that follows layer selection materially improve verification, first check what Rerun's built-in table options can cover. Spend up to one week prototyping any remaining interactions in a single Rust `ViewClass`. Register it in a pinned Rerun app using the upstream `custom_view` example. Check that selection and scrubbing stay synchronized and that the same view can render headlessly. The example starts a windowed viewer with `re_viewer::run_native_app`; the headless path is `re_viewer::run_headless_app`, which the example does not use, so try that first. Estimate the upgrade work before committing to it. Rune remains an option if a separate TensorLens app becomes necessary.

## Roadmap

Twelve weeks in five phases. Each phase ends in a gate, and the next phase starts only once the gate passes.

&#91;embedded content: 12-week roadmap · 5 phases, 5 gates\]

Gate 0 uses design-partner interviews to test willingness to pay for monitoring. If it fails, rethink the product before building the SDK. The Gate 1 viewer trial has three arms: stock Rerun with the bake-off blueprint, the Rune incident prototype, and the JSON evidence alone. Give the two viewer arms the same detector summary, status colors, and default expert table. Time how long engineers and agents take to verify claims that require inspecting the evidence. Track correctness and agent tool calls. Use the results to decide whether to prototype a custom view. If a later gate fails, spend up to one extra week, then cut scope. The gate numbers are targets, not measurements.

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
| Nobody pays for monitoring | No revenue | Five design-partner interviews before more code. Ask what last cost them a week of GPU time and what would have caught it |
| Trackers add per-layer views | The wedge narrows | Stay tracker-neutral; lead with MoE, attention and torch.compile support that they lack |
| Rerun's extension interfaces change | A custom incident view may need regular updates | Pin the Viewer and SDK, test recording read-back on upgrades, and budget maintenance if the custom view ships |
| Hooks stall training or break `torch.compile` | Users switch the SDK off | Overhead gate under 1%; CI with compile on; zero graph breaks |
| Agents misread the pictures | Wrong diagnoses, lost trust | Numbers travel as JSON; key values shown as text in the summary and table views beside each chart; score on seeded failures |
| Coding-agent clients handle tool images differently | The picture does not reach the model or the user | Test saved PNGs with the target agent clients and supply the same evidence in JSON |
| Tensor data leaves the cluster | Enterprise deals fall through | Self-hosted ingest; an option to send statistics only |
| W&B's terms restrict competing products and credential sharing | A W&B importer or hosted W&B sync breaches the customer's contract | Read only the active run inside the customer's script; legal review before anything that uses a customer's W&B key |

## Unproven and open decisions

The synthetic MoE bake-off verified recording read-back and headless rendering. Real-run capture, training overhead, and diagnosis times remain unmeasured.

**Unproven**

- Whether sorting and layer selection improve diagnosis enough to justify a custom Rerun view.
- Whether the stock Rerun viewer stays smooth on a real 5,000-metric MLflow run.
- The training overhead, and whether the hook's in-place write compiles without graph breaks.
- Whether a coding agent with TensorLens evidence and Rerun screenshots finds the right cause more often than one with tracker data alone.
- The data sizes in the tier table.
- No existing tool that combines Rerun with MLflow or training debugging turned up in a search. That does not prove none exists.

**Decisions that are yours**

- [ ] Product name; "TensorLens" is a placeholder.
- [ ] Team size against 12 weeks. With fewer than three engineers, move the MLflow importer and the paid pilot after launch.
- [ ] Whether W&B is the next tracker target after MLflow.
- [ ] Multi-GPU runs: which rank reports activations. Rank 0 is enough for plain data parallelism, and with pipeline or tensor parallelism every rank that owns layers reports; every rank still shares its small parameter fingerprints for the cross-GPU detector.
- [ ] Price, after the cost per monitored run is modelled.

## Sources

- [TechInformed: OpenAI buying Neptune](https://techinformed.com/openai-buying-neptune-ai-in-all-stock-deal-what-it-means-for-ai-architecture/)
- [ZenML: Neptune shutdown and migration](https://www.zenml.io/blog/neptune-to-zenml)
- [Trainy: What Neptune got right](https://www.trainy.ai/blog/what-neptune-got-right)
- [CoreWeave blog: Introducing ARIA](https://www.coreweave.com/blog/introducing-coreweave-aria-ai-research-and-iteration-agent)
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
