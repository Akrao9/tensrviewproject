# TensorLens v3 — Product and Build Plan

Nov 13, 2026 · @Ak B

TensorLens lets ML engineers and their coding agents see inside a PyTorch training run, so a failing run is caught and explained before it wastes days of GPU time. "TensorLens" is a placeholder name. This version replaces the larger v2 plan.

## Product

TensorLens owns the capture SDK, detectors, the evidence they produce, and (decided Nov 14, 2026) its own Rust viewer app built on Rune. Until that app covers the needed views, stock Rerun displays the evidence, and the team's coding agent uses it to investigate incidents.

- **Who it is for:** teams training large models on their own clusters with MLflow or W&B. Start with RL, robotics and mixture-of-experts teams, whose important signals are not loss curves.
- **The pitch:** add one line to your PyTorch script and see inside a failing run (activations, attention, MoE routing), with evidence an agent can read.
- **Against trackers:** MLflow and W&B show scalar curves, and their MCP servers let coding agents query those runs. This shows what happens inside the layers.
- **Against CoreWeave ARIA:** ARIA is an agent inside W&B that analyses experiments, proposes next runs and launches them with approval (public preview June 2026). It reasons over what W&B logged, mostly scalars, and is tied to W&B and CoreWeave. This captures layer internals and works with whichever tracker, agent and cloud the team already has.
- **Against Trainy Pluto:** Pluto is a Neptune replacement built for responsiveness with many runs, and its experimental MCP server lets an agent read per-layer panels in one call. It displays what the user logs; no capture SDK or failure detectors turned up. This captures the internals cheaply under `torch.compile` and FSDP and detects known failures.
- **Against debugging agents:** free add-ons and research agents such as RunSleuth read logs, loss curves and gradient norms. None found inspects activations, attention or MoE routing. This gives those agents a picture and the numbers behind it.

**The gap.** A search on Nov 13, 2026 found no product that captures layer internals (activations, attention, MoE routing, cross-GPU drift) and hands an agent the evidence to explain a failure. Public requests for such a tool were not found either, so demand is unproven; the closest signal is that CoreWeave and Trainy are both building agent access to training runs.

**Why now.** OpenAI bought Neptune, whose product logged thousands of per-layer metrics to debug unstable runs, and the hosted service shut down in March 2026. Teams that relied on it said the hard requirement was a tool that stays responsive with thousands of runs and heavy logging.

## How it works

Capture signals inside the training script, write them to an append-only local file on each rank first, and optionally stream them from that file's byte offset over Arrow Flight to a self-hosted Rust ingest server that stores partitioned Parquet. Detectors run in Python on either the local files or the server's Parquet files while the run is in progress, exporting JSON evidence for coding agents and an `.rrd` recording and blueprint for engineers to explore in Rerun.

Scalars keep going to the tracker the team already uses. Tensor summaries take their own path, so a slow tensor pipeline or network stall never blocks metric logging or drops training steps.

## Build and borrow

Start with stock Rerun and reuse the MoE recording and blueprint from the bake-off. Capture those signals from a real training run, stream them to the self-hosted ingest server for multi-node storage and live detection, and generate the detector evidence.

| Piece | Build or borrow | Implementation | Note |
| --- | --- | --- | --- |
| Capture SDK | Build | Python (`sdk/src/tensorlens/`) | One line to add (`tensorlens.init(model)`). Tags the tensor stream with the MLflow or W&B run ID and writes `tensorlens-out/<run_id>.rank<rank>.jsonl` through a bounded queue (`WRITER_QUEUE = 8`) |
| Opt-in Flight sender | Build | Python (`sdk/src/tensorlens/flight.py`) | Activated by `endpoint=` or `TENSORLENS_ENDPOINT`. Tails the local `.jsonl` file from `<run_id>.rank<rank>.offset` and streams `RecordBatch`es over Arrow Flight `DoPut` |
| Ingest server | Build | Rust (`server/`) | Single self-hosted binary (`tensorlens-server`). Receives Arrow Flight `DoPut`, deduplicates by `(run_id, rank, step)`, and flushes partitioned Parquet to S3, GCS, or local disk via `object_store` |
| Detectors | Build | Python (`sdk/src/tensorlens/evidence.py`) | Eleven incident kinds across MoE routing, weights, gradients, activations, step numbers, loss scaling, non-finite values, and cross-GPU drift. Runs on local `.jsonl` or live server Parquet (`python -m tensorlens --watch`). Port to Rust only if the 64-rank benchmark exceeds 250 ms per flush |
| Evidence and Rerun export | Build | Viewer-neutral JSON (`schema_version: 1`), `.rrd`, Rerun blueprint | Exports the incident summary, status heatmap, expert-load table, probe tab, and training-step timeline (`python -m tensorlens`) |
| Viewer app | Build | Rust on Rune (wgpu, headless PNG and element-tree inspection) | TensorLens's own desktop and web app (decided Nov 14, 2026). Reads `evidence.json` and the JSONL/Parquet step data directly, not `.rrd`. Must match Rerun's step timeline, layer × expert heatmap, histograms and attention maps before it replaces Rerun. Not started |
| Viewer and screenshots, interim | Borrow, temporary | Rerun 0.43.2 (MIT / Apache 2.0) | Stopgap until the Rune app covers the views above: stock desktop viewer, built-in views, timeline, headless screenshots, and `rerun viewer-mcp`. Keep `export_rerun` working until then |
| Run metadata | Borrow | MLflow or W&B | Params, lineage, artifacts |
| Diagnosis | Borrow | The team's own coding agent | Reads TensorLens `evidence.json`; inspects the recording through Rerun's viewer MCP for now, and through the Rune app's headless PNGs and element tree once it exists |

Postponed from v2 and v1: a built-in VLM agent loop, agent-written layout blueprints, live fixes to a running optimizer, a custom browser client, Tier 3 activation replay, and the MLflow scalar-only run importer (revisit the importer only if three or more engineers join the project).

## Capture SDK

Inside the compiled graph, hooks compute only cheap statistics of fixed size. Copies to the CPU, JSONL writing, Arrow Flight encoding, and probe passes happen outside the compiled step.

**Setup:** `python -m pip install -e 'sdk[rerun]'` and `tensorlens.init(model)`, with the team's MLflow or W&B configuration left unchanged. `init` tags the tensor stream with the tracker's run ID, and the SDK records its own global step with every batch.

- **Run ID.** Takes an explicit `run_id=` first, then `MLFLOW_RUN_ID`, `mlflow.active_run()`, or `wandb.run.id`. Without any of those, it uses the launcher's job ID (`TORCHELASTIC_RUN_ID` unless `"none"`, or `SLURM_JOB_ID` and `SLURM_STEP_ID`), and otherwise generates a unique `run-<UTC YYYYMMDD-HHMMSS>-<hex6>` ID, broadcast from rank 0 when `torch.distributed` is initialized, so separate runs never append to the same file and every rank shares one name (Nov 14, 2026). Because `mlflow.active_run()` is thread-local and `None` when a run was created through `MlflowClient` (as PyTorch Lightning's `MLFlowLogger` does on rank 0 via `@rank_zero_experiment`), the Lightning callback (`tensorlens.lightning.TensorLensCallback`, Nov 14, 2026) reads the logger's run ID on rank 0 and broadcasts it with `trainer.strategy.broadcast`.
- **Step.** An MLflow run has no current step; the step is passed with each `log_metric` call. By default (`auto_step=True`), TensorLens counts training iterations through `register_optimizer_step_post_hook` (imported from `torch.optim.optimizer`), or takes `step=` from `session.step(step=...)` / `trainer.global_step` when `auto_step=False`. With multiple optimizers on one model (such as Muon plus AdamW), the first optimizer to step becomes the lead optimizer and commits the iteration's step; other optimizers in the same iteration are deduplicated by parameter gradient `(id(p.grad), p.grad._version)` signatures (`sdk/src/tensorlens/capture.py`). When `torch.amp.GradScaler` skips `optimizer.step()` on gradient overflow, neither optimizer hook fires; use `auto_step=False` and call `session.step(lr=...)` once per iteration, or attach `scaler=` (planned in Step 10b) to record skipped overflow steps.

Do not ship a tracking-store plugin with its own URI scheme (`tensorlens+https://…`). MLflow selects the plugin by scheme, but its proxied artifact store (`mlflow-artifacts:`) accepts only `http` and `https` tracking URIs, so `log_artifact` would fail for teams that use it. The model registry would also need its own plugin.

**Rules (matching `sdk/src/tensorlens/capture.py` and `probe.py` at `ffdbabd`)**

- **Static compiled hooks.** Register forward hooks **before** `torch.compile` and keep them attached after `session.close()`. With default PyTorch settings, hooks added to a module that had none when compiled are silently ignored (`skip_nnmodule_hook_guards`); on a module that already had hooks, Dynamo guards the hook dictionary, so adding or removing a hook triggers a recompile. `init()` rejects an outer `torch.compile(model)` wrapper (`_orig_mod`).
- **Fixed-shape on-device hook bodies.** Hook bodies compute mean, std, absolute max, and zero fraction per watched module (or router expert counts via compare-and-sum into an `int64` buffer) and write into preallocated on-device buffers marked with `torch._dynamo.mark_static_address`. No histograms, `.cpu()`, `bincount`, `nonzero`, numpy, or I/O run inside a compiled hook. Across microbatches (gradient accumulation), hooks keep the largest `abs_max`, sum router counts, and keep the last microbatch's `mean`, `std`, and `zero_frac`; recomputed forward passes during activation checkpointing backward are skipped.
- **Gradient norms without backward hooks.** Read `p.grad` norms in `register_optimizer_step_pre_hook` (before the lead optimizer's `step()` or `zero_grad()`, or after `step(closure)` when a closure is used) with `torch._foreach_norm(..., dtype=torch.float32)`, grouped by `(device, dtype)` after unwrapping `DTensor.to_local()`, falling back to chunked `torch.linalg.vector_norm` when `_foreach_norm` is unavailable. Because the pre-hook runs after user gradient clipping, TensorLens also wraps `torch.nn.utils.clip_grad_norm_` while the session is open (and keeps the wrapper callable after `close()`) to record the pre-clipping total norm (`clip_grad_norm`) at zero extra cost; callers can also call `session.record_grad_norm(norm)` or `session.record_grads()` before manual clipping.
- **Ring buffer and bounded writer queue.** On each step, outside the compiled graph, hook buffers copy into slot `_slot` of an on-device ring buffer of length `every` (`every=50` by default). Every `every` steps (and on `close()`), the float32 ring buffer, int64 router counts, clip-norm ring, replica-spread ring, and logged-scalar ring move to the CPU and are enqueued to a background writer thread with a bounded queue (`WRITER_QUEUE = 8` blocks of `every` steps). If the sink stalls, training blocks on `queue.put` with a one-time `RuntimeWarning` rather than growing host memory or dropping steps.
- **Tier 2 probe pass (`probe=`, `probe_every=1000`).** Requires PyTorch 2.6+ (`torch.compiler.set_stance("force_eager")`). Runs after the step's ring-buffer copy and clears the capture hook buffers afterwards so probe activations never pollute training statistics. Snapshots and restores each submodule's `training` flag, turns off deterministic algorithms for `histc` / `index_add_`, and restores CPU/CUDA RNG states. Temporarily switches Hugging Face `flash_attention_2` and `flash_attention_3` modules to `sdpa` and wraps `F.scaled_dot_product_attention` so the fused kernel still produces the forward output while the wrapper recomputes `softmax(QKᵀ)` in query blocks of at most $2^{26}$ scores (256 MB), repeating `key` heads to match `query` heads under grouped-query attention (`enable_gqa`). Each query row's entropy is normalized by the log of that row's visible key count (`entropy_share`), excluding all-masked and non-finite rows. If a probe raises an exception, `probe.error` is recorded with a `RuntimeWarning` and the step record is preserved.
- **Per-rank files.** Every rank writes its own file, `tensorlens-out/<run_id>.rank<rank>.jsonl`, reading the rank from `torch.distributed` or `RANK` / `LOCAL_RANK`.
- **Compiler gate.** Verified in-process via Dynamo counters: [test_compile.py](sdk/tests/test_compile.py) and [test_probe.py](sdk/tests/test_probe.py) run on CPU with `error_on_recompile=True` and check `counters["graph_break"]` is empty; [gpu_check.py](sdk/examples/gpu_check.py) also checks `counters["inductor"]["cudagraph_skips"]` on CUDA, with `TORCH_LOGS="graph_breaks,recompiles,cudagraphs"` enabled for diagnostic logs.

**Data tiers**

| Tier | Data | Computed how | When | Approx. size |
| --- | --- | --- | --- | --- |
| 0 | Logged scalars (`session.log`), learning rate, `clip_grad_norm`, per-parameter `grad_norms`, per-block activation `mean`, `std`, `abs_max`, `zero_frac`, and 50-step float64 `weight_norms` / `weights_unchanged` | Forward hooks into static buffers; `_foreach_norm` in optimizer pre-hook | Computed every step (weights every 50), flushed to CPU every `every=50` steps | ~26 KB JSON per step on Granite full capture |
| 1 | MoE expert load counts (`moe`), router entropy, diversity `exp(H)/N`, dead-expert fraction (`moe_stats`), and `moe_top_k` | On-device compare-and-sum per router in forward hook; entropy and diversity computed on CPU flush | Accumulated every step, flushed every `every=50` steps | ~4 KB JSON per step for 24 routers × 32 experts |
| 1 | Cross-GPU spread (`replica_spread`): float64 L2 norm and sum per replicated parameter | Per-rank float64 norm and sum after `optimizer.step()`, one `all_gather` on `replica_group` | Every `replica_every=50` steps where `(step + 1) % replica_every == 0` | 16 bytes per parameter tensor per rank on the wire |
| 2 | Activation histograms (64 bins) per repeated block; per-head attention `max_logit`, `entropy`, `entropy_share`, `masked_rows`, `invalid_rows`, and 64×64 pooled attention map | Eager probe pass (`force_eager`, `no_grad`) on a fixed batch | Every `probe_every=1000` steps | ~0.5–2 MB JSON per probe step on Granite (24 layers) |
| 3 | Full activations for flagged layers (replay) | Postponed from v1 | — | — |

Attention requires the Tier 2 probe pass because FlashAttention and fused SDPA never materialize the attention matrix during training. The probe uses a fixed batch so distributions and attention maps compare cleanly across steps.

## W&B support

Support W&B from inside the customer's own training script. Hold off on anything that uses a customer's W&B key until a lawyer has read W&B's terms.

| Route | What it does | State and risk |
| --- | --- | --- |
| Read the active run in the script | `tensorlens.init(model)` reads `wandb.run.id` once when a run is active (after `run_id=`, `MLFLOW_RUN_ID`, and `mlflow.active_run()`) and tags the tensor stream with it | **Built.** Low risk: uses the MIT-licensed client on the user's machine and touches no W&B server |
| Wrap `wandb.log` | Keeps TensorLens steps aligned with the steps W&B records | **Not built.** Low legal risk, but leans on client internals (`commit=False`, explicit `step=`, `define_metric`) |
| Viewer reads W&B history with the customer's key, on their machine | Pulls scalars into the viewer next to the tensor views | **Not built.** Needs legal review; terms restrict accessing the service to build a competing product |
| Hosted service holds the customer's W&B key | Cloud service pulls their W&B data | **Not built.** Highest risk; terms also restrict sharing access credentials with a third party |

Do not read `wandb.run.step` per training step. It waits on a round trip to the W&B background process, with a 30-second timeout, and raises an error in `mode="shared"` runs. It is also W&B's own log counter, not the training step: PyTorch Lightning's `WandbLogger` records the step as a `trainer/global_step` metric. Use the SDK's own global step.

The W&B Python client is MIT-licensed. The W&B server is proprietary, and its free local licence is for personal projects only, so never bundle it. If TensorLens ever ships a default metadata server, use MLflow, whose client and server are both Apache 2.0. The terms were read through a third-party summary, not the original text.

## Cross-GPU detector

This detector checks that weights meant to be identical copies on several GPUs stay identical, and fires on the first step they drift apart.

**Why it is in the plan.** In the BLOOM-176B run, gradient clipping was applied on only one worker in each tensor-parallel group. LayerNorm weights that should have matched drifted apart while the loss curve looked healthy. It was found 13 days into the run, by a test that compared the weights across ranks, and the bug predated the run (TrainCheck's write-ups say ten or fifteen days). A separate study of faulty hardware found the same pattern: parameters on an unhealthy node drift while the loss stays close to normal.

**How it works (`sdk/src/tensorlens/distributed.py`)**

1. Every `replica_every` steps (`50` by default), where `(step + 1) % replica_every == 0` after `optimizer.step()`, each rank computes two fingerprints per parameter tensor: its L2 norm and its sum, both in float64. Scheduling from the shared step number rather than a rank-local call counter guarantees every rank enters the collective on the exact same step.
2. The ranks that hold copies of the same weights share their `[n_params, 2]` fingerprints with one `dist.all_gather` on `replica_group` (16 bytes per parameter tensor per rank).
3. Every rank in `replica_group` computes `max(norm_spread, sum_spread) / max(mean_norm, 1e-12)` per parameter and records it in its own step record as `replica_spread`; a NaN or Inf weight on any rank makes that parameter's spread non-finite.
4. The detector (`sdk/src/tensorlens/evidence.py`) fires when `max(norm_spread, sum_spread)` normalized by the parameter's mean L2 norm across ranks exceeds `drift_tolerance` (`build_evidence(..., drift_tolerance=1e-6)` by default, one relative tolerance for both spread components) or is not finite; non-finite parameters carry `"non_finite": true` in the `replica_drift` incident.

It runs outside the compiled step and needs no hooks. See the implementation in [distributed.py](sdk/src/tensorlens/distributed.py) and the two-process test script in [replica_check.py](sdk/examples/replica_check.py), which also handle empty parameter shards, Apple MPS fallback (where float64 is unavailable), missing `_foreach_norm`, and non-finite weights.

The sum is paired with the L2 norm because a norm cannot see a sign flip: `[1, -1]` and `[1, 1]` have the same norm and different sums. Both fingerprints are float64 because a norm or sum in bf16 or fp16 is too coarse to show small drift. On CUDA, `torch._foreach_norm(ts, 2, dtype=torch.float64)` reads bf16 and fp16 tensors directly in one fused kernel per `(device, dtype)` group; sums and non-fast-path norms split flat tensors into `1 << 24`-element chunks (at most 128 MB temporary float64 memory). A float64 norm pass over Granite's 2.6 GB of bf16 weights measured ~3.7 ms per check on the RTX PRO 6000 (~0.07 ms/step amortized at `replica_every=50`); the two-GPU `all_gather` cost over NCCL is not yet measured.

**Where it applies**

| Setup | What is checked | Note |
| --- | --- | --- |
| DDP (data parallel) | Every trainable parameter | Pass `replica_group=dist.group.WORLD`; each rank holds a full copy |
| Tensor parallelism | Only weights replicated across TP ranks, such as LayerNorm | Pass `replica_params=` (Nov 14, 2026): a `(name, param) -> bool` filter, or the parameters or their names. With Megatron, `lambda name, p: not getattr(p, "tensor_model_parallel", False)` selects the replicated weights. Without it, every trainable parameter is compared, which reports false drift under tensor parallelism |
| Hybrid FSDP (sharded inside a node, replicated across nodes) | Each rank's local shard, compared across the inter-node replication group | Pass the replication process group as `replica_group=`. `distributed.py` unwraps FSDP2 `DTensor`s via `.to_local()`; FSDP1 flat parameters are not supported |
| Fully sharded (FSDP full shard, ZeRO-3) | Nothing | Each rank holds a different shard, so there are no copies to compare |

**Limits**

- A swap of two values inside a tensor changes neither the norm nor the sum. A weighted sum with fixed pseudo-random weights would catch it, for one extra pass over the tensor. Add a full hash at checkpoint time if that matters.
- Buffers such as BatchNorm running statistics differ per rank by design and are left out.
- It does not catch faults that hit every rank alike, slow nodes, or crashes.
- Verified on two CPU processes with Gloo ([replica_check.py](sdk/examples/replica_check.py)); the two-GPU NCCL run and the `1e-6` tolerance on healthy multi-GPU training are pending Gate 2.

## Ingest server and local-first streaming

Multi-node runs need a single store that collects every rank's stream and runs detectors before the job ends. The ingest pipeline combines three patterns from the reference codebases:

- **Local log first and offset-based flow control (from W&B):** W&B writes every record to its local `.wandb` transaction log first and keeps at most 32 records in memory (`InMemorySize: 32`), collapsing any network backlog into a 24-byte file-offset range (`FlowControlBufferSavedChunk` in `wandb/core/internal/stream/flowcontrolbuffer.go`) and seeking back to `lastReadOffset` when a trailing record is incomplete (`ResetLastRead` in `wandb/core/internal/runsync/runreader.go`). By contrast, MLflow's `AsyncLoggingQueue` buffers in an unbounded Python `Queue()` with no disk log (`mlflow/utils/async_logging/async_logging_queue.py`), and Rerun's `re_grpc_client` drops queued messages if its 5-second flush times out (`re_grpc_client/src/write.rs`).
- **Single-binary self-hosted Rust server with bounded byte quota (from Rerun):** Rerun ships an open-source Rust server (`re_server` on port `51234`, `re_grpc_server` on port `9876`) using `tonic`, `arrow`, `parquet`, `object_store`, `tcp_nodelay(true)`, `http2_adaptive_window(true)`, a Bearer token `Authenticator` interceptor (`re_auth`), and a byte-bounded input channel (`CHANNEL_SIZE_BYTES = 128 MiB` in `re_quota_channel`).
- **Composite-key deduplication on retry (from W&B and MLflow):** W&B deduplicates `file_stream` retries by virtual line offset (`HistoryLineNum` in `filestreamstate.go`), and MLflow deduplicates `LogBatch` retries by upserting on `(run_uuid, key, step, timestamp)` (`sqlalchemy_store_upsert_utils.py`).

### Specification

1. **Local append-only file (`sdk/src/tensorlens/sink.py`):**
   - The SDK always writes `tensorlens-out/<run_id>.rank<rank>.jsonl` first via the bounded 8-block `JsonlSink` queue (`WRITER_QUEUE = 8`). Each JSON line carries `"schema_version": 1`, `"run_id"`, `"rank"`, `"step"`, `"ts"` (Unix seconds when the step ended) and the captured metrics.
   - Local files use JSON Lines rather than Parquet because Parquet writes its footer only on close; a killed or preempted training process would leave an unreadable Parquet file.
2. **Opt-in Flight sender (`sdk/src/tensorlens/flight.py`):**
   - Enabled only when `endpoint=` or `TENSORLENS_ENDPOINT` (`grpc://host:port` or `grpcs://host:port`) is set; default local runs open zero sockets.
   - A background daemon thread tails `tensorlens-out/<run_id>.rank<rank>.jsonl` from the byte offset saved in `tensorlens-out/<run_id>.rank<rank>.offset` (updated via atomic write-and-rename of `.offset.tmp` after each server `PutResult` ack).
   - **Torn-line recovery:** only complete lines ending in `\n` that parse as JSON and match `run_id` are batched. If EOF is reached mid-line (or a crashed prior process left a truncated line before the next newline), the sender does not advance past the incomplete tail until `\n` arrives.
   - **Batching and backoff:** sends a `pyarrow.RecordBatch` over Arrow Flight `DoPut` every **50 steps or 2 seconds** (whichever comes first, capped at **4 MiB** uncompressed). Holds at most one batch in RAM; on disconnect or `Status::resource_exhausted`, sleeps with full-jitter exponential backoff (`100 ms` base, `30 s` cap) and re-reads from the saved `.offset` on reconnect. On `session.close()`, flushes remaining complete lines with a 5-second timeout and prints the local `.jsonl` path if unacked bytes remain.
3. **Arrow Flight `DoPut` contract and schema (`server/src/ingest.rs`):**
   - Crate versions: `arrow-flight = "60.1.0"`, `arrow = "60.1.0"`, `parquet = "60.1.0"`, `object_store = "0.14.2"`, `tonic = "0.14"`, `tokio = "1"`.
   - `FlightDescriptor`: path `["runs", "<run_id>", "rank=<rank>"]`.
   - `FlightData.app_metadata`: UTF-8 JSON `{"schema_version": 1, "run_id": "<id>", "rank": 0, "min_step": 0, "max_step": 49, "end_byte_offset": 128450}`.
   - **Arrow schema (one row per step):**
     - Scalar/vector columns: `schema_version: Int32`, `run_id: Utf8`, `rank: Int32`, `step: Int64`, `ts: Float64`, `lr: List<Float64>`, `grad_norm: Float64` (nullable), `clip_grad_norm: Float64` (nullable), `weight_norm: Float64` (nullable), `weights_unchanged: List<Utf8>` (nullable).
     - Map columns: `scalars: Map<Utf8, Float64>`, `grad_norms: Map<Utf8, Float64>`, `weight_norms: Map<Utf8, Float64>`, `moe_top_k: Map<Utf8, Int32>`, `moe_counts: Map<Utf8, List<Int64>>`, `replica_spread: Map<Utf8, List<Float64>>`.
     - Canonical JSON-string columns for nested objects whose keys vary by model: `layers_json: Utf8` (nullable), `moe_stats_json: Utf8` (nullable), `probe_json: Utf8` (nullable). Storing these three nested structures as UTF-8 JSON strings inside Parquet preserves 100% lossless round-tripping back to the dict format `evidence.py` expects (`IM2`).
4. **Deduplication, backpressure, auth, and Parquet layout (`server/src/storage.rs`):**
   - **Watermark deduplication:** the server tracks `last_acked_step: HashMap<(String, i32), i64>`, initialized at startup by reading the `max(step)` row-group statistics from existing Parquet footers under `{store_root}/runs/{run_id}/rank={rank}/`. On each `DoPut` batch, rows with `step <= last_acked_step[(run_id, rank)]` are filtered out before buffering, making client retries idempotent.
   - **Memory cap:** total buffered Arrow bytes across all open streams are bounded at **128 MiB**. When full, `DoPut` returns `tonic::Status::resource_exhausted("ingest buffer full")` so senders pause on their local `.jsonl` file.
   - **Parquet flushing:** because S3 and GCS do not support in-place append, each `(run_id, rank)` buffer flushes an immutable Zstd-compressed Parquet part file every **500 steps or 10 seconds** (and on stream EOF or `SIGTERM`) via `object_store::ObjectStore::put` to:
     `{store_root}/runs/{run_id}/rank={rank}/part-{min_step:08}-{max_step:08}.parquet`
     and replies with `PutResult{app_metadata: end_byte_offset}` after the flush (or bounded buffer commit) succeeds.
   - **Authentication and TLS:** `tensorlens-server --listen 127.0.0.1:50051 --store file:///var/lib/tensorlens` binds to loopback by default. Binding to a non-loopback address requires either `TENSORLENS_TOKEN` (checked by a `tonic` interceptor against `Authorization: Bearer <token>` in constant time) or `--allow-unauthenticated`, with optional TLS via `--tls-cert` and `--tls-key`.
5. **Live detection and the Python-vs-Rust detector benchmark:**
   - For v1, `python -m tensorlens --watch {store_root}/runs/{run_id}` polls newly written `part-*.parquet` (or local `.jsonl`) files every 2 seconds, reconstructs step dicts losslessly from the Parquet columns, and writes `{run_id}.evidence.json` (`schema_version: 1`) atomically via rename.
   - **Benchmark protocol before porting detectors to Rust:** run a 64-rank, 1,000-step synthetic MoE stream (50-step flushes) and measure (a) Python `evidence.py` reading new Parquet parts via `pyarrow.parquet` vs. (b) an in-server Rust detector pass over incoming `RecordBatch`es. Keep detectors in Python unless the Python worker exceeds **250 ms** per 50-step flush across 64 ranks or **1 CPU core** steady-state.

## Viewer and agent evidence

Use stock Rerun 0.43.2 as the interim viewer, until TensorLens's own Rune app covers the views below (decided Nov 14, 2026: Rerun is temporary) (`rerun-sdk==0.43.2` pinned in [sdk/pyproject.toml](sdk/pyproject.toml)). The [MoE bake-off](experiments/moe-bakeoff/README.md) and shipped exporter (`export_rerun` in [evidence.py](sdk/src/tensorlens/evidence.py)) use a blueprint and built-in views for the detector summary, status heatmap, per-expert load table, probe tab, logged scalars tab, and diversity timeline. Keep the evidence schema independent of Rerun so engineers and agents read the exact same numbers in JSON.

- **Schemas and versioning (`sdk/src/tensorlens/evidence.py`, `capture.py`).** Every JSONL step record carries `"schema_version": 1` and `"ts"` (Nov 14, 2026); readers treat records without `schema_version` as version `0` (missing `moe_top_k` already defaults from `--top-k`). The exported `<input>.evidence.json` carries `"schema_version": 1`, `"run_id"`, `"layer_count"`, `"expert_count"`, `"router_names"`, `"overload_threshold"`, `"top_k"`, `"starve_window"`, `"starve_ratio"`, `"incidents"` (each with `kind` and `step`, kind-specific fields, and `scalars` when values were logged with `session.log`), and `"samples"` (one per router per step).
- **Headless screenshots and sample-step sidecar.** Use Rerun's headless Viewer for per-view PNGs and `rerun viewer-mcp` for live screenshots and the widget tree. Because a timeline view displays the most recent sample at or before the cursor, exporting a screenshot for `<run_id>` at requested `<req_step>` writes `<out_dir>/<run_id>.step<req_step>.<view>.png` alongside `<out_dir>/<run_id>.step<req_step>.<view>.meta.json` containing `{"run_id": "<id>", "requested_step": <req_step>, "sample_step": <actual_step>, "view": "<view>", "rrd_path": "<path>"}`.

**Own viewer app on Rune (decided Nov 14, 2026).** Rerun is temporary. TensorLens will ship its own Rust app built on Rune instead of a custom `ViewClass` inside Rerun. Rerun 0.43.2's built-in dataframe view cannot sort, and `ViewerClient.send_table` opens a separate table page outside the recording blueprint; a custom Rerun view would also tie the product to Rerun's extension interfaces.

- **Data.** The app reads `evidence.json` and the per-rank JSONL (and the ingest server's Parquet parts later) directly. `.rrd` stays an export for the interim Rerun viewer only, so capture and detectors do not change.
- **Parity before replacing Rerun.** Step timeline with scrubbing, layer × expert status heatmap, sortable expert-load table that follows layer selection, time series, probe histograms and attention maps, and loading a full Granite run (650 steps, about 17 MB of JSON) smoothly.
- **Agent access.** Rune apps render headless to PNG and expose their element tree, which replaces Rerun's headless viewer and `rerun viewer-mcp` for Step 19; keep the `requested_step` / `sample_step` sidecar.
- **Headless on Linux.** Step 20's software-Vulkan check (`lavapipe`) applies to the Rune app; Rune renders through wgpu.

## Roadmap

Twelve weeks in five phases starting Nov 13, 2026. There are no customer interviews: the author is the first user, and TensorLens is validated on the author's own training runs first, then through a free public release. Gate criteria are thresholds set before measuring and changed only at a gate review. If a gate fails after one extra week of work, follow the gate's cut order.

### Progress so far

Work ran ahead of the phase order: the SDK, eleven incident detector kinds, and the Rerun exporter were built before Gate 0 and Gate 1. Details and raw numbers are in the [SDK README](sdk/README.md); the [failure list](TensorLens%20failure%20list.md) records what each detector covers. State as of commit `ffdbabd` (Nov 14, 2026). To add TensorLens to a job, follow the [run-on-your-job guide](sdk/RUN_ON_YOUR_JOB.md).

Status values are fixed to **Done**, **In progress**, **Not started**, or **Postponed**, with the remaining done criterion stated for each unfinished row.

| Steps | Status | Unmet done criterion / notes |
| --- | --- | --- |
| 1–5, Gate 0 | **In progress** | Benchmark runs exist on IBM Granite 3.0 1B-A400M (`sdk/examples/moe_dogfood.py`) and SmolLM2-360M (`sdk/examples/dense_outlier_check.py`), plus 15 public failure types in [TensorLens failure list.md](TensorLens%20failure%20list.md); unmet: fill in the author's own failures table, record tracker/multi-GPU choice, and complete the competitor re-check including TrainCheck |
| 6–9, Gate 1 | **Not started** | Unmet: update `experiments/moe-bakeoff/generate.py` to export through `tensorlens.build_evidence` and `export_rerun`, then run the timed agent/engineer comparison |
| 10, SDK skeleton | **Done** | `init`, run ID, multi-optimizer iteration counting (`88a20e7`), `start_step=`, `session.log`, and `clip_grad_norm_` wrapping. Done Nov 14, 2026: the `TensorLensCallback` Lightning class (`sdk/src/tensorlens/lightning.py`, tested on one and two `ddp_spawn` CPU processes), unique default run ID, `schema_version` and `ts` on records, and `GradScaler` skipped-step and loss-scale recording (`scaler=`) with a `loss_scale_trouble` detector |
| 11, Tier 0 capture | **In progress** | Static buffers, GPU ring buffer (`every=50`), bounded writer queue (`WRITER_QUEUE = 8` in `88a20e7`), block activation summaries, per-step gradient norms, and 50-step float64 weight checks are done; unmet: build the opt-in Arrow Flight sender (`sdk/src/tensorlens/flight.py`) |
| 12, Tier 1 capture | **In progress** | MoE load (`moe`, `moe_stats`, `moe_top_k`) and 50-step cross-GPU fingerprints scheduled from `(step + 1) % replica_every == 0` (`88a20e7`) pass on two CPU processes; `replica_params=` for tensor parallelism done Nov 14, 2026; unmet: pass `replica_check.py` on two CUDA GPUs over NCCL |
| 13, CI gate | **In progress** | `gpu_check.py`, `test_compile.py`, and `test_probe.py` verify zero graph breaks, zero recompiles, and zero CUDA-graph skips on a toy MoE; `.github/workflows/ci.yml` added Nov 14, 2026: `pytest` and the CPU fallback of `gpu_check.py` on Python 3.11/3.12 × `torch` 2.6, 2.10, 2.14, plus one job with the Lightning and Rerun extras; unmet: the `gpu` job runs only by manual dispatch on a self-hosted CUDA runner, and no such runner is registered yet |
| 14, ingest server | **Not started** | Unmet: implement `server/` (`tensorlens-server`) with Arrow Flight `DoPut`, `(run_id, rank, step)` deduplication, 128 MiB memory quota, and Parquet part flushing via `object_store` |
| 15, detectors | **Done** | Eleven incident kinds ship in `sdk/src/tensorlens/evidence.py`: `moe_imbalance` (top-k aware), `starved_experts`, `stalled_weights`, `grad_spike`, `activation_spike`, `activation_drift`, `dead_activations`, `repeated_steps`, `loss_scale_trouble` (Nov 14, 2026; synthetic and CPU `GradScaler` only), `non_finite`, and `replica_drift`. Verified on Granite single-GPU runs (RTX PRO 6000 and A100) and synthetic fixtures |
| 16, probe pass | **In progress** | Tier 2 (`sdk/src/tensorlens/probe.py`, `9a5a39f` and `9c7d8eb`) is built and tested on CPU under `torch.compile`; unmet: measure GPU overhead and record size on Granite and SmolLM2 (`dense_outlier_check.py`). Tier 3 replay is **Postponed** |
| 17, Gate 2 | **In progress** | `capture="routers"` passes ≤1% on the RTX PRO 6000 (+0.78%, +0.02%, +0.54%) and misses on the A100 (+1.06% to +1.33%); unmet: profile A100 with `profile_overhead.py`, pass the two-GPU NCCL `replica_check.py`, and pass every case in the seeded failure suite |
| 18, product exporter | **Done** | `export_rerun` in [evidence.py](sdk/src/tensorlens/evidence.py) and `python -m tensorlens` export `.rrd` and blueprint; verified in the stock Rerun 0.43.2 viewer |
| 19–27, Gates 3–4 | **Not started** | Unmet: headless screenshot sidecars, Linux `lavapipe` verification, agent diagnosis benchmark (Gate 3), and package rename + open-source release (Gate 4) |

Findings along the way:

- **Fixed per-step thresholds failed on real data.** Healthy Granite has experts picked by 85–100% of tokens, and with two documents per batch single experts jump to over half the tokens for one step. Top-k overload therefore requires the router's diversity to drop below 0.6 of its recent median (healthy minimum 0.82; the seeded collapse 0.29). Experts nearly unused for hundreds of steps needed a 50-step windowed check, since one empty step is noise.
- **float32 norms cannot show whether bf16 weights changed.** A first stalled-weight check reported 147 of 218 parameters falsely; float64 norms fixed it but cost ~3.7 ms per check on the RTX PRO 6000. Reading the weights alone costs ~1.5 ms, so the check runs every 50 steps (`weights=50`), as `wandb.watch` (1,000 batches) and TrainCheck (`sampling_interval`) do.
- **Pure-bf16 training freezes norm weights.** In the Granite benchmark's setup (bf16 weights, AdamW, lr 1e-5, no float32 master weights) all 49 RMSNorm weights never change: their bf16 step size near 1.0 is ~0.008. TensorLens reports this.

Open gaps:

- **Full-capture overhead.** Block-level full capture (`Stage 3` in [sdk/README.md](sdk/README.md#rtx-pro-6000-run-with-block-level-capture-stage-3)) measured +4.15% on the RTX PRO 6000 (~0.36 ms routers, ~0.85 ms block activations, ~1.5 ms per-step gradient norms). *Blocks:* None (`capture="routers"` is the MoE default; `capture="full"` is opt-in for debugging). *Closed when:* optional `grads=N` step sampling is benchmarked against single-step spike recall.
- **A100 router-only overhead.** ~1.98 ms/step (+1.06% to +1.33%) against ~0.36–0.51 ms/step on the RTX PRO 6000. *Blocks:* Gate 2 on A100. *Closed when:* [profile_overhead.py](sdk/examples/profile_overhead.py) runs on an A100 and either brings `capture="routers"` ≤1.0% or documents the hardware-specific host-sync ceiling.
- **Headless rendering on Linux.** Headless screenshots have only been run on macOS. Rerun's native viewer on headless Linux requires Vulkan (`mesa-vulkan-drivers` `lavapipe` + `libvulkan1`). *Blocks:* Step 20 and Gate 3. *Closed when:* two consecutive headless exports in a Linux container produce byte-identical PNGs.
- **Many ranks on a shared filesystem.** `WRITER_QUEUE = 8` (`88a20e7`) bounds memory by pausing training with a `RuntimeWarning` when disk writes stall. *Blocks:* Gate 2 multi-GPU run. *Closed when:* tested with ≥2 ranks writing local `.jsonl` and streaming to `tensorlens-server`.
- **Dense models and probe GPU cost.** [dense_outlier_check.py](sdk/examples/dense_outlier_check.py) (SmolLM2-360M with a 100× seeded outlier at step 250) has only run on CPU. *Blocks:* Step 16 and Gate 2. *Closed when:* run on a CUDA GPU with `PASS` at step 250 and measured probe ms/step.
- **Untuned thresholds.** The spike factor (5×), diversity drop (0.6), starved ratio (10% of an even share), and stall count (3 checks) rest on Granite and synthetic fixtures. *Blocks:* Gate 3. *Closed when:* evaluated on the author's own dogfood job without false alarms on healthy segments.

### Ordered next actions

1. **Complete Gate 0 (Steps 1–5):** fill in the author's own failures table in [TensorLens failure list.md](TensorLens%20failure%20list.md#your-own-failures), record the dogfood job's tracker and multi-GPU setup, and check ARIA, Pluto, tracker MCP servers, and TrainCheck.
2. **Run pending single- and two-GPU checks (Steps 12, 16, 17):** run [replica_check.py](sdk/examples/replica_check.py) on 2 CUDA GPUs with NCCL, [profile_overhead.py](sdk/examples/profile_overhead.py) on A100, and [dense_outlier_check.py](sdk/examples/dense_outlier_check.py) on a CUDA GPU.
3. **Finish SDK remaining items (Steps 10, 12, 13, 18):** all done except registering a self-hosted CUDA runner for the CI `gpu` job (Steps 10a, 10b, 12's `replica_params=` and the CPU CI matrix are done).
4. **Build the Flight sender and Rust ingest server (Steps 11, 14):** implement `sdk/src/tensorlens/flight.py` and `server/` (`tensorlens-server`), pass the stall/resume, duplicate-batch, and torn-line tests, and run the 64-rank Python-vs-Rust detector benchmark.
5. **Run Gate 1 (Steps 6–9) and Gate 3 (Steps 19–23):** unify `experiments/moe-bakeoff/generate.py` with `export_rerun` in `sdk/src/tensorlens/evidence.py`, verify Linux `lavapipe` headless screenshots, and score coding-agent diagnosis accuracy with and without TensorLens evidence.

### File boundaries and test plan for remaining work

| Step | Path | Action | One-line purpose | Required tests and assertions |
| --- | --- | --- | --- | --- |
| 6 | `experiments/moe-bakeoff/generate.py` | Modify | Replace duplicated `routing_colors` and 0.35 rule with calls to `tensorlens.build_evidence` and `export_rerun` | `sdk/tests/test_cli.py`: generated fixture `.rrd` and `.evidence.json` match `python -m tensorlens` output byte-for-byte on `incidents` and `samples` status counts |
| 10a | `sdk/src/tensorlens/capture.py` | Done Nov 14, 2026 | `scaler=` (`torch.amp.GradScaler`) records `step_skipped` and `grad_scale` and ends the step at `scaler.update()`; `schema_version`, `ts` and the unique run ID are also done | `sdk/tests/test_capture.py`: (1) every record has `schema_version == 1` and float `ts`; (2) two `init()` calls without `run_id` write distinct files; (3) overflowing fp16 step under `GradScaler` increments step and writes `step_skipped is True` and finite `grad_scale` |
| 10b | `sdk/src/tensorlens/lightning.py` | Done Nov 14, 2026 | `TensorLensCallback(L.Callback)` calling `tensorlens.init` in `on_fit_start`, broadcasting rank 0's `logger.run_id` via `strategy.broadcast`, calling `session.log` in `on_train_batch_end`, and closing in `on_fit_end` | `sdk/tests/test_lightning.py`: (1) 2-rank CPU `ddp_spawn` run broadcasts rank 0 `run_id` so both ranks write `<run_id>.rank{0,1}.jsonl`; (2) `trainer.global_step` matches recorded steps; (3) `gradient_clip_val=1.0` records pre-clip `clip_grad_norm` |
| 11 | `sdk/src/tensorlens/flight.py` | Create | Background tailer reading `<run_id>.rank<rank>.jsonl` from `<run_id>.rank<rank>.offset` and streaming `RecordBatch`es via Arrow Flight `DoPut` when `endpoint=` / `TENSORLENS_ENDPOINT` is set | `sdk/tests/test_flight.py`: (1) partial trailing JSONL line without `\n` is not sent until completed; (2) stopping the server for 200 steps holds sender RAM growth <8 MiB and delivers all 200 steps in order upon restart; (3) `.offset` file matches file size after `close()` |
| 12 | `sdk/src/tensorlens/capture.py`, `distributed.py` | Done Nov 14, 2026 | Add `replica_params=` (optional iterable or name predicate) to `tensorlens.init` for tensor-parallel replicated weights | `sdk/tests/test_distributed.py`: passing `replica_params=lambda n, p: "norm" in n` fingerprints only matching parameters and ignores sharded linear weights |
| 13 | `.github/workflows/ci.yml` | Create | GitHub Actions workflow running `pytest sdk/tests` across Python 3.11/3.12 and PyTorch 2.6, 2.10, 2.14 on `ubuntu-latest`, plus `gpu_check.py` on a CUDA runner | Workflow exits 0 on zero test failures, `graph_break == 0`, `error_on_recompile`, and `cudagraph_skips == 0` |
| 14a | `server/Cargo.toml`, `server/src/main.rs` | Create | CLI entry point for `tensorlens-server` (`--listen`, `--store`, `--tls-cert`, `--tls-key`, `--allow-unauthenticated`, `TENSORLENS_TOKEN`) | `cargo test --manifest-path server/Cargo.toml`: non-loopback bind without token or `--allow-unauthenticated` exits with error; wrong Bearer token returns `Status::unauthenticated` |
| 14b | `server/src/ingest.rs`, `server/src/storage.rs` | Create | Arrow Flight `DoPut` service, `(run_id, rank, step)` watermark deduplication, 128 MiB memory cap, and Zstd Parquet part writer via `object_store` | `server/tests/ingest_test.rs`: (1) resending steps 0–49 produces zero duplicate rows in written Parquet; (2) exceeding 128 MiB returns `RESOURCE_EXHAUSTED`; (3) `SIGTERM` flushes buffered rows to `part-*.parquet` before exit |
| 15b | `sdk/src/tensorlens/__main__.py`, `evidence.py` | Modify | Support reading a directory of `part-*.parquet` files and `--watch` live polling mode | `sdk/tests/test_evidence.py`: `build_evidence` on Parquet parts produces identical `evidence.json` to the source `.jsonl` file (including `probe`, `scalars`, and `moe_stats`) |
| 19 | `sdk/src/tensorlens/screenshots.py` | Create | Export headless view PNGs from `.rrd` and write `<run_id>.step<req_step>.<view>.meta.json` with `requested_step` and `sample_step` | `sdk/tests/test_screenshots.py`: requesting step 455 when samples exist at 450 and 500 writes `requested_step: 455, sample_step: 450` alongside a valid PNG header |
| 22 | `viewer/Cargo.toml`, `viewer/src/main.rs` | Create | TensorLens's own Rune app reading `evidence.json` and JSONL step data, starting with the incident summary, layer × expert heatmap and sortable expert table | `cargo test --manifest-path viewer/Cargo.toml`: on a Granite capture, headless render produces a non-empty PNG and the element tree lists the first incident's step and layer |
| 24 | `sdk/pyproject.toml`, `LICENSE-APACHE`, `LICENSE-MIT` | Modify / Create | Rename package from placeholder `tensorlens` to an available PyPI name, set dual `Apache-2.0 OR MIT` license, and build wheel/sdist with zero telemetry | `twine check dist/*` passes; `pip install dist/*.whl` in a clean venv imports and runs `toy_moe.py` with zero outbound network connections |

### Week 1: set up the dogfood runs (Gate 0)

1. Pick one or two real training jobs run by the author, ideally an MoE model on multiple GPUs with `torch.compile`. These are the dogfood runs for every later gate.
2. Fill in the author's own failures table in [TensorLens failure list.md](TensorLens%20failure%20list.md#your-own-failures): what went wrong, how long it took to notice, and what signal would have caught it sooner.
3. Record the dogfood job's tracker (MLflow or W&B) and multi-GPU parallelism setup (DDP, FSDP2, TP, or Megatron).
4. Re-check the competitors named under Product (CoreWeave ARIA, Trainy Pluto, W&B MCP, MLflow MCP, RunSleuth, and TrainCheck) for per-layer capture or detectors launched since Nov 13, 2026.
5. **Gate 0 checklist (pass all four to proceed):**
   - [ ] At least one real training script and command are recorded in the plan or `sdk/RUN_ON_YOUR_JOB.md` and run for ≥100 steps with `tensorlens.init(model)` producing a valid `.jsonl` file.
   - [ ] The author's own failures table in `TensorLens failure list.md` contains ≥1 failure that scalar loss/grad-norm curves alone did not explain.
   - [ ] The tracker and multi-GPU parallelism mode for the dogfood job are recorded under Decisions below.
   - [ ] Competitor check table updated with Nov 2026 status for ARIA, Pluto, tracker MCPs, and TrainCheck.
   - *Cut order if blocked:* if no multi-GPU MoE job is available immediately, run Gate 2 on the 2-GPU Granite benchmark plus the author's single-GPU dense/MoE job.

### Weeks 2–3: what must the Rune app do first? (Gate 1)

6. Generate 6 unseen synthetic MoE and dense fixtures using `experiments/moe-bakeoff/generate.py` wired through `tensorlens.build_evidence` and `export_rerun`, varying incident steps, layers, and experts.
7. Pre-register 6 timed diagnosis questions (each requiring identifying the first failing step, layer, and failure kind/expert IDs) that neither default screen answers without interaction.
8. Run three arms across all 6 fixtures using fresh coding-agent sessions (same model and system prompt, 3 trials per fixture per arm): (A) stock Rerun 0.43.2 with the bake-off blueprint + `evidence.json`, (B) the Rune app prototype + `evidence.json`, and (C) `evidence.json` alone. Also time the author across counterbalanced fixtures on (A) vs. (B).
9. **Gate 1 checklist:**
   - [ ] Accuracy across the 6 fixtures is within ±5 percentage points between arms (or higher in Arm B).
   - [ ] Arm B cuts median agent tool calls or median human time-to-correct-answer by **≥2.0×** relative to Arm A.
   - *Decision rule (changed Nov 14, 2026):* the Rune app is built either way (Step 22). The trial decides its order of work: the interactions that cut time or tool calls most come first, and the release ships with Rerun as the viewer until the Rune app reaches parity.
10. Steps 10a and 10b are done.

### Weeks 4–7: capture, stream, and detect on a real run (Gate 2)

11. Complete Tier 0 capture and the opt-in Arrow Flight sender (`sdk/src/tensorlens/flight.py`).
12. Complete Tier 1 capture: run `sdk/examples/replica_check.py` on two CUDA GPUs over NCCL.
13. Wire the compiler gate (`gpu_check.py`, `test_compile.py`, `test_probe.py`) into `.github/workflows/ci.yml`. Note: in `moe_dogfood.py`, compiling two separate benchmark models with different hook sets in one process causes one expected Dynamo recompile when switching models; a single model with TensorLens attached must have **zero** recompiles after warmup.
14. Build the Rust ingest server (`server/`, binary `tensorlens-server`) per the [Ingest server specification](#ingest-server-and-local-first-streaming) above, and run the 64-rank Python-vs-Rust detector benchmark.
15. Verify all eleven incident detector kinds in `sdk/src/tensorlens/evidence.py` on both `.jsonl` and server-written `part-*.parquet` inputs.
16. Run the Tier 2 probe pass on a CUDA GPU via `sdk/examples/dense_outlier_check.py` and `sdk/examples/moe_dogfood.py --probe-every 1000`, recording per-probe ms and JSON/Parquet byte sizes.
17. **Gate 2 checklist (pass all five to proceed):**
    - [ ] **Overhead:** `capture="routers"` adds **≤1.0%** median ms/step against capture off across 3 paired 200-step rounds (50 warmup steps) on the dogfood GPU (verified on RTX PRO 6000 at +0.02% to +0.78%; A100 profiled and either ≤1.0% or documented host-sync bound).
    - [ ] **Compiler gate:** `gpu_check.py` and `test_compile.py` report `0` graph breaks, `0` recompiles after warmup, and `0` CUDA-graph skips under `mode="reduce-overhead"`.
    - [ ] **Seeded failures:** 100% of seeded cases caught within **±1 flush/check interval** on the seeded layer/parameter: Granite layer-12 routing collapse (`moe_dogfood.py --collapse-layer 12`), SmolLM2 block-8 100× outlier (`dense_outlier_check.py`), stalled weights, gradient spike, and 2-GPU NCCL weight drift + sign flip (`replica_check.py`).
    - [ ] **Ingest reliability:** stopping `tensorlens-server` mid-run for 200 steps and restarting loses `0` steps, produces `0` duplicate steps in Parquet, and keeps sender memory growth `<8 MiB`.
    - *Cut order if a criterion fails after one extra week:* if `capture="full"` or `probe=` exceeds budget, keep both opt-in; if the Python detector worker handles 64 ranks in ≤250 ms per flush, do not port detectors to Rust.

### Weeks 8–10: evidence for engineers and agents (Gate 3)

18. Verify `python -m tensorlens` exporting both local `.jsonl` and server `part-*.parquet` runs to `.rrd` + blueprint.
19. Build `sdk/src/tensorlens/screenshots.py` writing headless PNGs and `.meta.json` (`requested_step`, `sample_step`).
20. Verify headless Linux rendering in an Ubuntu container with `mesa-vulkan-drivers` (`lavapipe`) and `libvulkan1`: two repeat renders of the same `.rrd` and step must produce byte-identical PNGs.
21. *Postponed unless ≥3 engineers join:* MLflow scalar-only importer (`sdk/src/tensorlens/mlflow_import.py`).
22. Build TensorLens's own Rune app in `viewer/`, in the order Gate 1 set; Rerun stays the viewer until it reaches parity.
23. **Gate 3 checklist (pass both to proceed):**
    - [ ] Across ≥8 seeded failure runs (covering MoE collapse, starved experts, frozen bf16 norms, gradient spike, activation outlier, and 2-GPU replica drift), a coding agent given TensorLens `evidence.json` + Rerun screenshots identifies the exact failing step, layer/parameter, and failure kind in **≥85%** of trials, beating the same agent given scalar loss + total grad-norm curves alone by **≥30 percentage points**.
    - [ ] TensorLens has run on ≥1 of the author's real training jobs and either surfaced a real anomaly or completed ≥1,000 steps with zero false-positive incidents.
    - *Cut order if blocked:* if headless PNGs do not improve agent accuracy over `evidence.json` alone, make screenshot generation optional for agents and keep Rerun for human inspection.

### Weeks 11–12: public release (Gate 4)

24. Rename the Python package from placeholder `tensorlens` to an available PyPI/GitHub name, add `LICENSE-APACHE` and `LICENSE-MIT` (`Apache-2.0 OR MIT`), and publish the SDK, detectors, exporter, and `tensorlens-server` binary with the MoE bake-off as a self-contained demo (no external Rune checkout required). The open-source SDK and server include **zero outbound telemetry**.
25. Track adoption through public signals that require no SDK phone-home: GitHub stars, issues, PRs, PyPI download stats, and direct conversations with users who open issues or discussions.
26. Measure on the dogfood runs: (a) Parquet storage bytes per 1,000 steps per rank, (b) `tensorlens-server` CPU seconds and peak RSS per rank, and (c) headless screenshot render seconds per incident. (Diagnosis runs on the user's own coding agent, so TensorLens incurs zero LLM token cost.)
27. **Gate 4 checklist:**
    - [ ] Package and `tensorlens-server` install cleanly in a fresh environment and pass the smoke test suite.
    - [ ] ≥3 outside engineers or teams run the SDK on their own jobs and report results via GitHub issues/discussions, with ≥2 expressing willingness to pay for managed multi-node ingest and alerting at a stated monthly price before any hosted control plane is built.

## Free and paid

Everything that runs on the user's own machine or cluster is free and open source. The paid tier, built only if Gate 4 passes, is managed multi-tenant hosting and alerting.

- **Free (`Apache-2.0 OR MIT`):** the capture SDK, opt-in Flight sender, self-hosted `tensorlens-server`, detectors, and Rerun evidence exporter, with zero telemetry.
- **Paid, \$250–500 per seat per month (unvalidated, post-Gate 4):** managed `tensorlens-server` cloud ingest, object-store retention, Slack/PagerDuty incident alerts, and automated PR drafts.
- **Self-hosted enterprise:** managed deployment, SSO/RBAC, and SLA inside the customer's own cloud account.

Cost per monitored run (Parquet storage bytes, `tensorlens-server` CPU/RAM, and headless screenshot rendering; zero LLM calls since the customer's coding agent diagnoses incidents) is measured in Step 26 before setting price.

## Risks

The biggest risk is commercial, not technical: teams may value the tool and still not pay for hosted monitoring.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Nobody pays for hosted monitoring | No revenue | Release the SDK and self-hosted `tensorlens-server` free first; build a multi-tenant hosted cloud only if Gate 4 shows ≥2 outside teams willing to pay |
| ARIA, Pluto, or TrainCheck adds compiled per-layer capture and MoE/attention detectors | The gap closes | Re-check all competitors at Gate 0 and before Gate 4; stay tracker- and cloud-neutral; lead with MoE, attention, cross-GPU drift, and zero-break `torch.compile` support |
| Trackers add per-layer views | The wedge narrows | Stay tracker-neutral; lead with MoE, attention, and `torch.compile` capture that scalar trackers lack |
| Rerun's extension interfaces change | The interim export breaks | Pin `rerun-sdk==0.43.2`; only the `.rrd` exporter depends on Rerun |
| The viewer depends on Rune, itself under heavy development | Rune's schedule delays the viewer, or a Rune change breaks it | Keep the Rerun export as a working fallback until parity; pin a Rune commit in `viewer/`; keep the app's input to `evidence.json` and JSONL so it can move to another renderer |
| Hooks stall training or break `torch.compile` | Users switch the SDK off | Overhead gate ≤1% for default `capture="routers"`, CI gate with `error_on_recompile` and `graph_break == 0`, and bounded writer queue (`WRITER_QUEUE = 8`) |
| Agents misread the pictures | Wrong diagnoses, lost trust | Every number travels in `evidence.json` (`schema_version: 1`); screenshots include `.meta.json` with `sample_step`; score on Gate 3 |
| Coding-agent clients handle tool images differently | The picture does not reach the model or the user | Test saved PNGs with target agent clients and always supply `evidence.json` alongside |
| Tensor data leaves the cluster | Enterprise deals fall through | Local `.jsonl` by default (zero network calls); opt-in Flight sender targets self-hosted `tensorlens-server` on the user's own cluster |
| W&B's terms restrict competing products and credential sharing | A W&B importer or hosted W&B sync breaches the customer's contract | Read only `wandb.run.id` in-process; require legal review before anything touches a customer's W&B API key |

## Unproven and open decisions

The synthetic MoE bake-off verified recording read-back and headless rendering. Capture now runs on a real MoE model (Granite) on single-GPU RTX PRO 6000 and A100 instances, with router-only overhead under 1% on the RTX PRO 6000 (+0.02% to +0.78%) and +1.06% to +1.33% on the A100, a seeded layer-12 collapse detected, and eleven incident detector kinds implemented in `evidence.py` and checked on Granite or synthetic fixtures. Full capture with block defaults measured +4.15% on the RTX PRO 6000. Multi-GPU NCCL capture, Linux `lavapipe` rendering, the Rust ingest server, real failures on the author's own jobs, and coding-agent diagnosis accuracy remain unmeasured.

**Unproven**

- Which Rune-app interactions (sorting, layer selection, linked table) cut diagnosis time or agent tool calls most in Gate 1.
- Whether `capture="routers"` on the A100 can be brought under 1% once profiled with `profile_overhead.py`, and whether `capture="full"` (+4.15%) needs optional gradient-norm step sampling.
- Whether the detector thresholds (5× spike, 0.6 diversity drop, 10% starved share, 3 unchanged weight checks) hold on models other than Granite without false alarms.
- Whether the hooks keep CUDA graphs on real models that support CUDA graphs (verified on the toy MoE; Granite's `grouped_mm` fails CUDA-graph capture even without TensorLens).
- Whether the cross-GPU drift tolerance of `1e-6` holds on healthy multi-GPU NCCL runs without false positives from floating-point nondeterminism.
- Whether a coding agent with TensorLens `evidence.json` and Rerun screenshots beats scalar-only tracker data by ≥30 percentage points in Gate 3.
- Whether outside teams want and will pay for hosted monitoring (Gate 4).

**Decisions**

- [ ] **Product and PyPI package name** ("TensorLens" is a placeholder; `tensorlens` is taken on PyPI and GitHub). Pick an available name before Gate 4.
- [ ] **Author's dogfood job, tracker, and multi-GPU setup** (Gate 0).
- [ ] **Price for the post-Gate 4 hosted tier**, after Step 26 measures storage and server cost per run.
- [x] **Viewer (decided Nov 14, 2026):** Rerun is temporary. TensorLens builds its own Rust viewer app on Rune that reads `evidence.json` and the JSONL/Parquet data directly; the `.rrd` export stays as the interim viewer until the app reaches parity. This replaces the earlier rule that a custom view was built only after a ≥2× win in Gate 1.
- [x] **Framework adapters (decided Nov 14, 2026):** PyTorch stays the only capture engine; adapters are thin and exist only where a framework hides the optimizer, clipping or scaler loop. Order:
  1. A Hugging Face `TrainerCallback` (`sdk/src/tensorlens/transformers.py`), accepted on a dense single-GPU CUDA run. It does not wait for the multi-GPU gate. It covers TRL and PEFT; the first TRL test is `SFTTrainer` with LoRA, then DPO and GRPO separately (GRPO can generate rollouts in a separate vLLM server, which trainer-side capture does not see).
  2. An Accelerate recipe for custom loops.
  3. In parallel, validation of what is already supported: DDP over NCCL, then FSDP2 and tensor parallelism, Megatron-LM's `auto_step=False` recipe, and Lightning with `torch.compile`.
  4. One PPO example and one LeRobot imitation-learning example, to find what RL and robotics runs need beyond model internals (rollout and update boundaries, a link to policy evaluation).
  5. DeepSpeed only when someone brings a workload. JAX, TensorFlow, tree models and FSDP1 stay out of scope.

  Checked against the Transformers (5.24.0.dev0) and Accelerate (1.16.0.dev0) sources, which shape the callback:
  - `Trainer` wraps, prepares and (through Accelerate's `TorchDynamoPlugin`) compiles the model before `on_train_begin`. Callbacks receive the unwrapped `trainer.model`. The callback must attach hooks before the first compiled forward, and a second `train()` on the same trainer is an open case.
  - Step boundaries are `on_pre_optimizer_step` and `on_optimizer_step`, which surround `optimizer.step()`; TensorLens's own optimizer hooks already fire there. `on_log` fires every `logging_steps` with interval averages, so those are logged as separate series, not as per-step loss.
  - `accelerator.clip_grad_norm_` no longer calls `torch.nn.utils.clip_grad_norm_` (it uses `get_total_norm` and `clip_grads_with_norm_`), so the clip wrapper records nothing under `Trainer` or Accelerate. The callback has to capture the norm another way, such as wrapping `accelerator.clip_grad_norm_`.
  - `AcceleratedOptimizer.step` calls `scaler.step` and `scaler.update` itself, so `scaler=accelerator.scaler` should work unchanged.
  - Ray actors that share no process group all resolve to rank 0 and, given the same `run_id` and folder, append to one file. v1 answer: one learner writes, or each process passes its own `sink=` (documented in `RUN_ON_YOUR_JOB.md`). Per-role streams wait until actors are instrumented.
- [x] **CI (decided Nov 14, 2026):** GitHub Actions runs the CPU matrix on every push to `main` and every pull request, installing CPU-only PyTorch wheels. The GPU compiler gate is a manual `workflow_dispatch` job on a self-hosted runner: GitHub's free runners have no GPU, and a job that pull requests cannot start keeps outside code off that runner. On CPU the gate script runs on `aot_eager`, so Inductor and CUDA graphs are checked only by the `gpu` job.
- [x] **Default capture level (decided Nov 13, 2026):** `capture="routers"` is the always-on default for MoE models, and `capture="full"` is an opt-in debugging mode.
- [x] **Multi-GPU reporting (decided Nov 14, 2026):** every rank writes its own `tensorlens-out/<run_id>.rank<rank>.jsonl` and computes `replica_spread` on `replica_group` every `replica_every=50` steps where `(step + 1) % replica_every == 0`.
- [x] **Ingest architecture and v1 server hosting (decided Nov 14, 2026):** the SDK always writes the local `.jsonl` file first; when `endpoint=` / `TENSORLENS_ENDPOINT` is set, an opt-in sender tails that file from `<run_id>.rank<rank>.offset` and streams over Arrow Flight `DoPut` to the open-source, self-hosted Rust `tensorlens-server` binary (`server/`), which deduplicates by `(run_id, rank, step)` and writes Zstd Parquet parts via `object_store`.
- [x] **Where detectors run (decided Nov 14, 2026):** detectors run in Python (`sdk/src/tensorlens/evidence.py`) on local `.jsonl` or live server Parquet (`python -m tensorlens --watch`), and will only be ported into the Rust server if the 64-rank benchmark exceeds 250 ms per 50-step flush or 1 CPU core.
- [x] **Who explains failures (decided Nov 14, 2026):** v1 hands `evidence.json` and Rerun screenshots to the team's own coding agent; a built-in VLM explainer is postponed.
- [x] **Tier 3 activation replay and MLflow scalar importer (decided Nov 14, 2026):** both postponed from v1 (revisit the MLflow importer only if ≥3 engineers join).

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
