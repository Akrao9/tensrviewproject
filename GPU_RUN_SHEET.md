# GPU run sheet for step 5 of the v1 plan

Written Nov 22, 2026 against the harness wave's code commit `195ed27` (wave AI; the bug list's section 14), with the author's terms of that day for the run: smoke the first-use branches first; measure the loader delay with capture off and on before any fix; the fault-injection pass rule; the DDP wrapper's graph breaks shown in a control run and an instrumented one; the Granite gate scored by its original criteria. **Nothing here has run on a GPU.** Every command was read from the scripts on that commit, and the three lanes that wrote the new harnesses gave their own lines (`lane-ai1.md`, `lane-ai2.md`, `lane-ai3.md` in the side-session notes).

The "CPU" column says what was done on this Mac (macOS, CPU, PyTorch 2.15, offline): *dry-run* means the same script ran to its stated exit status with `--toy`, `--cpu` or a stand-in in place of the real model, router or NCCL; *refused* means the script stopped as it should without a GPU; *never run* means that branch of the script has executed nowhere. A CPU dry run shows that a harness starts, makes its checks and writes its files. Only the GPU run establishes the claim.

The GPU proofs run on the revision that passes Linux CI, not on this one if a fix moves it. Write that revision at the top of the results.

**Exit statuses**, the same for the replica driver and every new harness: 0 PASS, 1 FAIL, 2 bad arguments or a missing requirement, 3 NOT PROVEN (a part could not be checked: that is a result to record, not a failure of the harness).

## What the GPU machine needs that this Mac did not provide

| Needed | From | For |
|---|---|---|
| The repository at the CI-green revision | the private remote | everything |
| At least 2 CUDA GPUs; 8 on one node for sections 3 and 6 | | |
| Rust toolchain, to build `tensorquill-server` | rustup, or a server wheel from CI | section 3 |
| `torch` with CUDA and NCCL, `pyarrow`, `transformers`, `accelerate`, `huggingface_hub` | pip, per `sdk/pyproject.toml` extras | sections 1 to 4 |
| `lightning` (2.6.6 was used here) | pip, the `lightning` extra | section 4 |
| `ibm-granite/granite-3.0-1b-a400m-base` | Hugging Face | sections 1 and 2 |
| `HuggingFaceTB/SmolLM2-360M` | Hugging Face | sections 1 and 4 |
| `Salesforce/wikitext`, `wikitext-2-raw-v1` train parquet | Hugging Face | sections 1, 2 and 4 |
| `wandb` (offline mode: no account, no network for the run) | pip | section 2, the `wandb.watch` arm |
| NeuralDBG: the clone at `../NeuralDBG` (1.5.0 ran here), or `pip install neuraldbg` | sibling clone or pip | section 2, the NeuralDBG arm |
| `nvidia-ml-py` and `nvidia-smi` on the path | pip, the `nvml` extra; the driver | section 1, the system check |
| `megatron-core` (needs `triton`, which comes with CUDA torch), or a Megatron-LM clone | pip or `../Megatron-LM` | section 4, Megatron |
| **A second environment with PyTorch 2.10 to 2.13**, `pip install -e <lm-resiliency>`, TensorQuill installed in it | `../lm-resiliency` | section 6 only: it pins `torch>=2.10,<2.14` |

Neither model is in this Mac's cache, so no real-model line was dry-run. Installing and downloading on the GPU machine is the author's step. A missing package, model or dataset makes each new script print one line saying what to install and exit 2.

## Before anything

```bash
git rev-parse HEAD                      # must be the CI-green revision
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
(cd server && cargo build --release) && export TENSORQUILL_SERVER_BIN=$PWD/server/target/release/tensorquill-server
$TENSORQUILL_SERVER_BIN protocol        # must print 4
export OUT=$PWD/tensorquill-out/v1-gpu-$(git rev-parse --short HEAD) && mkdir -p $OUT
cd sdk
```

Every command below is run from `sdk/` through this, which keeps its output and exit status:

```bash
run() { name=$1; shift; echo "$ $*" > $OUT/$name.log; "$@" >> $OUT/$name.log 2>&1; echo "$name: exit $?" | tee -a $OUT/RESULTS.txt; }
```

## Where results go

`$OUT/` (`tensorquill-out/v1-gpu-<revision>/` at the repository root; `.gitignore` has `tensorquill-out/`) on the GPU machine: one `<name>.log` and one folder per name below, `RESULTS.txt` with each name's exit status, and a first file `ENVIRONMENT.txt` holding the output of the "Before anything" lines. Each new harness also writes a small result file under its `--out` (`gate.json`, `paired_dtype.json`, `coverage.json`, `system.json`, `result.json`, `tensorquill_comparison.json`). The folder is copied back whole. Nothing is entered in the bug list or the benchmarks file except from it. Each claim made from it carries: command, hardware, PyTorch and CUDA versions, the raw capture, the expected outcome, the observed outcome.

## 0. Smoke first: the branches that have never run

Before anything expensive. Results go to `$OUT/smoke/` and are **never scored**: a smoke line shows only that a branch starts, loads what it needs and writes its files on this machine. If one stops, fix the environment (or report the harness) before the scored runs; do not read a result from it. None of these branches could run on the Mac without installing, downloading or CUDA, so **every line here is never run** (one option combination, `gpu_probe_coverage.py --steps 50 --probe-every 50`, was dry-run with `--toy`: exit 0).

```bash
mkdir -p $OUT/smoke
# 1. the wandb.watch arm, never executed anywhere: first on the toy, then on Granite (one round, 50 + 50 steps)
run smoke-wandb-toy     python examples/granite_compiler_gate.py --out $OUT/smoke/wandb-toy --toy --arms off,tensorquill,wandb --rounds 1 --warmup 50 --steps 50
run smoke-gate-granite  python examples/granite_compiler_gate.py --out $OUT/smoke/gate-granite --arms off,tensorquill,wandb,neuraldbg --rounds 1 --warmup 50 --steps 50 --neuraldbg-path ../../NeuralDBG
# 2. both real models and the dataset load (Granite and WikiText-2 are loaded by the line above too)
run smoke-load-granite  python examples/gpu_probe_coverage.py --out $OUT/smoke/load-granite --model ibm-granite/granite-3.0-1b-a400m-base --steps 50 --probe-every 50
run smoke-load-smollm2  python examples/gpu_probe_coverage.py --out $OUT/smoke/load-smollm2 --steps 50 --probe-every 50
# 3. the NCCL environment: two GPUs, then the Flight Recorder variables with NCCL's dump on timeout
run smoke-nccl-2        python -m torch.distributed.run --nproc-per-node=2 examples/replica_check.py --out $OUT/smoke/nccl-2
run smoke-fr-nccl       python examples/flight_recorder_hang_check.py --out $OUT/smoke/fr-nccl --nproc 2 --skip-rank 1
# 4. the nvidia-smi field names the system check asks for, and nvidia-ml-py
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.total,memory.used,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total --format=csv,noheader,nounits | tee $OUT/smoke/nvidia-smi-fields.txt
python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlSystemGetDriverVersion())" | tee $OUT/smoke/nvml.txt
# 5. in the second environment (PyTorch 2.10 to 2.13): the campaign's imports and TensorQuill on that PyTorch
python -c "import torch, tensorquill, lm_resiliency; print(torch.__version__, torch.version.cuda, tensorquill.__file__)" | tee $OUT/smoke/fault-injection-env.txt
run smoke-fi-seeded     python examples/seeded_failures.py --case both --out $OUT/smoke/fi-seeded
python -c "import sys; sys.path.insert(0, 'examples'); import fault_injection_check_entry as e, json; print(json.dumps(e.environment(), indent=1))" | tee $OUT/smoke/fault-injection-environment.txt
# 6. back in the first environment: the framework branches that have never run, short
run smoke-lightning-fsdp python examples/lightning_check.py --out $OUT/smoke/lightning-fsdp --devices 2 --strategy fsdp --fsdp-wrap blocks --steps 100 --seed-step 60
run smoke-trainer-ddp    torchrun --nproc-per-node 2 --master-port 29551 examples/trainer_ddp_check.py --out $OUT/smoke/trainer-ddp --steps 120 --seed-step 60 --log-every 10 --probe-every 40
run smoke-trainer-ddp-control torchrun --nproc-per-node 2 --master-port 29552 examples/trainer_ddp_check.py --out $OUT/smoke/trainer-ddp-control --steps 120 --seed-step 60 --log-every 10 --probe-every 40 --control
run smoke-megatron       python examples/megatron_routing_check.py --out $OUT/smoke/megatron
run smoke-loader-delay   python examples/loader_delay_check.py --out $OUT/smoke/loader-delay --rounds 1
```

- **What to look for:** `smoke-wandb-toy` and `smoke-gate-granite`: the `wandb` arm's state is `measured` (offline mode: no account, no network), not `not_run` or `error`; NeuralDBG on Granite finishes inside the arm timeout. The two load lines: exit 0, which needs the checkpoints and WikiText-2 in the cache. `smoke-nccl-2`: exit 0. `smoke-fr-nccl`: **a dump from every rank, the sleeping one too**; exit 3 with "no dump" here means the scored Flight Recorder run will be not proven, so settle the NCCL variables first. The `nvidia-smi` line: a row of values, no "Field ... is not a valid field". `smoke-fi-seeded`, run in the second environment: TensorQuill's seeded cases pass on that PyTorch, which has never been tried. The `environment()` line prints the versions the campaign will record; it runs at the campaign's start on rank 0, so an error here would stop the eight-GPU campaign. Group 6: Lightning FSDP records steps and gradients at all; the `Trainer` under DDP gets through fp16 and Inductor, with and without TensorQuill; Megatron's own `TopKRouter` imports and routes; the loader measurement's loop runs with the device synchronized. The short step counts are the CPU tests' (the `Trainer` lines) or untried (Lightning): a failed spike check in a smoke is not a result.
- A smoke that fails is entered in `RESULTS.txt` like any line and reported; it is not rerun silently.

## 1. One GPU

| Name | Command | Expected | CPU |
|---|---|---|---|
| seeded-both | `run seeded-both python examples/seeded_failures.py --case both --out $OUT/seeded` | "All cases passed": the gradient spike and the stalled weights each found at the seeded step on the seeded parameter | dry-run |
| seeded-fp32 | `run seeded-fp32 python examples/seeded_failures.py --case stalled_weights --dtype float32 --out $OUT/seeded-fp32` | the stall found without BF16 | dry-run |
| seeded-spike | `run seeded-spike python examples/seeded_failures.py --case grad_spike --factor 100 --out $OUT/seeded-spike` | exit 0 | dry-run |
| seeded-healthy | `run seeded-healthy python examples/seeded_failures.py --case healthy --out $OUT/seeded-healthy` | no incident | dry-run |
| update-spike | `run update-spike python examples/seeded_failures.py --case update_spike --out $OUT/update-spike` | the incident at the seeded step on the seeded parameter | dry-run |
| paired-dtype | `run paired-dtype python examples/seeded_failures.py --case paired_dtype --out $OUT/paired-dtype` | BF16 and FP32 from one float32 model and the same batches: each capture complete, TensorQuill's loss and gradient norm equal the script's own on every step, first losses within 2%. Where the two arms differ is reported, not judged | dry-run (7 of 7 checks) |
| compile-gate-toy | `TORCH_LOGS="graph_breaks,recompiles,cudagraphs" run compile-gate-toy python examples/gpu_check.py` | zero graph breaks, zero recompiles after warm-up, zero CUDA-graph skips; routing totals verified | dry-run (`aot_eager`) |
| dense-outlier | `run dense-outlier python examples/dense_outlier_check.py --out $OUT/dense-outlier` | the SmolLM2 activation outlier found at the seeded step; "Graph breaks: 0" | dry-run |
| trainer-fp16, -bf16, -fp32 | `run trainer-fp16 python examples/trainer_check.py --out $OUT/trainer-fp16 --precision fp16`, then `bf16`, then `fp32` | "All checks passed" under the Hugging Face `Trainer`; under `fp16` the count of real scaler skips is reported, whatever it is | dry-run (`fp32` only) |
| data-checkpoint | `run data-checkpoint python examples/data_checkpoint_check.py --out $OUT/data-check` | "PASS" for the bad and the good case | dry-run |
| probe-coverage | `run probe-coverage python examples/gpu_probe_coverage.py --out $OUT/probe-coverage` and `run probe-coverage-granite python examples/gpu_probe_coverage.py --out $OUT/probe-coverage-granite --model ibm-granite/granite-3.0-1b-a400m-base` | every block has a histogram, every attention call the script itself counted has an entry with one value per head, and the head's positions add up to batch × sequence, at each probe pass | dry-run on the toy (15 of 15) and on a tiny random Granite MoE config; the real checkpoints never loaded |
| system-check | `run system-check python examples/gpu_system_check.py --out $OUT/system-check` | TensorQuill's data wait equals a known injected sleep and the script's own timer; its CUDA memory equals `torch.cuda` and `nvidia-smi`; its NVML readings equal `nvidia-smi` sampled alongside, within stated tolerances | dry-run: exit 3, data wait checked, CUDA and NVML not proven. The `nvidia-smi` field names have never run: a rejected field gives exit 3 with the error |
| granite-collapse | `run granite-collapse python examples/moe_dogfood.py --out $OUT/granite-collapse --collapse-layer <N>` | routing collapse reported on layer N in the final round; N is chosen and written in `ENVIRONMENT.txt` before the run | refused (needs CUDA with BF16) |

### The loader delay: measured with capture off and on, before any fix

```bash
run loader-delay python examples/loader_delay_check.py --out $OUT/loader-delay
```

- **Open, and to be measured here before a fix lane starts (the author, Nov 22).** On CPU, with TensorQuill on and a 50-step flush, one step of a run spends tens of milliseconds longer in the loader's `next()` than the loader's sleep; with TensorQuill off none does.
- **What the line keeps:** three rounds, capture off and capture on in fresh processes, the same loop as `system-check`. For every step of every arm and round: the injected sleep, the time in `next()`, the step's whole time, and for the `on` arm TensorQuill's own `data_wait` and `step_time` (`loader_delay.json`, `arm-<arm>-r<round>.json`). The flush-boundary steps are reported by name, **step 100 among them**, for both arms; so are the worst delay per arm with its step and round, every step over 15 ms, the slowest rest of a step, and each garbage collection of 1 ms or more with the thread it ran on.
- **Nothing is judged:** exit 0 means both arms were measured.
- **What CPU showed (PyTorch 2.15, macOS; not a GPU result):** in 5 of 5 `on` rounds one generation-2 garbage collection of 78 to 92 ms ran on TensorQuill's writer thread at a flush boundary (step 150 in this loop), while the training thread was in `next()`, which was 60 to 74 ms over its sleep. The `off` arm had no collection at all. `system-check`'s own loop shows its delay at step 100 (36 and 62 ms in two runs); with a collection callback around it the delay moved to step 150, beside a 92.5 ms collection. Where the pause lands moves with what is watching, so read every step, not step 100 alone. That is evidence of a cause, on one machine. It is not a fix and not a GPU measurement.
- **How it is reported:** the worst delay (this line, both arms) **beside** the preregistered median overhead (section 2). A passing median does not explain the spike, and the spike is reported whatever the median says. The toy loop's own step-time overhead in `loader_delay.json` is labelled as such and is not the preregistered figure.
- `system-check` still lists every step over its slack with both timers in `system.json`.

## 2. The Granite compiler gate, four arms

```bash
TORCH_LOGS="graph_breaks,recompiles,cudagraphs" run granite-gate-4arm python examples/granite_compiler_gate.py --out $OUT/granite-gate-4arm --neuraldbg-path ../../NeuralDBG
# only if every arm, capture off too, shows CUDA-graph skips (see below):
run granite-gate-4arm-nocg python examples/granite_compiler_gate.py --out $OUT/granite-gate-4arm-nocg --mode default --neuraldbg-path ../../NeuralDBG
run profile-routers python examples/profile_overhead.py --out $OUT/profile --capture routers --steps 200 --warmup 50 --rounds 3
run profile-full python examples/profile_overhead.py --out $OUT/profile-full --capture full --trace
```

- **The arms:** capture off, TensorQuill router-only, NeuralDBG, `wandb.watch`. Each runs in a fresh process with the same model, the same batches, 50 warm-up and 200 timed steps, three paired rounds, the order rotating by round. `gate.json` holds every arm and round, the failing and absent ones too, with what each tool observes and how often it samples.
- **Pass line, for the TensorQuill arm, the original criteria unchanged:** median overhead 1% or less against capture off (per round, on mean step time, so the flush steps count); **zero graph breaks**; zero recompiles after warm-up; zero CUDA-graph skips under `reduce-overhead`; every step recorded and every router's assignments summing to tokens × top-k. The three breaks of DDP's wrapper (section 4) are another run's observation and allow nothing here.
- **Reported beside the pass line, never scored:** the slowest timed step of every arm and round with how far it is over that round's median step (`worst_step_by_round`; every step's time is `step_ms` in `arm-<arm>-r<round>.json`), for capture off and on alike; and the overhead as `moe_dogfood.py` printed it on earlier GPU runs (each arm's median round, then the ratio), said aloud if it falls on the other side of 1% from the scored figure. **Which of the two figures is "the preregistered median" is one line for the author to confirm:** the harness scores the paired-round one, as the plan's "three paired rounds" reads.
- **This loop has no data loader** (the batches are made before timing), so it cannot show a delay in `next()`. A pause of the training thread would show here as a slow step. The loader delay is section 1's `loader-delay`.
- **Statuses to expect:** an arm whose package is absent is `not_run`, never passed. A run without CUDA graphs is `not_proven` (exit 3). Granite under `reduce-overhead` may skip CUDA graphs in every arm, capture off included (`moe_dogfood.py` notes it on an RTX PRO 6000 with PyTorch 2.14.1): then the cause is the model, and the second line gives the other criteria with skips "not checked".
- **CPU:** dry-run with `--toy` (12 arm processes; off, TensorQuill and NeuralDBG measured; `wandb` `not_run`). **The `wandb` arm has never run:** it is written from wandb's documented calls. NeuralDBG has never run on Granite and will be slow (hooks with several host reads on every leaf module); raise `--arm-timeout` if an arm passes an hour.
- `moe_dogfood.py` (two arms, in one process) is kept for `granite-collapse`; the gate is the new script.

## 3. Distributed, over NCCL

Two GPUs first as a smoke, then eight on one node, each through files and through a live server.

```bash
T="python -m torch.distributed.run"
run replica-2-files     $T --nproc-per-node=2 examples/replica_check.py --out $OUT/replica-2-files
run replica-8-files     $T --nproc-per-node=8 examples/replica_check.py --out $OUT/replica-8-files
run replica-8-learning  $T --nproc-per-node=8 examples/replica_check.py --out $OUT/replica-8-learning --arm learning --steps 40
run replica-8-clean     $T --nproc-per-node=8 examples/replica_check.py --out $OUT/replica-8-clean --no-plants
run replica-8-bits      $T --nproc-per-node=8 examples/replica_check.py --out $OUT/replica-8-bits --plant 1:2:0.weight:negzero --plant 7:4:1.weight:ulp
run replica-8-late      $T --nproc-per-node=8 examples/replica_check.py --out $OUT/replica-8-late --hold-rank 7 --release after-verdict
run replica-8-missing   $T --nproc-per-node=8 examples/replica_check.py --out $OUT/replica-8-missing --hold-rank 7 --release never

# a live server, in its own shell, stopped after the two runs
$TENSORQUILL_SERVER_BIN --listen 127.0.0.1:50051 --store $OUT/store --watchdog-seconds 0.5 --flush-seconds 0.3
run replica-8-server    $T --nproc-per-node=8 examples/replica_check.py --mode server --endpoint grpc://127.0.0.1:50051 --store $OUT/store --status-interval 0.5 --out $OUT/replica-8-server --steps 24 --step-seconds 0.3
run replica-8-server-learning $T --nproc-per-node=8 examples/replica_check.py --mode server --endpoint grpc://127.0.0.1:50051 --store $OUT/store --status-interval 0.5 --out $OUT/replica-8-server-learning --arm learning --steps 24 --step-seconds 0.3
```

- Each run leaves a `verdict.json` under its `--out`. The default plants are three, on steps 1, 3 and 5. With delayed or missing telemetry, training keeps moving and coverage reads as incomplete.
- Reported for each: the processing latency, with the clock started when the server has stored and acknowledged the last required observation, and the end-to-end time from the training step to the verdict.
- `--healthy-mismatch report` (the default) counts healthy steps whose signatures differ: how often that happens on eight GPUs decides the signature-only replica rule (the plan's step 6.1).
- CPU: dry-run on 2 and 3 Gloo ranks for the files arm, the learning arm, `--no-plants`, the `negzero` and `ulp` plants and both `--release` values; the server lines are the wave exit gate's (2, 3 and 8 CPU ranks).

## 4. Frameworks

```bash
# Lightning: a real LightningModule (SmolLM2-360M on WikiText-2); Lightning launches the ranks itself
run lightning-ddp        python examples/lightning_check.py --out $OUT/lightning-ddp --devices 2
run lightning-fsdp       python examples/lightning_check.py --out $OUT/lightning-fsdp --devices 2 --strategy fsdp --fsdp-wrap blocks
run lightning-fsdp-flat  python examples/lightning_check.py --out $OUT/lightning-fsdp-flat --devices 2 --strategy fsdp --fsdp-wrap none
# Hugging Face Trainer under DDP, torch_compile, 2 GPUs
run trainer-ddp-fp16     torchrun --nproc-per-node 2 --master-port 29541 examples/trainer_ddp_check.py --out $OUT/trainer-ddp-fp16
run trainer-ddp-bf16     torchrun --nproc-per-node 2 --master-port 29542 examples/trainer_ddp_check.py --out $OUT/trainer-ddp-bf16 --precision bf16
# the same two runs with no TensorQuill callback: Dynamo's graph breaks per rank, for the control side
run trainer-ddp-fp16-control torchrun --nproc-per-node 2 --master-port 29543 examples/trainer_ddp_check.py --out $OUT/trainer-ddp-fp16-control --control
run trainer-ddp-bf16-control torchrun --nproc-per-node 2 --master-port 29544 examples/trainer_ddp_check.py --out $OUT/trainer-ddp-bf16-control --precision bf16 --control
# Megatron's own TopKRouter, one GPU, capacity dropping off, on, and on with padding
run megatron-routing     python examples/megatron_routing_check.py --out $OUT/megatron-routing            # installed megatron-core; or add --megatron /path/to/Megatron-LM
run megatron-position    python examples/megatron_routing_check.py --out $OUT/megatron-routing-position --drop-policy position
```

- **Lightning checks:** one capture per rank, closed cleanly; each rank's recorded steps and `train/loss` equal the `global_step` and loss its own module logged; the ranks' losses differ; the planted activation spike is at the planted step in the planted block on every rank. Under mixed precision the loss comparison may need `--loss-tolerance 1e-3` (default 1e-5).
- **`Trainer` checks:** one capture per rank; steps equal the `Trainer`'s `global_step`; the mean over ranks of the recorded loss equals the loss the `Trainer` logged; each rank's clip norm equals its logged `grad_norm`; under `fp16`, each rank's recorded skip equals Accelerate's own flag step by step; spike and probe as in `trainer_check.py`.
- **Megatron check:** the top-k map before dropping is read from Megatron's own `apply_router_token_dropping` and the map after from what the router returned; TensorQuill's per-expert counts must equal kept, padding, and their sum equal what the dispatcher sends, with dropping off, on, and on with padding. Guards make each arm fail if nothing was dropped or padded where it must be.
- **CPU:** Lightning DDP dry-run on two CPU processes (exit 0); FSDP refused by Lightning 2.6.6 itself without CUDA (exit 2), so **both FSDP lines have never run**; the `Trainer` under DDP dry-run on two processes, eager and `aot_eager` (exit 0); the Megatron check dry-run against **a stand-in router with Megatron's own dropping code**, read from the clone. **Megatron's `TopKRouter` itself has never run**: its package needs `triton` and its gating needs CUDA.
- **The DDP wrapper's graph breaks (the author, Nov 22: documented as observed in both control and instrumented runs).** Under DDP, Dynamo breaks inside `DistributedDataParallel.forward` itself: three times on CPU with PyTorch 2.15 and `aot_eager`, the same reasons and counts with and without TensorQuill (a test holds the two equal). On the GPU the `-control` lines give the count without TensorQuill and the checked lines the count with it, per rank, in each `result.json` (`ddp_wrapper_breaks`, `other_breaks`). The checked run fails on any break outside the wrapper; the wrapper's are reported, not failed. **If the two runs' counts differ, report it: the difference would be TensorQuill's.** Inductor under DDP has never run. The Granite gate (section 2) keeps its zero-break line.
- **If Lightning FSDP records no steps or no gradients:** stop and report. That would be the SDK's to fix, not the harness's.
- TensorQuill records per-expert counts only, so "dropped" in the Megatron check is derived from Megatron's pre-drop map. Reporting drops itself would be an SDK change, not asked for.

## 5. Flight Recorder

```bash
run fr-hang    python examples/flight_recorder_hang_check.py --out $OUT/fr-hang --nproc 2 --skip-rank 1
run fr-hang-4  python examples/flight_recorder_hang_check.py --out $OUT/fr-hang-4 --nproc 4 --skip-rank 2   # more than one rank waiting
```

- **What it does:** a small job in which one rank does not enter one `all_reduce` at step 5, so the others hang for real. The driver sets the Flight Recorder variables (`TORCH_FR_BUFFER_SIZE`, `TORCH_FR_DUMP_TEMP_FILE`, `TORCH_NCCL_ENABLE_MONITORING`, `TORCH_NCCL_DUMP_ON_TIMEOUT`, `TORCH_FR_DUMP_ON_TIMEOUT`, and the two former names), waits a bounded time (150 s by default), ends every process itself, fails if any is left, and runs PyTorch's own analyzer and `python -m tensorquill ... --flight-recorder` on the dumps PyTorch wrote.
- **PASS:** TensorQuill's first `collective_hang` names `all_reduce`, sequence 6, the rank at fault `[1]` and the waiting ranks, at step 3 (the newest step is held back by design), with every rank's dump present.
- **NOT PROVEN (exit 3):** no dump, a rank without a dump, or dumps PyTorch's own analyzer cannot read. **If the sleeping rank writes no dump the result is NOT PROVEN: that is the likeliest GPU outcome to watch for.** If NCCL numbers its first collective otherwise, the check fails with both numbers printed: decide then, do not re-derive the expectation from the result.
- **CPU:** dry-run, exit 3. The hang, the bound and the ending work and no process is left; PyTorch writes the dumps on request (Gloo has no dump on timeout); but PyTorch 2.15's own analyzer cannot read its Gloo dumps (`KeyError`, fixed upstream after 2.15). The check against the plan is tested on dumps built in the analyzer's test format. **The NCCL path has never run.**

## 6. Independent fault injection (`lm-resiliency`), in its own environment

```bash
# in the PyTorch 2.10 to 2.13 environment, 8 GPUs
run fault-injection         torchrun --standalone --nproc-per-node=8 examples/fault_injection_check_entry.py --lm-resiliency /path/to/lm-resiliency --artifact-dir $OUT/fault-injection
run fault-injection-compare python examples/fault_injection_check.py --artifact-dir $OUT/fault-injection
```

- **What it does:** runs their campaign unchanged, with a TensorQuill session attached on every rank, and keeps the captures beside their `injection.json`. The comparison reads only `injection.json` as ground truth (never their own localization) and credits a TensorQuill report only with the exact rank and an iteration inside the injection's window.
- **The pass rule (the author, Nov 22):** an occurrence is found only when TensorQuill names the exact injected rank within the iteration window of `injection.json`. The campaign passes if every occurrence is found with no false report (exit 0); otherwise it fails (exit 1). **Partials, misses and false reports are published whatever the verdict**: they are all in `tensorquill_comparison.json`, whose `status` is the strict verdict. Missing ground truth or an incomplete run is **not proven** (exit 3): no `injection.json`, no `completed_iterations`, an injection still pending when the campaign ended, a rank's capture missing or ending before the campaign's last iteration. The scored line carries no `--measure`.
- **The environment:** the campaign's own (PyTorch 2.10 to 2.13). The entry script writes its exact versions on rank 0, inside that environment (`tensorquill/attach.json`: PyTorch, CUDA, NCCL, Python, TensorQuill), and the comparison carries them into the result as `campaign_environment`. Quote that with the result.
- `torchrun` may exit non-zero if their own tool misses something; the artifacts are still written, so run the comparison anyway.
- **To expect:** delay faults are probably misses (no server is attached); TensorQuill's hooks also fire during their replay forwards, which may add reports. Both are results to read, not to filter.
- **CPU:** the comparison and the wrapper dry-run on **a stub campaign of ours with a constructed `injection.json`** (exit 0, 3 of 3 found). **The real campaign has never run**, and TensorQuill has never run on PyTorch below 2.15 here (CI covers 2.6 and 2.9).

## 7. TrainCheck

A disposition for B5, B7, B11, B12, B13, B15, B16 and B20 from `experiments/traincheck-bugs`. No GPU command is written: the experiment's tests pass on CPU (18 in the exit gate), and which of the eight need a GPU has not been worked out.

## 8. The independent real job

**Not named (decision D10): the job, its tracker and its parallelism are the author's.** Once named: 100 steps as a smoke, then at least 1,000, with `tensorquill.init(...)` as `sdk/RUN_ON_YOUR_JOB.md` describes, a server on the node, and a watcher (`python -m tensorquill $OUT/store/runs/<run_id> --rank all --watch`). Every false incident is recorded. The author's failure table gets an entry only for a genuine example.

## 9. Correctness audit, pass 5

After the runs, on the same revision: the audit's fifth pass over the captures in `$OUT`.

## What the report of the run carries

- For the overhead claim: the preregistered median overhead (section 2's scored figure, with each round) **and** the worst delay, for capture off and on: `loader-delay`'s worst step and its step-100 rows, and the gate's slowest step per arm and round. Neither replaces the other.
- For fault injection: found, partial, missed and false reports with their counts, the verdict, and the campaign environment's exact PyTorch version.
- For the DDP `Trainer`: the wrapper's graph-break count in the control run and in the instrumented run, per rank.
- Every smoke line's exit status, apart from the scored results.

## What is ready, and what is not

- **Order:** section 0 (smoke, never scored), then sections 1 to 6.
- **Ready to execute as written:** sections 0 to 6.
- **Written and never executed anywhere, so the first GPU run is also the harness's first run (section 0 smokes each):** the `wandb` arm; NeuralDBG on Granite; Lightning FSDP; fp16 and Inductor under DDP, with and without TensorQuill (`--control`); Megatron's own router; NCCL's dump on timeout; the real fault-injection campaign and the entry script's `environment()` on CUDA; `loader_delay_check.py` on CUDA; every real-model load.
- **Needs the author first:** the job of section 8; which of section 7's cases need a GPU.
- Nothing from the earlier sheet's list is left without a script.
