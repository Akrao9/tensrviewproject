# GPU run sheet for step 5 of the v1 plan

Written Nov 22, 2026 against `main` at `542c9de` (code as gated at `626fff8`). **Nothing here has run on a GPU.** Every command was read from the scripts on that commit. The "CPU" column says what was done with it on this Mac (macOS, CPU, PyTorch 2.15, offline, toy models): *dry-run* means the same script and options ran to exit 0 with `--toy` or `--cpu` in place of the real model or NCCL; *refused* means the script stopped as it should without a GPU; *not run* means nothing here can run it.

The GPU proofs run on the revision that passes Linux CI, not on this one if a fix moves it. Write that revision at the top of the results.

## What the GPU machine needs that this Mac did not provide

| Needed | From | For |
|---|---|---|
| The repository at the CI-green revision | the private remote | everything |
| Rust toolchain, to build `tensorquill-server` | rustup, or a server wheel from CI | sections 3 and 5 |
| `torch` with CUDA, `pyarrow`, `transformers`, `accelerate`, `huggingface_hub` | pip, per `sdk/pyproject.toml` extras | sections 1 to 3 |
| `ibm-granite/granite-3.0-1b-a400m-base` (about 2.7 GB) | Hugging Face | `moe_dogfood.py`, `profile_overhead.py` |
| `HuggingFaceTB/SmolLM2-360M` (about 0.7 GB) | Hugging Face | `seeded_failures.py`, `dense_outlier_check.py`, `trainer_check.py` |
| `Salesforce/wikitext`, `wikitext-2-raw-v1` train split (about 1 MB) | Hugging Face datasets | `moe_dogfood.py` |
| `wandb`, and NeuralDBG from the sibling clone | pip; `../NeuralDBG` | the two comparison arms of section 2, if they are run |
| `lm-resiliency` (PyTorch 2.10 to 2.13 by its README) | `../lm-resiliency`, `pip install -e .` | section 6 |

The sizes are from the model cards as remembered, not measured here: neither model is in this Mac's cache. Installing and downloading on the GPU machine is the author's step.

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

`$OUT/` (`tensorquill-out/v1-gpu-<revision>/` at the repository root; `.gitignore` has `tensorquill-out/`) on the GPU machine: one `<name>.log` and one folder per name below, `RESULTS.txt` with each name's exit status, and a first file `ENVIRONMENT.txt` holding the output of the "Before anything" lines. The folder is copied back whole. Nothing is entered in the bug list or the benchmarks file except from it. Each claim made from it carries: command, hardware, PyTorch and CUDA versions, the raw capture, the expected outcome, the observed outcome.

## 1. One GPU

| Name | Command | Expected | CPU |
|---|---|---|---|
| seeded-both | `run seeded-both python examples/seeded_failures.py --case both --out $OUT/seeded` | "All cases passed", exit 0: the gradient spike and the stalled weights each found at the seeded step on the seeded parameter | dry-run |
| seeded-fp32-control | `run seeded-fp32 python examples/seeded_failures.py --case stalled_weights --dtype float32 --out $OUT/seeded-fp32` | exit 0: the stall found without BF16 | dry-run |
| seeded-spike-100 | `run seeded-spike python examples/seeded_failures.py --case grad_spike --factor 100 --out $OUT/seeded-spike` | exit 0 | dry-run |
| seeded-healthy | `run seeded-healthy python examples/seeded_failures.py --case healthy --out $OUT/seeded-healthy` | exit 0, no incident | dry-run |
| update-spike | `run update-spike python examples/seeded_failures.py --case update_spike --out $OUT/update-spike` | exit 0: the incident at the seeded step on the seeded parameter | dry-run |
| compile-gate-toy | `TORCH_LOGS="graph_breaks,recompiles,cudagraphs" run compile-gate-toy python examples/gpu_check.py` | zero graph breaks, zero recompiles after warm-up, zero CUDA-graph skips; routing totals verified | dry-run (`aot_eager`; overhead on a CPU toy is not the GPU figure) |
| dense-outlier | `run dense-outlier python examples/dense_outlier_check.py --out $OUT/dense-outlier` | the SmolLM2 activation outlier found at the seeded step; "Graph breaks: 0" | dry-run |
| trainer-fp16, -bf16, -fp32 | `run trainer-fp16 python examples/trainer_check.py --out $OUT/trainer-fp16 --precision fp16`, then `bf16`, then `fp32` | "All checks passed" under the Hugging Face `Trainer`; under `fp16` the count of real scaler skips is reported, whatever it is | dry-run (`fp32` only: no scaler on CPU) |
| data-checkpoint | `run data-checkpoint python examples/data_checkpoint_check.py --out $OUT/data-check` | "PASS" for the bad and the good case | dry-run |
| granite-collapse | `run granite-collapse python examples/moe_dogfood.py --out $OUT/granite-collapse --collapse-layer <N>` | routing collapse reported on layer N in the final round; N is chosen and written in `ENVIRONMENT.txt` before the run | refused (needs CUDA with BF16) |

**No script covers, so no claim is made from this sheet:** BF16 against FP32 on the same batches as an explicit pair; head and probe coverage counts; GPU memory, NVML and data-wait readings checked against `nvidia-smi` taken alongside. Each needs a script or stays unverified (the plan's step 5 allows that).

## 2. The Granite compiler gate

```bash
TORCH_LOGS="graph_breaks,recompiles,cudagraphs" run granite-gate python examples/moe_dogfood.py --out $OUT/granite-gate --steps 200 --warmup 50 --rounds 3 --cuda-graphs
run profile-routers python examples/profile_overhead.py --out $OUT/profile --capture routers --steps 200 --warmup 50 --rounds 3
run profile-full python examples/profile_overhead.py --out $OUT/profile-full --capture full --trace
```

Pass: router-only median overhead 1% or less; zero graph breaks; zero recompiles after warm-up; zero CUDA-graph skips under `reduce-overhead`. Every arm and its configuration is published, the failing ones too.

- CPU: `profile_overhead.py` dry-run with `--toy` for both captures; `moe_dogfood.py` refused.
- **`moe_dogfood.py` has no NeuralDBG arm and no `wandb.watch` arm** (read on this commit: it times capture off against TensorQuill's capture). The plan's four-arm comparison therefore needs two arms nobody has written. **Needs the author:** write them, or publish the gate as two arms and leave the comparison unverified.

## 3. Distributed, over NCCL

Two GPUs first as a smoke, then eight on one node, each through files and through a live server. Without `--cpu` the script uses CUDA and NCCL.

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

- Expected: exit 0 is PASS, 1 FAIL, 2 bad arguments, 3 not proven; each run leaves a `verdict.json` under its `--out`. The default plants are three, on steps 1, 3 and 5. With delayed or missing telemetry, training keeps moving and coverage reads as incomplete.
- Reported for each: the processing latency, with the clock started when the server has stored and acknowledged the last required observation, and the end-to-end time from the training step to the verdict. Both are in the run's output.
- `--healthy-mismatch report` (the default) counts healthy steps whose signatures differ: how often that happens on eight GPUs is what decides the signature-only replica rule (the plan's step 6.1).
- CPU: dry-run on 2 and 3 Gloo ranks for the files arm, the learning arm, `--no-plants`, `--plant ...:negzero` with `...:ulp`, and both `--release` values, each exit 0; the server lines are the wave exit gate's (2, 3 and 8 CPU ranks, exit 0). Rank 7 in the lines above was rank 2 or 1 here.

## 4. Frameworks

| Claim | State |
|---|---|
| The Hugging Face `Trainer` on one GPU | section 1, `trainer-*` |
| The `Trainer` on more than one GPU | **No script.** `trainer_check.py` is single-process |
| A real `LightningModule` under DDP and FSDP | **No script** under `sdk/examples/`. `lightning` 2.6.6 is installed here; the callback has tests on CPU |
| Megatron's routing map, capacity dropping on and off | **No script.** `../Megatron-LM` is cloned beside the project |

**Needs the author:** have these three scripts written before the GPU session (each is new code, so it goes through a lane and the exit gate), or leave the three claims unverified for v1.

## 5. Flight Recorder

```bash
export TORCH_FR_BUFFER_SIZE=2000 TORCH_FR_DUMP_TEMP_FILE=$OUT/fr/fr_trace_rank_ TORCH_NCCL_ENABLE_MONITORING=1 TORCH_NCCL_DUMP_ON_TIMEOUT=1
mkdir -p $OUT/fr
# ... a job with a forced collective hang ...
python -m tensorquill <the job's capture or run folder> --flight-recorder $OUT/fr --flight-recorder-prefix fr_trace_rank_
```

The variable names are the README's (its line on Flight Recorder), which says PyTorch 2.10 or later and "checked only on dumps built in the analyzer's own test format". **No script forces a collective hang**, and the names were not checked against the CUDA build that will run: confirm them there first. Expected, once a hang exists: `collective_hang` naming the stuck collective, and "rank waiting" told from "rank at fault". **Needs the author:** a hang script, or the claim stays unverified.

## 6. Independent fault injection (`lm-resiliency`)

Its own campaign, from `../lm-resiliency/tests/validation/fault_injection/README.md` (read, not run): `torchrun --standalone --nproc-per-node=8 --module ... --artifact-dir <dir>` writes `injection.json`, "executed campaign manifest and verified injection ground truth", and `python -m tests.validation.fault_injection.compare --artifact-dir <dir>` scores against it. **Not prepared:** TensorQuill is not attached to that workload, and how to attach it (an `init` in its training entry point, the capture kept beside `injection.json`) has not been worked out or tried. Its README names PyTorch 2.10 to 2.13; this project's floor is 2.6 and this Mac has 2.15. **Needs the author** before it can be a command.

## 7. TrainCheck

A disposition for B5, B7, B11, B12, B13, B15, B16 and B20 from `experiments/traincheck-bugs`. No GPU command is written: the experiment's tests pass on CPU (18 in the exit gate), and which of the eight need a GPU has not been worked out.

## 8. The independent real job

**Not named (decision D10): the job, its tracker and its parallelism are the author's.** Once named: 100 steps as a smoke, then at least 1,000, with `tensorquill.init(...)` as `sdk/RUN_ON_YOUR_JOB.md` describes, a server on the node, and a watcher (`python -m tensorquill $OUT/store/runs/<run_id> --rank all --watch`). Every false incident is recorded. The author's failure table gets an entry only for a genuine example.

## 9. Correctness audit, pass 5

After the runs, on the same revision: the audit's fifth pass over the captures in `$OUT`.

## What is ready, and what is not

- **Ready to execute as written:** section 1 except its three uncovered items; section 2's three commands (two arms); section 3.
- **Needs the author first:** the two comparison arms of section 2; the three framework scripts of section 4; a hang script for section 5; the attachment for section 6; the job of section 8.
- **CPU dry runs:** 18 commands, each exit 0, on `542c9de`, Nov 22, 2026 14:54 to 14:57, plus the plant line after; `moe_dogfood.py` refused with exit 2, as it must without a GPU.
