# Run TensorQuill on your own training job

This guide adds TensorQuill to a training job you already run. TensorQuill has so far been tested on a benchmark with planted failures; your job is the test that matters. It takes about 15 minutes to set up and one normal training run.

## 1. Install

From a clone of this repository, in the training job's Python environment:

```bash
python -m pip install -e 'sdk[rerun]'
```

Add extras in the same way when you need them: `sdk[lightning]`, `sdk[huggingface]` (Transformers and Accelerate), or `sdk[flight]` to stream to an ingest server. TensorQuill is not on PyPI yet, so install it from the clone as above. Keep the job's own PyTorch build. TensorQuill needs PyTorch 2.6 or later and has been run with 2.10 and 2.14.

## 2. Choose a capture mode

| Your model | Call | Cost measured on Granite |
| --- | --- | --- |
| MoE model | `tensorquill.init(model, routers={...})`: router-only, the default | +0.02% to +0.78% on an RTX PRO 6000; +1.1% to +1.3% on an A100 |
| MoE model, while debugging | add `capture="full"` | about +4% |
| Dense model | `tensorquill.init(model)`: full capture | about +4% on Granite; not measured on a dense model |

Router-only records each router's expert load every step. Full capture adds per-block activation statistics, per-parameter gradient norms every step and a weight check every 50 steps. In either mode, if your job clips gradients, the total gradient norm is recorded for free, which is enough to detect gradient spikes.

## 3. Find your router modules (MoE models only)

TensorQuill needs each router's module name, its number of experts, and how many experts each token picks (`top_k`). Router outputs differ between model families, so run this once on one batch to see what each candidate returns:

```python
import torch

def show_routers(model, batch, words=("router", "gate")):
    seen = []
    def hook(name):
        def fn(module, args, output):
            items = output if isinstance(output, (tuple, list)) else (output,)
            shapes = [
                (tuple(t.shape), str(t.dtype).replace("torch.", ""))
                for t in items if isinstance(t, torch.Tensor)
            ]
            seen.append((name, type(module).__name__, shapes))
        return fn
    handles = [
        module.register_forward_hook(hook(name))
        for name, module in model.named_modules()
        if any(w in name.rsplit(".", 1)[-1].lower() for w in words)
    ]
    with torch.no_grad():
        model(batch)  # use your model's normal call, e.g. model(input_ids=batch)
    for handle in handles:
        handle.remove()
    for name, kind, shapes in seen:
        print(name, kind, shapes)
```

Then describe each router with `tensorquill.Router`:

- **The router returns the chosen expert indices** (an integer tensor, shape `[tokens, top_k]` or `[tokens]`): pass `indices=` if it is not the first integer tensor in the output. IBM Granite's `GraniteMoeTopKRouter` returns `(logits, weights, indices)`, so the benchmark uses `tensorquill.Router(32, top_k=8, indices=lambda out: out[2])`.
- **The router returns only logits** (a float tensor whose last dimension is the number of experts): pass `top_k` and TensorQuill takes the top-k itself, for example `tensorquill.Router(8, top_k=2)`.

Always set `top_k` for top-k routers: the overload detector needs it. Check the printed shapes rather than assuming a model family's layout.

```python
routers = {
    f"model.layers.{i}.mlp.router": tensorquill.Router(64, top_k=8)  # your names and sizes
    for i in range(num_layers)
}
```

## 4. Add TensorQuill to the training loop

Call `init` after the model is on its device and before `torch.compile`. Pass your tracker's run ID if TensorQuill cannot find it on its own: it looks for `run_id=`, then `MLFLOW_RUN_ID`, then the active MLflow run, then the active W&B run. Without any of those it uses the launcher's job ID (torchrun's `TORCHELASTIC_RUN_ID`, or SLURM's job and step IDs), and otherwise generates a unique `run-<UTC time>-<6 hex>` ID, broadcast from rank 0 when `torch.distributed` is initialized so every rank writes under the same name.

### Plain PyTorch

```python
import tensorquill

model.to(device)
session = tensorquill.init(model, routers=routers, run_id=run_id)  # before torch.compile
model = torch.compile(model)

for batch in loader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    session.log({"train/loss": loss})                        # no GPU sync
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # recorded automatically
    optimizer.step()                                         # counted automatically
session.close()
```

- **Mixed precision with `torch.amp.GradScaler`:** pass it, `init(..., scaler=scaler)`. Steps the scaler skips on overflow are then still counted, each record says whether the step was skipped and which loss scale was used, and the summary reports a run of skips or a collapsing loss scale.
- **Logging the loss and other scalars:** `session.log` takes the values your tracker logs, as numbers or one-element tensors, and attaches them to the step in progress. Call it before `optimizer.step()`; afterwards, the value lands on the next step. Log the unscaled loss, not `scaler.scale(loss)`. With gradient accumulation, log the step's mean once, since a name logged twice keeps the last value.
- **Histograms and attention:** pass `probe=` a fixed batch, for example `probe={"input_ids": fixed_batch}`, to run it through the model every 1,000 steps (`probe_every=`) for activation histograms per block and attention statistics per layer. It runs without the compiler and without gradients, so expect a pause of about one forward pass each time; on a model that does not use `scaled_dot_product_attention`, only histograms are recorded.
- **Two optimizers on one model**, such as Muon and AdamW: steps count training iterations, not optimizer steps. Nothing to change.
- **Resuming from a checkpoint:** pass `start_step=` the step you resume from, or step numbers start at 0 again. If the checkpoint is older than the last recorded step, the steps written again replace the earlier ones in the evidence.
- **`from torch.nn.utils import clip_grad_norm_` at the top of your script** binds the function before `init` wraps it. Either call `torch.nn.utils.clip_grad_norm_`, or pass the returned norm yourself: `session.record_grad_norm(norm)`.

### PyTorch Lightning

Add the callback; it starts TensorQuill when fitting starts, after Lightning has moved the model to its device, and stops it when fitting ends or fails:

```python
from tensorquill.lightning import TensorQuillCallback

trainer = L.Trainer(callbacks=[TensorQuillCallback(routers=routers)], logger=MLFlowLogger(...))
```

- **Run ID.** It reads the run from the trainer's logger (`MLFlowLogger.run_id`, or `WandbLogger`'s run) on rank 0 and broadcasts it to every rank, so you do not pass `run_id=`.
- **Steps** are counted from optimizer steps, like Lightning's `global_step`, and start at `trainer.global_step`, so a resumed fit continues the numbering.
- **Loss** is logged as `train/loss` for each optimizer step, summed over `accumulate_grad_batches` microbatches (which Lightning has already divided, so it is their mean). Change the name with `loss_name=`, or turn it off with `loss_name=None`.
- **Mixed precision.** With `precision="16-mixed"`, the plugin's `GradScaler` is passed in, so skipped steps are counted and marked.
- **Clipping.** `Trainer(gradient_clip_val=...)` is recorded as `clip_grad_norm`.
- Other keyword arguments go to `tensorquill.init`, for example `capture="full"` or `probe=`.

Tested on CPU with Lightning 2.6.5, on one process and on two with `strategy="ddp_spawn"`. Combining it with `torch.compile` under Lightning has not been tested.

### Megatron-LM

Megatron's mixed-precision optimizer steps float32 copies of the weights, so TensorQuill's automatic step counting does not see it. Count steps yourself and pass the norm `optimizer.step()` returns:

```python
session = tensorquill.init(model_chunk, routers=routers, auto_step=False, start_step=iteration)
...
update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
if grad_norm is not None:
    session.record_grad_norm(grad_norm)
session.log({"train/loss": loss})  # e.g. the reduced loss train_step returns; before session.step
session.step(step=iteration, lr=optimizer.param_groups[0]["lr"])
```

This follows Megatron's `train_step` in `megatron/training/training.py`; it has not been run.

### Hugging Face Trainer (and TRL)

Add the callback; it starts TensorQuill when training starts and stops it when training ends:

```python
from tensorquill.huggingface import TensorQuillCallback

trainer = Trainer(model=model, args=args, train_dataset=data, callbacks=[TensorQuillCallback(capture="full")])
```

Install with `python -m pip install -e 'sdk[huggingface]'`. `SFTTrainer` and the other TRL trainers subclass `Trainer` and take the same callback, but none has been run with it.

- **Run ID.** `Trainer` starts the W&B or MLflow run on the main process only; the callback reads it there and broadcasts it to the other ranks.
- **Steps** are counted from optimizer steps, like `global_step`, and start at `state.global_step`, so a resumed run continues the numbering.
- **Loss** is logged as `train/loss` for each optimizer step: the loss `Trainer` back-propagates, summed over `gradient_accumulation_steps` microbatches, which is their mean. `loss_name=None` turns it off. The interval averages `Trainer` logs every `logging_steps`, and evaluation metrics, are not recorded.
- **Clipping.** The norm before clipping is recorded as `clip_grad_norm`, including with `max_grad_norm=0`, where `Trainer` still computes it.
- **Mixed precision.** Accelerate's `GradScaler`, when it has one, is passed in, so skipped steps are counted and marked. Not run: it needs a GPU.
- **`torch_compile=True`** works: the compiled forward is traced on its first call, after the callback has added its hooks.
- **A failed run.** `Trainer` has no event for an exception. If `trainer.train()` raises, call `callback.close()` to write what was captured.
- Other keyword arguments go to `tensorquill.init`, for example `routers=` or `probe=`.

Tested on CPU, one process, with Transformers 5.23.0 and Accelerate 1.15.0, and with their development versions (5.24.0.dev0, 1.16.0.dev0). Not tested: several ranks, FSDP, DeepSpeed, `nn.DataParallel` (several GPUs in one process), and a GPU.

### Hugging Face Accelerate, in your own loop

```python
from tensorquill.accelerator import watch_accelerator

session = tensorquill.init(model, scaler=accelerator.scaler)  # before accelerator.prepare if it compiles the model
watch_accelerator(session)
```

- **Clipping.** The norm `accelerator.clip_grad_norm_` returns, before clipping, is recorded as `clip_grad_norm`. Without `watch_accelerator` this depends on the Accelerate version: up to 1.15.0 it is recorded anyway, and the development version (1.16.0.dev0) clips another way and is not.
- **Loss.** The loss passed to `accelerator.backward` is logged as `train/loss` for each optimizer step, as the mean over gradient-accumulation microbatches. `loss_name=None` turns it off; do that if two models train through Accelerate in one process, since every loss passed to `backward` counts.
- **Mixed precision.** Pass `scaler=accelerator.scaler` to `init`; it is `None` when there is no scaler, which is fine.
- **`init` refuses a model that is already compiled**, so call it before `accelerator.prepare` if Accelerate compiles the model.
- **Clipping a single tensor** (`accelerator.clip_grad_norm_(weight, max_norm)`) is not recorded. Accelerate, unlike PyTorch, iterates a bare tensor into its rows and returns 0 for it, so there is no real norm to record. Pass a list (`[weight]`) or `model.parameters()`.
- Watching stops when the session closes. It wraps two `Accelerator` methods for the whole process in the meantime.

Tested on CPU, one process, with Accelerate 1.15.0 and 1.16.0.dev0, with gradient accumulation and a learning-rate scheduler. Not run: mixed precision, several ranks, a GPU.

### DeepSpeed

With ZeRO stage 1 or 2, DeepSpeed's optimizer steps flattened float32 copies of the weights, so automatic step counting sees no step on the model and records nothing. TensorQuill warns after 10 such steps. Count steps yourself:

```python
engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
session = tensorquill.init(engine.module, routers=routers, auto_step=False, start_step=engine.global_steps)
...
loss = engine(batch)
engine.backward(loss)
engine.step()
if engine.is_gradient_accumulation_boundary():  # after step(): the update just happened
    norm = engine.get_global_grad_norm()
    if norm is not None:
        session.record_grad_norm(norm)
    session.log({"train/loss": loss.detach()})
    session.step(step=engine.global_steps - 1)
```

- **Without ZeRO (stage 0)** the default `tensorquill.init(engine.module)` counts steps and reads per-layer gradient norms. DeepSpeed clips with its own function there, so `clip_grad_norm` is not recorded.
- **Per-layer gradient norms and weight checks** are not recorded in the recipe above; `clip_grad_norm` carries DeepSpeed's global norm, which the gradient-spike detector uses.
- **Skipped steps.** `engine.was_step_applied()` is false after an overflow; the recipe still counts the step, but does not mark it `step_skipped`.

Run on CPU, one process, with DeepSpeed 0.20.1 at stages 0, 1 and 2, without gradient accumulation or mixed precision. Not run: stage 3, fp16 or bf16, several ranks, a GPU. The accumulation-boundary line follows DeepSpeed's source and was not run.

### Ray, or several processes outside one process group

Pass the same `run_id=` to every process from the driver, and give each process its own file:

```python
sink = tensorquill.JsonlSink(f"tensorquill-out/{run_id}.{role}{index}.jsonl")
session = tensorquill.init(model, run_id=run_id, sink=sink)
```

Without `sink=`, the file name uses the rank from `torch.distributed`, and processes that share no process group are all rank 0, so with a shared folder they would append to the same file. If only one learner trains the model, running TensorQuill there alone avoids the problem.

### Several GPUs

Every rank writes its own file, `tensorquill-out/<run_id>.rank<rank>.jsonl`. With `replica_group=`, give every rank the same step numbers (the same `start_step`, or the same `step=` to `session.step`): the drift check is a collective scheduled from the step number, and ranks that disagree would wait on each other. Start with rank 0 only if that is simpler. The cross-GPU drift check (`replica_group=`, every 50 steps by default, `replica_every=` to change it) has passed only on two CPU processes. Under tensor parallelism, pass `replica_params=` so only weights that are identical across the group, such as LayerNorm, are compared. Files are written by a background thread, so training does not wait on the filesystem; with many ranks on a shared filesystem such as NFS or Lustre, you can write to node-local disk instead with `sink=tensorquill.JsonlSink(path)`.

## 5. After the run

```bash
python -m tensorquill tensorquill-out/<run_id>.rank0.jsonl --rerun tensorquill-out/<run_id>.rrd
rerun tensorquill-out/<run_id>.rrd
```

The first command prints a summary of what the detectors found and writes `<run_id>.rank0.evidence.json`; for top-k routers recorded before `moe_top_k` existed, add `--top-k <k>`. The second opens the recording in the Rerun viewer. On a cluster, copy the `.rrd` to your laptop and open it there, or run `rerun --web-viewer <file>.rrd` on the node and forward its ports over SSH (the page is served on port 9090 by default). To render on a Linux node without a display driver, install a software Vulkan driver (`mesa-vulkan-drivers` on Ubuntu). Neither has been tested.

What the summary can report:

- **MoE collapse or overload** at a step and layer, with the experts involved.
- **Starved experts**: experts that got almost no tokens over 50 steps. A pretrained model can arrive with some; on Granite, 20 came that way.
- **Gradient spikes**: a step at 5 times the recent median, and in full capture the parameter where it started.
- **Learning rate after a resume** (`resume_lr_change`): at the first step of a resumed attempt, the learning rate differs from the earlier attempt's. An observation, not a verdict: restarts that change the schedule on purpose look the same. Resumes that cannot be compared are listed under `not_checked` in `evidence.json`; `weight_trends` there gives numbers only for each parameter kind's norm change.
- **Cross-GPU drift:** now also carries a seeded projection that catches swapped elements; `replica_projection=False` turns it off if its cost matters (GPU cost not yet measured).
- **Weights not updating** (full capture): parameters with gradients whose float64 norm did not change (a warning, not proof the weights are equal; the incident says whether an optimizer holds them, or that it cannot tell, as under DeepSpeed ZeRO). In pure-bf16 training without float32 master weights, RMSNorm weights near 1.0 show up here because their updates round away.
- **Activation spikes, drift or dead outputs** (full capture): a block whose largest value or spread jumps or drifts far from its own history, or whose outputs turn to zero.
- **Repeated steps, NaN or Inf, cross-GPU drift.**

Each incident also lists what you logged with `session.log` at its step, next to the median of the previous 50 values, so you can see whether the loss moved.

If you streamed to `tensorquill-server`, point the same command at the run's directory in the store, `python -m tensorquill <store>/runs/<run_id>`; add `--rank <n>` for a rank other than 0. The server's `watchdog.json` in that directory is read too: `job_stalled` or `rank_silent` means ranks stopped sending steps, whatever the cause (a hang, a long checkpoint or evaluation pause, a killed job); raise the server's `--silent-seconds` above your longest pause. Add `--watch` to keep it running while the job trains: it re-reads the input every 2 seconds and replaces the evidence file.

If you also run TrainCheck, add `--traincheck <violations_summary.json>` (or the results directory `traincheck-check` wrote): its violated invariants appear as one `traincheck_violation` incident, with TrainCheck's own step numbers.

If a multi-GPU job hangs, enable PyTorch's Flight Recorder dumps (`TORCH_FR_BUFFER_SIZE`, `TORCH_FR_DUMP_TEMP_FILE`, `TORCH_NCCL_ENABLE_MONITORING=1`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`; see the SDK README) and add `--flight-recorder <dump dir>` to the command: `collective_hang` names the ranks that never entered the stuck collective, or says every rank did and it did not complete (a network, NCCL or GPU problem); `collective_desync` says ranks issued different collectives. The dumps are pickles: read only dumps your own job wrote.

The thresholds are first guesses from one model, so treat a report as something to check, not a verdict.

## 6. What to send back

- The `.jsonl` file (or files) and the `.evidence.json`.
- Your job's own log around anything TensorQuill reported, and the tracker run.
- Training time per step with and without TensorQuill, if you can run both.
- In the [failure list](../TensorQuill%20failure%20list.md), fill in "Your own failures": what went wrong in this or earlier runs, how you noticed, how long it took, and whether TensorQuill would have caught it sooner.
