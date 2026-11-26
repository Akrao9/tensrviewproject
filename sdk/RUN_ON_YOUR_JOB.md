# Run TensorQuill on your own training job

This guide adds TensorQuill to a training job you already run. So far TensorQuill has been tested on benchmarks with planted failures. Your job is the test that matters, and it needs one normal training run.

TensorQuill only records and reports: it does not stop your training. A call you make with a wrong argument raises at that call, before anything of it is recorded: `init` with a bad setting, or `session.step` with a `skipped=` that is not `True` or `False` (below).

## 1. Install

From a clone of this repository, in the training job's Python environment:

```bash
python -m pip install -r <(python sdk/packaging/requirements.py rerun)
```

```bash
python -m pip install --no-deps -e sdk
```

The first command needs bash or zsh, for `<(...)`. Two steps are needed for now. The SDK depends on the `tensorquill-server` package, which is not on PyPI yet, so `pip install -e 'sdk[rerun]'` cannot resolve. The first command installs what the SDK needs without that package; the second installs the SDK itself.

Add extras by naming them in the first command, for example `requirements.py rerun flight lightning`:

- `rerun`: the Rerun viewer's SDK, for `--rerun` below. Rerun's `rerun` command and viewer send anonymous usage analytics unless you run `rerun analytics disable` once, as each user on each machine that opens a recording; TensorQuill sends none (see [No telemetry](README.md#no-telemetry)).
- `lightning`: PyTorch Lightning.
- `huggingface`: Transformers and Accelerate.
- `flight`: `pyarrow`, to stream to an ingest server and to run the Rust detectors.

Keep the job's own PyTorch build. TensorQuill needs PyTorch 2.6 or later, and an installed 2.6 or later meets the requirement as it is. CI is set to run the tests on CPU builds of PyTorch 2.6, 2.9, 2.10, 2.14 and 2.15 (`.github/workflows/ci.yml`). NumPy is not required, except by the `rerun` extra.

**Two sets of detectors.** The detectors are the rules that read the records and report incidents. There are two copies of them: the Python detectors, which need nothing more, and the Rust detectors, which run in the `tensorquill-server` binary and need `pyarrow`. Until the package is on PyPI, build the binary from the clone (this needs a Rust toolchain) and say where it is:

```bash
cargo build --release --manifest-path server/Cargo.toml
```

```bash
export TENSORQUILL_SERVER_BIN=$PWD/server/target/release/tensorquill-server
```

Without the binary, the commands in section 5 still work with the Python detectors (one read prints a warning; `--watch` needs `--detectors python`).

## 2. Choose a capture mode

| Your model | Call | Cost measured on Granite |
| --- | --- | --- |
| MoE model | `tensorquill.init(model, routers={...})`: router-only, the default | +0.02% to +0.78% on an RTX PRO 6000; +1.06% to +1.33% on an A100 at one commit and +0.41% to +0.95% at a later one |
| MoE model, while debugging | add `capture="full"` | +4.15% on an RTX PRO 6000 |
| Dense model | `tensorquill.init(model)`: full capture | +4.15% on Granite (RTX PRO 6000); not measured on a dense model |

Router-only records each router's expert load every step. Full capture adds per-block activation statistics, per-parameter gradient norms every step and a weight check every 50 steps. In either mode, if your job clips gradients, TensorQuill records the total gradient norm that clipping already computes. That is enough to detect gradient spikes.

**If your schedule unfreezes layers, use `capture="full"` or add `grads=True`.** In the router-only mode a parameter frozen at `init` and unfrozen later is never recorded, with no warning, also when you call `session.record_grads()` by hand: that call records only the parameters trainable at `init`. Full capture, or `routers=...` with `grads=True`, records such a parameter from the step it is unfrozen, and says so once in a warning.

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

- **The router returns the chosen expert indices** (an integer tensor, shape `[tokens, top_k]` or `[tokens]`): pass `indices=` if it is not the first integer tensor in the output. IBM Granite's `GraniteMoeTopKRouter` returns `(logits, weights, indices)`, so the benchmark uses `tensorquill.Router(32, top_k=8, indices=lambda out: out[2])`. A negative index (Megatron's -1 for no route) is not counted; their number is written as `moe_unassigned`.
- **The router returns a routing map** (a bool tensor `[tokens, n_experts]` beside a float tensor, as Megatron's `TopKRouter` does): it is counted from the map. See the Megatron section below.
- **The router returns only logits** (a float tensor whose last dimension is the number of experts): pass `top_k`, and TensorQuill takes the top-k itself, for example `tensorquill.Router(8, top_k=2)`.

Always set `top_k` for top-k routers: the overload detector needs it. Check the printed shapes rather than assuming a model family's layout.

**When the counts are not the routing.** When TensorQuill takes the top-k of the logits itself, its counts are the model's routing only if the model picks the plain top-k of those same scores. They are wrong when the model adds a per-expert balancing bias used only for the choice (DeepSeek-V3, Motif 3 and Mamoda2.5 do), adds noise to the logits after the hooked module, or applies a capacity limit after it. A healthy model then shows overloaded and dead experts. For a bias you can reach, pass an `indices=` function that applies it, such as `indices=lambda out: (out + bias).topk(top_k, dim=-1).indices`, where `bias` is the model's own bias tensor. For noise or a capacity limit, name the module whose output holds the chosen experts instead. This was checked on a CPU toy with a bias, not on a real model.

A shared expert, which takes every token without the router, is not seen.

```python
routers = {
    f"model.layers.{i}.mlp.router": tensorquill.Router(64, top_k=8)  # your names and sizes
    for i in range(num_layers)
}
```

## 4. Add TensorQuill to the training loop

Call `init` after the model is on its device and before `torch.compile`.

**The run ID.** Every file and every server folder is named by the run ID, so all ranks of one job must share it. Pass your tracker's run ID if TensorQuill cannot find it on its own. It looks for `run_id=`, then `MLFLOW_RUN_ID`, then the active MLflow run, then the active W&B run. Without any of those it uses the launcher's job ID (torchrun's `TORCHELASTIC_RUN_ID`, or SLURM's job and step IDs), and otherwise generates a unique `run-<UTC time>-<6 hex>` ID.

When `torch.distributed` is initialized with several ranks, rank 0 works the ID out, including a `run_id=` given on rank 0 only. Every rank then takes rank 0's ID through the process group's store, so every rank writes under the same name. A rank given another `run_id=` is warned and writes under rank 0's.

Call `init` on every rank. A rank that has not heard from rank 0 within 300 s (or the process group's timeout, if shorter) writes under an ID of its own, with a warning.

If you start several ranks with plain `torchrun` (no `--rdzv-id`, so `TORCHELASTIC_RUN_ID` is `none`) and call `init` before `init_process_group`, nothing is shared yet: each rank generates its own ID, `init` warns, and the job is written as one run per rank. Pass the same `run_id=` on every rank, or call `init` after `init_process_group`.

**Not supported in v1: a restart under a torchrun launch with several agents.** torchrun keeps one store across a restart of its workers. With one agent (`GROUP_WORLD_SIZE` 1: one node, `--standalone` or `--nnodes 1`), each restart exchanges the run ID and the launch under keys of its own, scoped by `TORCHELASTIC_RESTART_COUNT`. With several agents (`GROUP_WORLD_SIZE` above 1: more than one node in the rendezvous) that count is not trusted, since each agent counts its own restarts and a restart for a change of membership counts none, the keys are the same in every start, and a restarted rank other than 0 can take the earlier start's run ID and launch before the new rank 0 has set its own. A run ID generated afresh in each start can then differ between ranks; the same `run_id=` on every rank keeps the ID the same, and the launch can still be the earlier start's (bug list, L25).

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

- **Mixed precision with `torch.amp.GradScaler`:** pass it, `init(..., scaler=scaler)`. Steps the scaler skips on overflow are then still counted. Each record says whether the step was skipped (`step_skipped`) and which loss scale was used (`grad_scale`), and the summary reports a run of skips or a collapsing loss scale. What the detectors do with a skipped step is under "Steps you count yourself, and skipped steps" below.
- **Logging the loss and other scalars:** `session.log` takes the values your tracker logs, as numbers or one-element tensors, and attaches them to the step in progress. Call it before `optimizer.step()`; afterwards, the value lands on the next step. Log the unscaled loss, not `scaler.scale(loss)`. With gradient accumulation, log the step's mean once, since a name logged twice keeps the last value.
- **Histograms and attention:** pass `probe=` a fixed batch, for example `probe={"input_ids": fixed_batch}`, to run it through the model every 1,000 steps (`probe_every=`) for activation histograms per block and attention statistics per layer. It runs without the compiler and without gradients, so expect a pause of about one forward pass each time. On a model that does not use `scaled_dot_product_attention`, only histograms are recorded.
- **Two optimizers on one model**, such as Muon and AdamW: steps count training iterations, not optimizer steps. Nothing to change.
- **Resuming from a checkpoint:** pass `start_step=` the step you resume from, or step numbers start at 0 again. If the checkpoint is older than the last recorded step, the steps written again replace the earlier ones in the evidence. Keep the run ID as well: pass the same `run_id=`, or let the tracker or the launcher give it. A resume under another run ID is a second run with a file of its own, and the resume checks do not reach across the two.
- **Clipping groups separately.** With more than one `clip_grad_norm_` call in a step the record says so (`clip_calls`), and `update_spike` leaves out the comparisons that would need the total split by parameter, saying how many under `not_checked` (`clip_groups`). Under `torch.compile` the number of calls is not known and there is no key.
- **Several optimizers.** The first optimizer that ever steps ends each step. If another optimizer sits before it in your loop, pass `init(..., ends_step=optimizer)` with the optimizer whose step closes the iteration; it cannot be combined with `scaler=` or `auto_step=False`. The Lightning, Hugging Face and Accelerate adapters do not pass it.
- **Several optimizers through one `GradScaler`.** A step on which the scaler skipped some and applied others is recorded as a partial step (`step_skipped_part`) and does not count toward `loss_scale_trouble`'s ten skips in a row.
- **A job stopped by SIGTERM** (a scheduler's preemption, or torchrun stopping the other ranks after one failed). TensorQuill installs no signal handler, so with the default handling no `close()` runs: the steps not yet written (those since the last flush, up to `every`, and any still in the writer's queue) are lost and the file reads not closed. To keep them, turn SIGTERM into an exception, `signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))`, and train inside `with session:` or a `try` whose `finally` calls `session.close()`: every step is written and the end marker says `raised`. Under torchrun the agent sends SIGKILL to a worker still alive 30 seconds after its SIGTERM (PyTorch 2.15), and `session.close()` waits up to 60 seconds for a writer that is behind, so a close still waiting then is cut short.
- **`from torch.nn.utils import clip_grad_norm_` at the top of your script** binds the function before `init` wraps it. Either call `torch.nn.utils.clip_grad_norm_`, or pass the returned norm yourself: `session.record_grad_norm(norm)`.

### Steps you count yourself, and skipped steps

Some frameworks step copies of the weights, so TensorQuill's optimizer hooks never see a step (Megatron-LM and DeepSpeed ZeRO, below). Then pass `auto_step=False` to `init` and call `session.step()` yourself:

- **Once per optimizer update that was due.** An accumulation micro-batch is not a step: make no call there.
- **Call `session.step()` before `optimizer.zero_grad()`.** With `auto_step=False` the gradient norms are read inside `session.step()`. A loop that clears the gradients first records no gradient norm (`grad_norms` empty, `grad_norm` null); one whose `zero_grad()` leaves some gradients (a parameter the optimizer does not hold) records those alone, and the total is over them only, with nothing saying so. Call `session.step()` before clearing, or `session.record_grads()` before clearing and `session.step()` after.
- **Say whether the update happened, by keyword.** `session.step(skipped=True)` for an update that was due and not applied, `session.step(skipped=False)` for one that was applied. Leave it out, and the record says only what a scaler given to `init` said. `skipped` takes only `True`, `False` or nothing; any other value, such as `1` or a tensor, raises `TypeError` at the call, before any of the step is recorded. On a closed session, or one whose capture has stopped, the call returns without raising.
- **What is recorded.** `step_skipped` is true when either you or the scaler given to `init` says so. `step_skipped_by` says who: `"caller"`, `"scaler"` or `"both"`. On a skip you reported for an update the scaler was seen to apply, the record also carries `scaler_applied: true`.
- **With a `GradScaler` as well, call `scaler.update()` before `session.step()`.** The scaler's verdict is taken at `scaler.update()`. If `session.step()` comes first, the scaler's outcome for that update is unknown when the step ends: your skip gets no `scaler_applied`, and the scaler's own `step_skipped` and `grad_scale` land on the next record, one step late.
- **What the detectors do with a skipped step,** whoever reported it:
  - Gradient spikes (`grad_spike`) and update-size jumps (`update_spike`) are not judged on it.
  - It cannot advance a stalled-weights streak (`stalled_weights`); a streak that exists carries across it.
  - Only a skip the scaler confirmed counts toward `loss_scale_trouble`. A skip you reported never does, even when your framework's own loss scaler caused it. A loss scale below 1 recorded on the step still raises it.
  - A gradient that is not finite on a skip you reported raises nothing by itself. It is kept in `evidence.json` under `skipped_updates`, which counts the skipped steps by who reported them and, under `non_finite_gradients`, the steps with such a gradient and what was not finite at the first of them. A layer statistic that is not finite still raises `non_finite`.
- **Captures written before `step_skipped_by` existed.** A skipped step without it was written by a scaler and is read as the scaler's. Every kind reads such a capture as before except `stalled_weights`: since a scaler's skips no longer advance a streak, a stall whose streak ran through skipped steps counts fewer steps than it did.
- **Accelerate's `optimizer_step_was_skipped`** is true when any of its optimizers skipped. With one optimizer it is the answer; with several it cannot say that every one skipped, so do not pass it as `skipped`.

### PyTorch Lightning

Add the callback. It starts TensorQuill when fitting starts, after Lightning has moved the model to its device, and stops it when fitting ends or fails:

```python
from tensorquill.lightning import TensorQuillCallback

trainer = L.Trainer(callbacks=[TensorQuillCallback(routers=routers)], logger=MLFlowLogger(...))
```

- **Run ID.** It reads the run from the trainer's logger (`MLFlowLogger.run_id`, or `WandbLogger`'s run) on rank 0 and passes it to `init`, which shares it with every rank through the process group's store, so you do not pass `run_id=`. Under Lightning's XLA strategy, which starts no `torch.distributed` group, the other ranks each write under an ID of their own.
- **Steps** are counted from optimizer steps, like Lightning's `global_step`, and start at `trainer.global_step`, so a resumed fit continues the numbering.
- **Loss** is logged as `train/loss` for each optimizer step, summed over `accumulate_grad_batches` micro-batches (which Lightning has already divided, so it is their mean). Change the name with `loss_name=`, or turn it off with `loss_name=None`.
- **Mixed precision.** With `precision="16-mixed"`, the plugin's `GradScaler` is passed in, so skipped steps are counted and marked.
- **Clipping.** `Trainer(gradient_clip_val=...)` is recorded as `clip_grad_norm`.
- Other keyword arguments go to `tensorquill.init`, for example `capture="full"` or `probe=`.

Tested on CPU with Lightning 2.6.5, on one process and on two with `strategy="ddp_spawn"`. Combining it with `torch.compile` under Lightning has not been tested.

### Megatron-LM

Megatron's mixed-precision optimizer steps float32 copies of the weights, so TensorQuill's automatic step counting does not see it. Count steps yourself, pass the norm `optimizer.step()` returns, and say whether Megatron applied the update:

```python
session = tensorquill.init(model_chunk, routers=routers, auto_step=False, start_step=iteration)
...
update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
# In train_step, after Megatron has combined these over the model-parallel group:
#   update_successful = logical_and_across_model_parallel_group(update_successful, group=mp_group)
#   grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm, group=mp_group)
if grad_norm is not None:
    session.record_grad_norm(grad_norm)
# The iteration is recorded as step `iteration` whether or not it updated the weights; whether it
# did is `skipped=`, and is also logged as a value of its own, 1 or 0. Both before session.step ends it.
session.log({"train/loss": loss, "megatron/update_successful": int(update_successful)})
session.step(step=iteration, lr=optimizer.param_groups[0]["lr"], skipped=not update_successful)
```

This follows Megatron's `train_step` in `megatron/training/training.py` (at `4b11b8168`: `optimizer.step()` at line 3645, the model-parallel AND at 3676, and the learning-rate scheduler stepped only when `update_successful`, at 3691, as `skipped_iter` records). It has not been run on Megatron. The block above is run by `sdk/tests/test_lane_v8_examples.py` with a stand-in optimizer that skips its update.

A step Megatron skipped (Inf or NaN gradients under its loss scaler) is still a step in the record, numbered by the iteration it was attempted at, with `step_skipped: true`, `step_skipped_by: "caller"` and `scalars["megatron/update_successful"]` 0. The detectors read it as "Steps you count yourself, and skipped steps" above says. It is your report, not a scaler's, so it never counts toward `loss_scale_trouble`, even though Megatron's own loss scaler caused it.

With `routers=`, a `TopKRouter`'s output `(probs, routing_map)` is counted from the map, so an assignment dropped at `moe_expert_capacity_factor` is not counted, and with `pad_to_capacity` the padding is left out. With a routing map, `moe_left_out` gives the entries left out for each expert. That was run on a CPU stand-in that follows Megatron's code (Nov 19, 2026), not on Megatron.

### Hugging Face Trainer (and TRL)

Add the callback. It starts TensorQuill when training starts and stops it when training ends:

```python
from tensorquill.huggingface import TensorQuillCallback

trainer = Trainer(model=model, args=args, train_dataset=data, callbacks=[TensorQuillCallback(capture="full")])
```

Install the `huggingface` extra (section 1). `SFTTrainer` and the other TRL trainers subclass `Trainer` and take the same callback, but none has been run with it.

- **Run ID.** `Trainer` starts the W&B or MLflow run on the main process only; `init` reads it there and shares it with the other ranks through the process group's store.
- **Steps** are counted from optimizer steps, like `global_step`, and start at `state.global_step`, so a resumed run continues the numbering.
- **Loss** is logged as `train/loss` for each optimizer step: the loss `Trainer` back-propagates, summed over `gradient_accumulation_steps` micro-batches, which is their mean. `loss_name=None` turns it off. The interval averages `Trainer` logs every `logging_steps`, and evaluation metrics, are not recorded.
- **Clipping.** The norm before clipping is recorded as `clip_grad_norm`, including with `max_grad_norm=0`, where `Trainer` still computes it.
- **Mixed precision.** Accelerate's `GradScaler`, when it has one, is passed in, so skipped steps are counted and marked. Not run: it needs a GPU.
- **`torch_compile=True`** works: the compiled forward is traced on its first call, after the callback has added its hooks.
- **A failed run.** `Trainer` has no event for an exception. The callback closes its session when the process ends, when `callback.close()` is called, or at the next `train()`, and writes the end marker as raised. In a process that goes on after the raise (a notebook, a caught `KeyboardInterrupt`), call `callback.close()` in a `finally:`. A rank that torchrun stops with SIGTERM because another rank failed loses its newest step and leaves no marker.
- Other keyword arguments go to `tensorquill.init`, for example `routers=` or `probe=`.

Tested on CPU, one process, with Transformers 5.23.0 and Accelerate 1.15.0, and with their development versions (5.24.0.dev0, 1.16.0.dev0). An interrupted run was also run on two CPU ranks under torchrun. Not tested: FSDP, DeepSpeed, `nn.DataParallel` (several GPUs in one process), and a GPU.

### Hugging Face Accelerate, in your own loop

```python
from tensorquill.accelerator import watch_accelerator

session = tensorquill.init(model, scaler=accelerator.scaler)  # before accelerator.prepare if it compiles the model
watch_accelerator(session)
```

- **Clipping.** The norm `accelerator.clip_grad_norm_` returns, before clipping, is recorded as `clip_grad_norm`. Without `watch_accelerator` this depends on the Accelerate version: up to 1.15.0 it is recorded anyway, and the development version (1.16.0.dev0) clips another way and is not.
- **Loss.** The loss passed to `accelerator.backward` is logged as `train/loss` for each optimizer step, as the mean over gradient-accumulation micro-batches. `loss_name=None` turns it off; do that if two models train through Accelerate in one process, since every loss passed to `backward` counts.
- **Mixed precision.** Pass `scaler=accelerator.scaler` to `init`; it is `None` when there is no scaler, which is fine.
- **`init` refuses a model that is already compiled**, so call it before `accelerator.prepare` if Accelerate compiles the model.
- **Clipping a single tensor** (`accelerator.clip_grad_norm_(weight, max_norm)`) is not recorded. Accelerate, unlike PyTorch, iterates a bare tensor into its rows and returns 0 for it, so there is no real norm to record. Pass a list (`[weight]`) or `model.parameters()`.
- Watching stops when the session closes. It wraps two `Accelerator` methods for the whole process in the meantime.

Tested on CPU, one process, with Accelerate 1.15.0 and 1.16.0.dev0, with gradient accumulation and a learning-rate scheduler. Not run: mixed precision, several ranks, a GPU.

### DeepSpeed

With ZeRO stage 1 or 2, DeepSpeed's optimizer steps flattened float32 copies of the weights, so automatic step counting sees no step on the model and records nothing. TensorQuill warns after 10 such steps. Count steps yourself, and keep your own count of micro-batches to know when an update was due:

```python
engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
session = tensorquill.init(engine.module, routers=routers, auto_step=False, start_step=engine.global_steps)
accumulation, micro = engine.gradient_accumulation_steps(), 0
...
loss = engine(batch)
engine.backward(loss)
engine.step()
micro += 1
if micro % accumulation == 0:  # your own count: this micro-batch ended an update
    norm = engine.get_global_grad_norm()
    if norm is not None:
        session.record_grad_norm(norm)
    session.log({"train/loss": loss.detach()})
    session.step(step=engine.global_steps - 1, skipped=not engine.was_step_applied())
```

- **Why your own count.** `engine.was_step_applied()` is false at an ordinary accumulation micro-batch as well as after an overflow, so it says whether an update was skipped only at a micro-batch that ended one. And `engine.step()` advances DeepSpeed's own micro-batch counter as it finishes, so `engine.is_gradient_accumulation_boundary()` asked after it is about the next micro-batch (DeepSpeed 0.20.1's `engine.py`, read, not run). The count above fits DeepSpeed's default, where it counts micro-batches itself. With `managed_gradient_accumulation` off, every `engine.step()` is an update, and with `engine.set_gradient_accumulation_boundary()` you decide the boundary yourself: then call `session.step` where your own code ends an update.
- **Skipped steps.** After an overflow the step is still recorded, numbered as DeepSpeed numbers it (its `global_steps` goes up on an overflow too), and marked `step_skipped_by: "caller"`.
- **Without ZeRO (stage 0)** the default `tensorquill.init(engine.module)` counts steps and reads per-layer gradient norms. DeepSpeed clips with its own function there, so `clip_grad_norm` is not recorded.
- **Per-layer gradient norms and weight checks** are not recorded in the recipe above; `clip_grad_norm` carries DeepSpeed's global norm, which the gradient-spike detector uses.

An earlier form of this recipe, without gradient accumulation or `skipped=`, was run on CPU, one process, with DeepSpeed 0.20.1 at stages 0, 1 and 2. The recipe as written here has been run only with a stand-in engine on CPU, not with DeepSpeed. Not run: stage 3, fp16 or bf16, several ranks, a GPU.

### Ray, or several processes outside one process group

Pass the same `run_id=` to every process from the driver, and give each process its own file:

```python
sink = tensorquill.JsonlSink(f"tensorquill-out/{run_id}.{role}{index}.jsonl")
session = tensorquill.init(model, run_id=run_id, sink=sink)
```

Without `sink=`, the file name uses the rank from `torch.distributed`, and processes that share no process group are all rank 0, so with a shared folder they would append to the same file. If only one learner trains the model, running TensorQuill there alone avoids the problem.

### Several GPUs

Every rank writes its own file, `tensorquill-out/<run_id>.rank<rank>.jsonl`. Start with rank 0 only if that is simpler. Files are written by a background thread, so training does not wait on the filesystem. With many ranks on a shared filesystem such as NFS or Lustre, you can write to node-local disk instead with `sink=tensorquill.JsonlSink(path)`.

**The cross-GPU drift check** (`replica_group=`) compares the copies of each parameter that data-parallel ranks hold, which should stay equal.

- **How it runs.** Every 50 steps by default (`replica_every=`), each rank records its own observation of its parameters: a float64 norm and sum, a seeded projection that catches swapped elements, and an exact signature. It is not a collective and no rank waits for another. A reader pairs the ranks' observations by step.
- **Give every rank the same step numbers** (the same `start_step`, or the same `step=` to `session.step`). A rank whose numbers disagree leaves those steps `insufficient`, which is not a pass.
- **Tensor parallelism.** Pass `replica_params=` so only weights that are identical across the group, such as LayerNorm, are compared.
- **What it cannot see.** A fault every rank shares, such as a gradient corrupted before or during the reduction, reaches every replica alike, so the replicas stay equal. Under full sharding alone (FSDP full shard, ZeRO stage 3) no parameter is held twice, so there is nothing to compare.
- **Which rank.** Where the ranks differ on a parameter, the result's `attribution` says which ranks stand apart from a strict majority (`verdict` `attributed`, with `outliers`), or why no rank can be named (`verdict` `inconclusive`, with `why`): `two_ranks` (two ranks that differ show a divergence and cannot say which is wrong), `no_outlier` (no rank is further than the tolerance from the median) or `no_majority`. With two ranks no rank is ever named. A named rank stands apart; that does not make it the faulty one, since a fault shared by every other rank would name the healthy one.
- **A rank with no peer** is never said to agree. A group of one rank reads "one rank observed, zero peers compared".
- **The two checks you call yourself are collectives.** `tensorquill.distributed.replica_spread(params, group)` and `session.record_replica_spread(group)` gather from every rank of `group`, unlike the scheduled check. Every rank of the group must call them on the same step, and each call waits for the others as any collective does: a rank that never calls leaves the others waiting until the process group times out (by default 10 minutes on NCCL, where PyTorch then aborts the process, and 30 minutes on Gloo), and a gather that fails once issued raises into your code. Nothing else TensorQuill does on its own after `init` waits for another rank or raises into training.
- **Where it was run.** Comparing the observations was run on 2, 3 and 8 CPU ranks over Gloo, through local files and through a live server. Not on a GPU, and its cost on a GPU is not measured; `replica_projection=False` turns the projection off.

## 5. After the run

```bash
python -m tensorquill tensorquill-out/<run_id>.rank0.jsonl --rerun tensorquill-out/<run_id>.rrd
```

```bash
rerun tensorquill-out/<run_id>.rrd
```

The first command prints a summary of what the detectors found and writes `<run_id>.rank0.evidence.json`. For top-k routers recorded before `moe_top_k` existed, add `--top-k <k>`. It uses the Rust detectors when the binary and `pyarrow` are there (section 1), and otherwise the Python detectors with one warning; `--detectors python` asks for the Python ones without the warning.

The second command opens the recording in the Rerun viewer. On a cluster, copy the `.rrd` to your laptop and open it there, or run `rerun --web-viewer <file>.rrd` on the node and forward its ports over SSH (the page is served on port 9090 by default). `rerun analytics disable` does not reach the web viewer, which keeps its own analytics config in the browser with no on-or-off setting (read in Rerun 0.43.2's source, not traced on the network): to send nothing, open the file in the native viewer after running the command. To render on a Linux node without a display driver, install a software Vulkan driver (`mesa-vulkan-drivers` on Ubuntu). Neither has been tested.

**A record the readers cannot place is refused.** Every record needs a `step`, an integer of 0 or more; `rank` and `attempt`, when present, must be integers of 0 or more, and `schema_version` 0 or 1. A Boolean, a float (also `1.0`), text or `null` is not an integer. Such a record refuses the whole file: the command prints one line, `tensorquill: error: attempt is 1.0 at <file>, line 4: tensorquill reads an integer of 0 or more`, exits with status 2 and writes no evidence. Nothing of the file is judged. The capture writes a `session.step(step=...)` below 0 as given, and the file is then refused this way, so give steps of 0 or more; `init` refuses a `start_step` below 0 before anything is written.

What the summary can report:

- **MoE collapse or overload** at a step and layer, with the experts involved. Read "When the counts are not the routing" (section 3) before calling it a fault.
- **Starved experts**: experts that got almost no tokens over 50 steps. A pretrained model can arrive with some; on Granite, 20 came that way.
- **Gradient spikes**: a step at 5 times the recent median, and in full capture the parameter where it started; also, in full capture, one parameter that stays at 5 times its own median for 3 steps in a row while the total does not move.
- **Learning rate after a resume** (`resume_lr_change`): at the first step of a resumed attempt, the learning rate differs from the earlier attempt's. An observation, not a verdict: restarts that change the schedule on purpose look the same. Resumes that cannot be compared are listed under `not_checked` in `evidence.json`; `weight_trends` there gives numbers only for each parameter kind's norm change.
- **A resume that did not restore a component** (`resume_incomplete`), when you record checkpoints with `session.checkpoint(state)`. For the model's state, `not_identified` ("model state could not be identified") means the rule that looks for the model's entry in the checkpoint found none. That is not the same as `missing`: the entry may be there under a name the rule does not know. Pass the entry you load the model from as `session.checkpoint(state, event="load", model_state=state["model"])` and nothing is inferred. A trainable tensor the checkpoint does not hold is then named in the finding `missing_tensors`, and one both hold with another shape in `shape_differs`; the evidence's `checkpoint_tensors` says `compared` when the tensors were matched and `unavailable`, with why, when they could not be. **Only a load at the start of a later attempt is judged.** A checkpoint loaded inside one attempt (a rollback to the last good checkpoint in the same process) is recorded and compared, and its record names the missing tensors under `checkpoint.load.tensors`, but no detector judges it: no incident, `resume_incomplete` reads `checked`, and the evidence has no `checkpoint_tensors` for it.
- **Weights not updating** (full capture): parameters with gradients whose float64 norm did not change (a warning, not proof the weights are equal; the incident says whether an optimizer holds them, or that it cannot tell, as under DeepSpeed ZeRO). Skipped steps do not count toward it. In pure-bf16 training without float32 master weights, RMSNorm weights near 1.0 show up here because their updates round away.
- **Activation spikes, drift or dead outputs** (full capture): a block whose largest value or spread jumps or drifts far from its own history, or whose outputs turn to zero.
- **Repeated steps, NaN or Inf, cross-GPU drift.** A drift incident names, for each parameter, the ranks that stand apart, or why none can be named (section 4, "Several GPUs").
- **Router indices that are not experts:** a negative index is no assignment and is counted in `moe_unassigned`; an index at or above the number of experts refuses that router, with the reason, and nothing is raised. With a routing map, `moe_left_out` gives the entries left out per expert. A 0 there means the check ran and found none; a missing field means it did not run, or the record is older.
- **Gradients that could not be read** (one on `meta`, for example) are named on the record under `grads_not_read` and in a `not_checked` entry, and the run reads `degraded`. Gradient spikes are then `unavailable`, since the total leaves those parameters out (a spike in the total as read is still reported). Weights not updating is still checked for the parameters that were read and names the others under `parameters_not_judged`. NaN or Inf is still checked on everything that was read.
- **MoE routers recorded with no tokens on every step** were not read: a `not_checked` entry `moe_routing` names them. With no router read, MoE collapse and starved experts are `unavailable` and the run reads `degraded`; with some read, they are judged on those.
- **Capture stopped:** if TensorQuill's own writing fails or falls behind, it stops capturing, warns once and lets training go on. With `endpoint=` the server then shows the rank as `capture_stopped`, not as hung and not as finished, and the run is never called healthy: nothing was judged after the stop.

Each incident also lists what you logged with `session.log` at its step, next to the median of the previous 50 values, so you can see whether the loss moved.

**"No incidents" is not "nothing wrong".** It covers only the kinds that were checked with enough history. The summary's line "N of 33 kinds checked" says how many were, and names those still warming up. A kind that was not configured (a router-only capture has no weight checks, a run without `scaler=` no loss-scale check) or is unavailable was not looked at.

### Reading a run from a server's store

If you streamed to `tensorquill-server`, point the same command at the run's directory in the store: `python -m tensorquill <store>/runs/<run_id>`. Add `--rank <n>` for a rank other than 0, or `--rank all` for every rank (below). Without `--watch` it reads that rank's stored parts and runs the detectors here, as on a local file. From Python, `build_evidence(<run directory>, rank=n)` does the same with the Python detectors; with several ranks in the directory it raises unless you give a rank, because the detectors read one rank's steps.

- **What the server writes.** As the run goes, the server writes `rank=<rank>/incidents.json` from its own detectors for the 26 incident kinds read from a rank's steps. It puts `job_stalled` and `rank_silent` into the run's `watchdog.json` as `incidents`. The command also reads Flight Recorder dumps and TrainCheck results (below), which never reach the server.
- **Stalls.** `job_stalled` or `rank_silent` in `watchdog.json` means ranks stopped sending steps, whatever the cause: a hang, a long checkpoint or evaluation pause, or a killed job. Raise the server's `--silent-seconds` above your longest pause.
- **A rank that reads `reconnecting`.** Its connection dropped less than `--silent-seconds` ago (300 by default): the process was killed or crashed, or the network failed. It is the state of the connection, not a clean bill. A rank that was healthy when it dropped has no severity and is in no incident for that time, and the run reads `active`; after it the rank reads `lost` and the stall is labelled `lost_connection`. A rank that was already silent or unresponsive when it dropped keeps its `warning` or `critical` and stays in the run's `rank_silent` or `job_stalled` while it reconnects, with `stopped_as` in its entry: a hung process that is killed does not make the stall disappear.
- **Rows the server did not store.** The server counts rows it dropped (a null run, rank, step or attempt, or a schema version it does not support) and whole batches it refused (a key column missing or of another type), apart from each other. They appear only on paths that read the server's status: `--rank all` (in its `watchdog` block), `python -m tensorquill runs` and `compare` on a store, the MCP tools, and one rank's evidence built from the server's report under `--watch`. A local `.jsonl` file has neither count, and `tensorquill-server detect --run`, which reads no store, shows neither.

**Watching while the job trains.** Add `--watch` to keep the command running: it looks every 2 seconds and replaces the evidence file when something changed.

- `--watch` needs the Rust detectors (section 1). Without the binary or `pyarrow` it stops before its first look and says what to install; add `--detectors python` to watch with the Python ones.
- The SDK and `tensorquill-server` must be built from the same version (protocol 5). With a binary of another protocol, `--watch` stops before its first look with exit status 2 and says what to install, one read with `--detectors rust` stops too, and one read without `--detectors` uses the Python detectors and says so. A binary you built with `cargo build` must be built again after you upgrade the SDK.
- With the Rust detectors, `--watch` keeps one `tensorquill-server detect --follow` process for the whole watch and sends it only the records appended since the last look. A change that is not an append (late rows, a resume, a rewritten or truncated file, a part replaced or removed, a router not seen before) judges every record again, and so does a change of the thresholds the watcher reads, at once. That process holds the rank's detectors in memory for as long as the watch runs (on a 10,000-step run of 24 routers, about 240 MB resident between looks and 386 MB at its peak, the detector process's own memory, beside the watcher's), and it ends with the watcher, also when the watcher is killed.
- On a server's store it takes the incidents from the server's own report and reads no part, as long as that report covers everything stored. The last line of the summary says so ("current through step N"), or says why the detectors ran here instead: no report, a report that stopped at an earlier step, or other thresholds. `--recompute` always runs the detectors here, for the one rank.
- "No incidents detected" followed by that "current" line means the server's detectors found nothing in what is stored, among the kinds they checked. Followed by "The server has stopped" or "status was last written N s ago", it also means nobody is watching for stalls.

**All ranks at once.** With several ranks, `--watch` follows one (rank 0 unless you pass `--rank <n>`). To see all of them, point it at the run's directory with `--rank all`: `python -m tensorquill <store>/runs/<run_id> --rank all --watch`.

- It reads every rank's `incidents.json` and the run's `watchdog.json`, and keeps one evidence file for the whole run, `<run_id>.all-ranks.evidence.json`. That file has a shape of its own (`"view": "all_ranks"`), not the per-rank `evidence.json`.
- The summary says what is common to all ranks ("ranks 0-63"), what only some ranks show (it names them), and which ranks have no incidents. Incidents with the same kind, step and names are one group; a group keeps one rank's numbers and says whether the others' differ.
- A rank is unknown, with the reason, when its report is behind the stored steps, stale, missing, unreadable or from other thresholds, or its folder holds nothing yet. An unknown rank is never listed as a rank with no incidents. The summary also says when the server has stopped or its status is old. After a restart, ranks that have not reconnected still show their last report, which stays current for what is stored.
- The stalled-job and silent-rank findings are shown once for the run.
- A look reads two small files for each rank and no part: 34 ms for 64 ranks of 10,000 steps on an 8-core laptop. So it never runs the detectors here. `--recompute` and `--rerun` are refused with `--rank all` (use `--rank <n>` for those), a threshold option such as `--top-k` must be the one the server applied, and without `--watch` it makes one look and exits. To judge an unknown rank yourself from its stored parts, run the command with `--rank <n>`.

**Restarts.** A job that restarts is a new attempt of the same run when it keeps its `run_id`. The newest attempt owns each rank, by when the server first heard from it and not by its number, so a process of the earlier attempt that is still alive cannot take the rank back; its late rows are stored and counted.

**Evaluation and checkpoints are not hangs.** With `endpoint=`, mark your evaluation and checkpoint code: `with session.phase("eval"): ...` (the Lightning and Hugging Face callbacks do it for you). The server then measures how long those pauses last, and `watchdog.json` and the summary recommend a `--silent-seconds` above them. The thresholds (silence 300 s, critical 900 s) are uncalibrated starting values.

**Where a silent rank was.** When a rank is reported silent, the summary also says where it was: "Last stack: rank 3: main thread in `all_reduce` (distributed_c10d.py:2940), called from `train_step` (train.py:88), unchanged for 240 s". Once a rank has gone 60 seconds with no step and no `session.tick()`, its heartbeat thread sends the file, line and function names of its Python threads (no source text and no values; `init(stack_seconds=None)` turns it off). This needs nothing set up before the job starts. It shows the Python line that made a call, not what happens inside C or CUDA. A main thread that hangs while holding the interpreter lock stops the heartbeat thread too: that rank reads as `unresponsive` and has no stack.

**A rank that went silent after its capture had stopped.** Such a rank still raises `rank_silent` or `job_stalled` if its heartbeats later stop. The incident then carries `capture_stopped` (the reason, such as `sink_error`, and how long ago) and `capture_last_step`, the last step this attempt of the rank stored, or null when it stored none. TensorQuill cannot tell from that rank whether training itself stalled, and the incident says so.

### Other tools' results

If you also run TrainCheck, add `--traincheck <violations_summary.json>` (or the results directory `traincheck-check` wrote): its violated invariants appear as one `traincheck_violation` incident, with TrainCheck's own step numbers.

If a multi-GPU job hangs, enable PyTorch's Flight Recorder dumps (`TORCH_FR_BUFFER_SIZE`, `TORCH_FR_DUMP_TEMP_FILE`, `TORCH_NCCL_ENABLE_MONITORING=1`, `TORCH_NCCL_DUMP_ON_TIMEOUT=1`; see the SDK README) and add `--flight-recorder <dump dir>` to the command. `collective_hang` names the ranks that never entered the stuck collective, or says every rank did and it did not complete (a network, NCCL or GPU problem); `collective_desync` says ranks issued different collectives. The dumps are pickles: read only dumps your own job wrote.

The thresholds are first guesses from one model, so treat a report as something to check, not a verdict.

## 6. What to send back

- The `.jsonl` file (or files) and the `.evidence.json`.
- Your job's own log around anything TensorQuill reported, and the tracker run.
- Training time per step with and without TensorQuill, if you can run both.
- In the [failure list](../TensorQuill%20failure%20list.md), fill in "Your own failures": what went wrong in this or earlier runs, how you noticed, how long it took, and whether TensorQuill would have caught it sooner.
