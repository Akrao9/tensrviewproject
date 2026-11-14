# Run TensorLens on your own training job

This guide adds TensorLens to a training job you already run. TensorLens has so far been tested on a benchmark with planted failures; your job is the test that matters. It takes about 15 minutes to set up and one normal training run.

## 1. Install

From a clone of this repository, in the training job's Python environment:

```bash
python -m pip install -e 'sdk[rerun]'
```

Keep the job's own PyTorch build. TensorLens needs PyTorch 2.6 or later and has been run with 2.10 and 2.14.

## 2. Choose a capture mode

| Your model | Call | Cost measured on Granite |
| --- | --- | --- |
| MoE model | `tensorlens.init(model, routers={...})`: router-only, the default | +0.02% to +0.78% on an RTX PRO 6000; +1.1% to +1.3% on an A100 |
| MoE model, while debugging | add `capture="full"` | about +4% |
| Dense model | `tensorlens.init(model)`: full capture | about +4% on Granite; not measured on a dense model |

Router-only records each router's expert load every step. Full capture adds per-block activation statistics, per-parameter gradient norms every step and a weight check every 50 steps. In either mode, if your job clips gradients, the total gradient norm is recorded for free, which is enough to detect gradient spikes.

## 3. Find your router modules (MoE models only)

TensorLens needs each router's module name, its number of experts, and how many experts each token picks (`top_k`). Router outputs differ between model families, so run this once on one batch to see what each candidate returns:

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

Then describe each router with `tensorlens.Router`:

- **The router returns the chosen expert indices** (an integer tensor, shape `[tokens, top_k]` or `[tokens]`): pass `indices=` if it is not the first integer tensor in the output. IBM Granite's `GraniteMoeTopKRouter` returns `(logits, weights, indices)`, so the benchmark uses `tensorlens.Router(32, top_k=8, indices=lambda out: out[2])`.
- **The router returns only logits** (a float tensor whose last dimension is the number of experts): pass `top_k` and TensorLens takes the top-k itself, for example `tensorlens.Router(8, top_k=2)`.

Always set `top_k` for top-k routers: the overload detector needs it. Check the printed shapes rather than assuming a model family's layout.

```python
routers = {
    f"model.layers.{i}.mlp.router": tensorlens.Router(64, top_k=8)  # your names and sizes
    for i in range(num_layers)
}
```

## 4. Add TensorLens to the training loop

Call `init` after the model is on its device and before `torch.compile`. Pass your tracker's run ID if TensorLens cannot find it on its own: it looks for `run_id=`, then `MLFLOW_RUN_ID`, then the active MLflow run, then the active W&B run.

### Plain PyTorch

```python
import tensorlens

model.to(device)
session = tensorlens.init(model, routers=routers, run_id=run_id)  # before torch.compile
model = torch.compile(model)

for batch in loader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # recorded automatically
    optimizer.step()                                         # counted automatically
session.close()
```

- **Mixed precision with `torch.amp.GradScaler`:** the scaler skips `optimizer.step()` when gradients overflow, so count steps yourself: `init(..., auto_step=False)` and call `session.step(lr=optimizer.param_groups[0]["lr"])` once per iteration.
- **Resuming from a checkpoint:** pass `start_step=` the step you resume from, or step numbers start at 0 again.
- **`from torch.nn.utils import clip_grad_norm_` at the top of your script** binds the function before `init` wraps it. Either call `torch.nn.utils.clip_grad_norm_`, or pass the returned norm yourself: `session.record_grad_norm(norm)`.

### PyTorch Lightning

There is no TensorLens callback yet. Start the session in `on_fit_start`, after Lightning has moved the model to its device:

```python
class MyModule(L.LightningModule):
    def on_fit_start(self):
        self.tensorlens = tensorlens.init(self, routers=routers, run_id=run_id)

    def on_fit_end(self):
        self.tensorlens.close()
```

Steps are counted from the optimizer, and `Trainer(gradient_clip_val=...)` clipping is recorded. This was tested on CPU with Lightning 2.6.5. With `MLFlowLogger`, pass `run_id=self.logger.run_id`. Combining this with `torch.compile` under Lightning has not been tested.

### Megatron-LM

Megatron's mixed-precision optimizer steps float32 copies of the weights, so TensorLens's automatic step counting does not see it. Count steps yourself and pass the norm `optimizer.step()` returns:

```python
session = tensorlens.init(model_chunk, routers=routers, auto_step=False, start_step=iteration)
...
update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
if grad_norm is not None:
    session.record_grad_norm(grad_norm)
session.step(step=iteration, lr=optimizer.param_groups[0]["lr"])
```

This follows Megatron's `train_step` in `megatron/training/training.py`; it has not been run.

### Several GPUs

Every rank writes its own file, `tensorlens-out/<run_id>.rank<rank>.jsonl`. Start with rank 0 only if that is simpler. The cross-GPU drift check (`replica_group=`) has passed only on two CPU processes.

## 5. After the run

```bash
python -m tensorlens tensorlens-out/<run_id>.rank0.jsonl --rerun tensorlens-out/<run_id>.rrd
rerun tensorlens-out/<run_id>.rrd
```

The first command prints a summary of what the detectors found and writes `<run_id>.rank0.evidence.json`; for top-k routers recorded before `moe_top_k` existed, add `--top-k <k>`. The second opens the recording in the Rerun viewer.

What the summary can report:

- **MoE collapse or overload** at a step and layer, with the experts involved.
- **Starved experts**: experts that got almost no tokens over 50 steps. A pretrained model can arrive with some; on Granite, 20 came that way.
- **Gradient spikes**: a step at 5 times the recent median, and in full capture the parameter where it started.
- **Weights not updating** (full capture): parameters with gradients that the optimizer left exactly unchanged. In pure-bf16 training without float32 master weights, RMSNorm weights near 1.0 show up here because their updates round away.
- **Activation spikes, drift or dead outputs** (full capture): a block whose largest value or spread jumps or drifts far from its own history, or whose outputs turn to zero.
- **Repeated steps, NaN or Inf, cross-GPU drift.**

The thresholds are first guesses from one model, so treat a report as something to check, not a verdict.

## 6. What to send back

- The `.jsonl` file (or files) and the `.evidence.json`.
- Your job's own log around anything TensorLens reported, and the tracker run.
- Training time per step with and without TensorLens, if you can run both.
- In the [failure list](../TensorLens%20failure%20list.md), fill in "Your own failures": what went wrong in this or earlier runs, how you noticed, how long it took, and whether TensorLens would have caught it sooner.
