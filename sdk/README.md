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
- **`moe`** and **`moe_stats`**: tokens routed to each expert, plus router entropy, diversity (`exp(H)/N`), and dead-expert count/fraction, for every router you name (works whether the router outputs integer indices or floating-point logits with `Router(n_experts, top_k=...)`).

## What is built

| Part | State |
| --- | --- |
| Forward hooks into preallocated on-device buffers, marked static | Built, tested on CPU (including AOTAutograd `static_input_indices` check) |
| On-device ring buffer, moved to the CPU every `every` steps, written by a worker thread | Built, tested |
| Step counting through PyTorch's global optimizer hooks, limited to this model's optimizer | Built, tested |
| Gradient norms grouped by device and dtype (`torch._foreach_norm`, with `DTensor.to_local()` and fallback) | Built, tested |
| MoE expert load (`scatter_add_`) and router entropy / diversity (`moe_stats`) | Built, tested |
| Cross-GPU parameter fingerprints and replica spread (`parameter_fingerprints`, `replica_spread`) | Built, unit-tested on drift and sign-flip cases |
| Detectors and Rerun exporter (`build_evidence`, `export_rerun`, `python -m tensorlens`) | Built, tested on MoE collapse, NaN/Inf, and replica drift |
| Run ID from `run_id=`, `MLFLOW_RUN_ID`, the active MLflow run or the active W&B run | Built, tested with stand-in tracker modules |
| No graph breaks and no recompiles under `torch.compile` | Tested with `eager`, `aot_eager`, and AOTAutograd static-input checks |

## What is not built or not verified

- **Nothing has run on a GPU.** The Inductor backend, `mode="reduce-overhead"` and the CUDA-graph skip check are untested on CUDA hardware, and the training overhead is unmeasured. The buffers do work on Apple's MPS device, and AOTAutograd's `static_input_indices` check is verified on CPU.
- **Arrow Flight and the Rust ingest server.** Records go to a JSON Lines file, and Python detectors/exporters turn that file into `evidence.json` and `.rrd`.
- **The probe pass** for histograms and attention (Tier 2) and replay (Tier 3).
- **Multi-rank reporting.** Every rank writes its own file.
- **A PyTorch Lightning callback.** Until then, pass `run_id=` yourself when the run was created through `MlflowClient`.

## Things to know

- **Mixed precision.** `torch.amp.GradScaler` skips `optimizer.step()` when gradients overflow, so the automatic step count misses those steps. Pass `auto_step=False` and call `session.step()` once per iteration.
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
