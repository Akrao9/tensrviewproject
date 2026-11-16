"""Acceptance check for the Hugging Face `Trainer` callback on a dense transformer.

Fine-tunes a small dense model on WikiText-2 through `transformers.Trainer` with
`torch_compile=True`, gradient clipping and (on CUDA) fp16 mixed precision, which puts a
`GradScaler` in the loop. From `--seed-step` it scales part of one middle block's output by
`--factor`, as `dense_outlier_check.py` does: the channels with the largest outputs by default,
or the whole block (`--fault`; see there for why not a few ordinary channels). It fails unless:

- TensorQuill recorded every optimizer step, numbered as `Trainer` numbers them;
- the loss and the gradient norm before clipping are on every step, and agree with the
  interval averages and norms `Trainer` itself logs;
- the activation detector finds the outlier at the seeded step and block;
- there are no graph breaks and no recompiles after the first step.

    python examples/trainer_check.py --out runs/trainer-check

`--toy` uses a tiny random Llama on random tokens, for checking the script on CPU; add
`--backend aot_eager` where Inductor is not set up.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

import tensorquill

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_outlier_check import FAULTS, MODEL, install_outlier, load_batches, outlier_channels  # noqa: E402


def reset_compile_state() -> None:
    """Forget what Dynamo compiled earlier in this process; `reset()` leaves the metrics list alone."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    clear = getattr(torch._dynamo.utils, "clear_compilation_metrics", None)
    if clear is not None:
        clear()


def recompiled_frames(metrics) -> tuple[int, list[str]]:
    """The number of frames Dynamo compiled, and a line for each one it compiled more than once.

    Only forward compiles count. Under Inductor each backward graph is a metric of its own with
    no code object (`co_filename` is None, `is_forward` False): on the first A100 run those
    made one key look compiled many times and `Path(None)` raised.
    """
    frames: dict[tuple, int] = {}
    for m in metrics:
        if getattr(m, "is_forward", True) is False or m.co_filename is None:
            continue
        key = (m.co_filename, m.co_name, m.co_firstlineno)
        frames[key] = frames.get(key, 0) + 1
    lines = [f"{k[1]} ({Path(k[0]).name}:{k[2]}) compiled {n} times" for k, n in frames.items() if n > 1]
    return sum(frames.values()), lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the run")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed-step", type=int, default=250, help="step the outlier starts")
    parser.add_argument("--fault", choices=FAULTS, default="massive",
                        help="what to scale: the largest channels, the whole block, or the --channels named")
    parser.add_argument("--channels", type=int, nargs="+", default=[7, 300],
                        help="for --fault channels; for massive, only how many channels")
    parser.add_argument("--factor", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accumulate", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--log-every", type=int, default=50, help="Trainer's logging_steps")
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], help="default: fp16 on CUDA, else fp32")
    parser.add_argument("--eager", action="store_true", help="do not compile")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend (aot_eager to skip Inductor)")
    parser.add_argument("--toy", action="store_true", help="tiny random Llama on random tokens, CPU allowed")
    args = parser.parse_args()

    from transformers import (
        AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, Trainer, TrainerCallback, TrainingArguments,
    )

    from tensorquill.huggingface import TensorQuillCallback

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if not args.toy and not cuda:
        parser.error("a CUDA GPU is required; use --toy to check the script on CPU")
    if not 50 <= args.seed_step < args.steps:
        parser.error("--seed-step must be at least 50 and before --steps")
    precision = args.precision or ("fp16" if cuda else "fp32")
    if precision != "fp32" and not cuda:
        parser.error("mixed precision needs CUDA")

    args.out.mkdir(parents=True)
    reset_compile_state()
    torch.manual_seed(0)
    micro = args.steps * args.accumulate
    if args.toy:
        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(config)
        args.channels = [c % config.hidden_size for c in args.channels]
        args.batch_size, batches = 2, torch.randint(0, 128, (micro, 2, 32))
    else:
        # float32 weights: fp16 and bf16 come from autocast, as Trainer sets them up.
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
        batches = load_batches(args.model, args.batch_size, args.seq_len, micro, "cpu")
    model.config.use_cache = False
    model.to(device)
    rows = batches.reshape(-1, batches.shape[-1])
    dataset = [{"input_ids": row, "labels": row} for row in rows]  # in order: no shuffling below
    fixed = batches[0].to(device)

    blocks = [name for name, _ in tensorquill.capture._select_layers(model, "blocks")]
    seeded_block = blocks[len(blocks) // 2]
    # Before the Trainer and its callback, so the hook runs before TensorQuill's hook.
    channels = outlier_channels(model, seeded_block, args.fault, args.channels, model.config.hidden_size, fixed)
    flag, handle = install_outlier(
        model.get_submodule(seeded_block), model.config.hidden_size, channels, args.factor, device,
        next(model.parameters()).dtype,
    )

    class SeedOutlier(TrainerCallback):
        def on_step_begin(self, _args, state, control, **kwargs):
            if state.global_step == args.seed_step:
                flag.fill_(True)

    jsonl = args.out / "trainer.jsonl"
    callback = TensorQuillCallback(
        run_id="trainer-check", capture="full", every=50, sink=tensorquill.JsonlSink(jsonl),
        probe=lambda m: m(input_ids=fixed, use_cache=False), probe_every=args.probe_every,
    )
    training_args = TrainingArguments(
        output_dir=str(args.out / "trainer"), max_steps=args.steps, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulate, learning_rate=args.lr, max_grad_norm=args.max_grad_norm,
        logging_steps=args.log_every, report_to="none", save_strategy="no", disable_tqdm=True,
        fp16=precision == "fp16", bf16=precision == "bf16", use_cpu=not cuda,
        torch_compile=not args.eager, torch_compile_backend=None if args.eager else args.backend,
        dataloader_drop_last=True,
    )

    class InOrder(Trainer):
        def _get_train_sampler(self, *a, **k):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    trainer = InOrder(model=model, args=training_args, train_dataset=dataset, callbacks=[SeedOutlier(), callback])
    name = torch.cuda.get_device_name() if cuda else "CPU"
    mode = "eager" if args.eager else args.backend
    print(f"{torch.__version__} on {name} | {'toy Llama' if args.toy else args.model} | {mode} | {precision} | "
          f"clip {args.max_grad_norm:g} | seeding {seeded_block} from step {args.seed_step} "
          f"({args.fault}: {'all channels' if args.fault == 'block' else channels})", flush=True)

    recompiles: list[str] = []
    try:
        with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
            trainer.train()
    finally:
        callback.close()
        handle.remove()
    compiled = 0
    if not args.eager:
        compiled, recompiles = recompiled_frames(torch._dynamo.utils.get_compilation_metrics())

    # ---- checks ----------------------------------------------------------------------
    records = tensorquill.load_records(jsonl)
    history = [e for e in trainer.state.log_history if "loss" in e and "grad_norm" in e]
    failures: list[str] = []

    def check(ok: bool, text: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {text}")
        if not ok:
            failures.append(text)

    print("\n--- Steps, loss and clipping ---")
    steps = [r["step"] for r in records]
    check(steps == list(range(args.steps)) and trainer.state.global_step == args.steps,
          f"{len(records)} records, steps 0 to {args.steps - 1}, matching Trainer's global_step")
    losses = [r.get("scalars", {}).get("train/loss") for r in records]
    norms = [r.get("clip_grad_norm") for r in records]
    check(all(isinstance(v, float) for v in losses), "train/loss recorded on every step")
    check(all(isinstance(v, (float, str)) for v in norms), "clip_grad_norm recorded on every step")
    loss_off = norm_off = 0.0
    for entry in history:
        end = entry["step"]
        window = [v for v in losses[end - args.log_every : end] if isinstance(v, float)]
        if window and math.isfinite(entry["loss"]):
            loss_off = max(loss_off, abs(sum(window) / len(window) - entry["loss"]) / max(abs(entry["loss"]), 1e-8))
        norm = norms[end - 1]
        if isinstance(norm, float) and math.isfinite(entry["grad_norm"]):
            norm_off = max(norm_off, abs(norm - entry["grad_norm"]) / max(abs(entry["grad_norm"]), 1e-8))
    check(bool(history) and loss_off < 0.01,
          f"interval mean of train/loss agrees with Trainer's logged loss (largest difference {100 * loss_off:.3f}%)")
    check(bool(history) and norm_off < 0.01,
          f"clip_grad_norm agrees with Trainer's logged grad_norm (largest difference {100 * norm_off:.3f}%)")
    if precision == "fp16":
        skipped = sum(1 for r in records if r.get("step_skipped"))
        check(all("grad_scale" in r for r in records), f"grad_scale recorded on every step ({skipped} steps skipped)")

    print("\n--- Detection ---")
    evidence = tensorquill.build_evidence(records)
    print(tensorquill.incident_summary(evidence))
    spike = next((i for i in evidence["incidents"] if i["kind"] == "activation_spike"), None)
    # The fault shows in every later block too; the incident must lead with the first, the seeded one.
    hit = spike is not None and spike["step"] == args.seed_step and spike["modules"][0]["module"] == seeded_block
    check(hit, f"activation spike at step {args.seed_step}, first in {seeded_block}")
    if spike is not None and not hit:
        print(f"  First spike: step {spike['step']}, modules {[m['module'] for m in spike['modules'][:5]]}")
    probes = [r["probe"] for r in records if "probe" in r]
    check(bool(probes) and all("error" not in p for p in probes), f"{len(probes)} probe passes, none failed")

    if not args.eager:
        print("\n--- Compile ---")
        breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
        check(breaks == 0, f"{breaks} graph breaks")
        check(compiled > 0 and not recompiles,
              f"{compiled} frames compiled, none more than once" + ("" if not recompiles else ": " + "; ".join(recompiles)))

    print(f"\nCapture: {jsonl}")
    if failures:
        print(f"\n{len(failures)} check(s) failed")
        sys.exit(1)
    print("\nAll checks passed")


if __name__ == "__main__":
    main()
